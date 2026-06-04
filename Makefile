# IBNN-LM local harness. Run `make help` for the available targets.
#
# All targets use the project-local virtualenv at ./.venv (created by `make setup`), so they
# work without activating it. Override knobs on the command line, e.g.:
#   make train-ibnn STEPS=4000 DATASET=tinyshakespeare

PY := .venv/bin/python
DATASET ?= tinyshakespeare
STEPS ?= 2500
CKPT ?= checkpoints/ibnn_$(DATASET).pt
PROMPT ?= "\n"

.DEFAULT_GOAL := help

.PHONY: help setup sanity train-ibnn train-sm warmstart sample chat benchmark compare data-efficiency clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install torch + deps (uv if present, else python -m venv)
	@bash setup.sh

sanity: ## Run the IBNN neuron correctness checks
	$(PY) -m ibnn_lm.sanity

train-ibnn: ## Train the IBNN language model (override STEPS=, DATASET=)
	$(PY) -m ibnn_lm.train --dataset $(DATASET) --ffn ibnn --num_iters 1 \
		--d_model 256 --d_ff 384 --n_layer 4 --n_head 8 --block_size 128 \
		--batch_size 32 --steps $(STEPS) --dropout 0.1 --patience 8 \
		--eval_interval 250 --sample_interval 500 \
		--out checkpoints/ibnn_$(DATASET).pt

train-sm: ## Train the Standard-Model baseline at matching size (for comparison)
	$(PY) -m ibnn_lm.train --dataset $(DATASET) --ffn sm \
		--d_model 256 --d_ff 384 --n_layer 4 --n_head 8 --block_size 128 \
		--batch_size 32 --steps $(STEPS) --dropout 0.1 \
		--eval_interval 250 --sample_interval 500 \
		--out checkpoints/sm_$(DATASET).pt

warmstart: ## Warm-start an IBNN model from the trained SM baseline (paper's surrogate trick)
	$(PY) -m ibnn_lm.train --dataset $(DATASET) --ffn ibnn --num_iters 1 \
		--init_from checkpoints/sm_$(DATASET).pt \
		--d_model 256 --d_ff 384 --n_layer 4 --n_head 8 --block_size 128 \
		--batch_size 32 --steps $(STEPS) --dropout 0.1 \
		--eval_interval 250 --sample_interval 500 \
		--out checkpoints/ibnn_warmstart_$(DATASET).pt

sample: ## Generate text from a checkpoint (override CKPT=, PROMPT=)
	$(PY) -m ibnn_lm.generate --ckpt $(CKPT) --prompt $(PROMPT) --stream --max_new_tokens 600

chat: ## Interactive generation REPL
	$(PY) -m ibnn_lm.generate --ckpt $(CKPT) --interactive

benchmark: ## Exact held-out bits-per-char / perplexity for a checkpoint (override CKPT=)
	$(PY) -m ibnn_lm.evaluate --ckpt $(CKPT)

compare: ## Controlled IBNN-vs-SM at equal size, 3 seeds, error bars (override STEPS=, DATASET=)
	$(PY) -m ibnn_lm.compare --dataset $(DATASET) --ffns sm ibnn --seeds 0 1 2 \
		--train_fracs 1.0 --steps $(STEPS)

data-efficiency: ## Same comparison across full vs scarce data (the paper's headline claim)
	$(PY) -m ibnn_lm.compare --dataset $(DATASET) --ffns sm ibnn --seeds 0 1 2 \
		--train_fracs 1.0 0.3 0.1 --steps $(STEPS)

tune: ## Per-model LR search + 3-seed final: SM vs IBNN n=1 vs IBNN n=3 (full implicit)
	$(PY) -m ibnn_lm.tune --dataset $(DATASET) --lrs 1e-3 3e-3 1e-2 --seeds 0 1 2

coupling-test: ## Mean-field vs learned IBNN coupling, with a param-matched wide-FFN control
	$(PY) -m ibnn_lm.coupling_test --dataset $(DATASET) --seeds 0 1 2 --steps $(STEPS)

clean: ## Remove checkpoints and run logs (keeps downloaded data and the venv)
	rm -rf checkpoints runs_*.log
