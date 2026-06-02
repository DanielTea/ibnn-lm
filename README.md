# ibnn-lm

A transformer/language-model fork of the **Implicit Bias Neural Network (IBNN)** neuron from
Mohedano, Batard, Velasco-Salido, De Los Santos Mendoza, Martínez, Levine & Bertalmío,
*Updating the standard neuron model in artificial neural networks* (arXiv, 2026),
upstream code at `github.com/vmg-io-csic/ibnn`.

The upstream implementation targets 2D image convolutions, with the IBNN lateral interaction
realized as a spatial cross-difference convolution. That stack is too image-specific to drop
into a transformer, so this fork re-implements the **essential neuron math** as a
fully-connected layer and wires it into a nanoGPT-style decoder. The authors themselves list
"a Transformer using the updated neuron model" as open future work, so treat this as a
research starting point, not a validated recipe.

## The neuron

Standard transformer FFN hidden unit (the "Standard Model", SM):

    u_i = phi( (W x)_i - b_i )

IBNN hidden unit (this fork):

    z_i = (W x)_i - b_i  -  lambda * (1/D) * sum_k tanh( p * (z_k - z_i) )
    v_i = phi( z_i )

Key choices, following the paper:
- The lateral coupling runs over the **hidden channels** of the FFN, applied **independently
  per token position**. This is automatically causal (no masking needed) and parallel across
  the sequence. Coupling across the token axis would re-invent attention and break causality.
- `w_ik` is uniform (`1/D`), so the new term adds **no weight parameters**. `lambda` is a
  single scalar per layer, optionally trainable (recommended). Verified empirically: an IBNN
  model has exactly `+n_layer` parameters vs its SM twin.
- `z` is implicit. We solve by a damped fixed-point iteration started from the SM
  pre-activation. `num_iters=1` is the cheap **"lite"** layer (purely forward, no solver);
  `num_iters>1` unrolls the solve and is differentiated directly by autograd.

## Files

    ibnn_lm/layers.py      IBNNLinear (the neuron) and IBNNMLP (the FFN block)
    ibnn_lm/model.py       nanoGPT-style GPT; pick ffn="ibnn" or ffn="sm"; SM->IBNN warm-start
    ibnn_lm/train_demo.py  char-level training demo (synthetic corpus by default)

## Quick start

    pip install -r requirements.txt
    # head-to-head at equal parameter count:
    python -m ibnn_lm.train_demo --ffn sm   --steps 300
    python -m ibnn_lm.train_demo --ffn ibnn --num_iters 1 --steps 300
    # data-efficiency probe (the paper's headline claim): shrink the training set
    python -m ibnn_lm.train_demo --ffn ibnn --train_frac 0.3 --steps 300

Both modes train; on the toy corpus the IBNN-lite LM drops from perplexity ~1e18 to single
digits within a few dozen steps.

## Known limitations (read before scaling)

1. **O(D^2) lateral term.** `tanh(z_k - z_i)` over all hidden-unit pairs builds a
   `(B, T, D_ff, D_ff)` tensor. At `d_model=128` / `d_ff=512` with a modest batch this is
   already ~2 GB and will OOM a small box. Keep `d_ff` small, or replace the mean-field term
   with a cheaper approximation, before going wide. This is the main thing standing between
   the toy demo and a real run.
2. **No evidence it scales.** The paper validated CNNs on Fashion-MNIST/SVHN/CIFAR-10. Whether
   the data-efficiency / robustness / anti-memorization gains carry to autoregressive LMs is
   an open empirical question this repo is meant to help answer.
3. **Implicit solve cost.** `num_iters>1` multiplies FFN forward cost by the iteration count.
   The unrolled autograd here is simple but memory-hungry; for real training, swap in proper
   implicit differentiation (IFT) or TorchDEQ behind the same `IBNNLinear` interface.
4. **Stability.** The paper warm-starts IBNN nets from a trained SM surrogate. The analog here
   is `copy_sm_weights_into_ibnn`: train `ffn="sm"`, transfer, then switch lambda on. Expect
   to need this (plus the usual AdamW + grad-clip care) at non-toy scale.

## License

Apache 2.0, matching upstream.
