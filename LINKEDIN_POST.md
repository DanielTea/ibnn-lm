# LinkedIn post (draft)

**Suggested images to attach (in order):**
1. `figures/fig1_ibnn_vs_sm.png` — the tie, across every scale
2. `figures/fig2_coupling.png` — the decisive test
3. `figures/fig3_architecture.png` — (optional) architecture context

**Links** (LinkedIn sometimes down-ranks posts with external links in the body — if you care
about reach, put these in the first comment instead):
- Repo: https://github.com/DanielTea/ibnn-lm
- Paper: https://arxiv.org/abs/2605.30370

---

## Post text

I spent some evenings testing whether a brand-new "neuron" from a 2026 paper makes a
Transformer better. Honest answer: it didn't — and the *why* turned out to be the interesting
part. 🧵

Almost every neural net still uses the 1950s point-neuron. A recent paper — Mohedano, Bertalmío
et al., "Updating the standard neuron model in artificial neural networks" — proposes a richer
neuron whose units laterally inhibit each other, and shows real gains on image CNNs. The authors
note that a Transformer version is open future work. So I tried it.

I dropped their neuron into the feed-forward layer of a small GPT and ran a controlled
head-to-head against a standard Transformer FFN: same data, same everything — only the neuron
changes (it adds ~1 extra number per layer). All local on a MacBook (Apple MPS), fully
reproducible.

The result, across full vs scarce data, per-model hyperparameter tuning, the full "implicit"
version, and a 13× scale-up to a 5.5M-param model on enwik8:

➡️ A statistical tie. No measurable benefit. (chart 1)

My first instinct was "I must have wired it in wrong." So I checked: with the new coupling
turned off, the model is *bit-identical* to a standard Transformer; with it on, it's clearly
doing something (its coupling strength trains to ~8× its starting value). The null result is
real, not a bug.

Then the fun part. I guessed the problem was that my coupling was "structureless" — a uniform,
all-to-all interaction. So I gave it a fully *learned* coupling instead. It got **worse** — worse
than simply spending those same parameters on a slightly wider plain FFN. (chart 2)

My best explanation: the paper's gains come from *structured, spatial* coupling between
neighbouring neurons in an image. A Transformer FFN's channels are an unordered set with no such
structure to exploit — and the one axis that *does* have structure (the sequence) is already
handled, far better, by attention.

A few caveats, because they matter:
• This is small-scale and character-level — one particular way of porting the neuron.
• It says nothing about the paper's CNN/image results, which look solid.
• I'd genuinely welcome being shown a better integration — that's why the code is public.

Negative results are still results. Huge credit to the authors for a paper that was fun to
think with, and a good reminder that "works great in CNNs" doesn't automatically transfer to
Transformers — and that it's worth verifying *why*, not just *that*.

Code (Apache-2.0, runs on a laptop, every number reproducible): https://github.com/DanielTea/ibnn-lm
Paper: https://arxiv.org/abs/2605.30370

#MachineLearning #DeepLearning #Transformers #NeuralNetworks #ReproducibleResearch #AIResearch
