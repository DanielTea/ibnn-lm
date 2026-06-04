# Copyright 2026. Apache License 2.0.
#
# Regenerate the result figures (figures/*.png) from the measured numbers. All values are the
# exact held-out bits-per-char (lower is better) produced by the harness; see runs/*.json and
# the experiment commands in the README. Run: python make_figures.py
#
# Single seed where noted (the enwik8 run); otherwise mean +/- std over 3 seeds.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

SM   = "#4C72B0"   # standard FFN
IBNN = "#C44E52"   # IBNN neuron
LSTM = "#55A868"   # recurrent baseline
CTRL = "#8172B3"   # parameter-matched control

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})


def _labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2),
                    textcoords="offset points")


# ---------------------------------------------------------------- Figure 1: the main result
# SM vs IBNN held-out BPC across four settings (all comparable; 10%-data regime shown in README).
fig, ax = plt.subplots(figsize=(9, 5.2))
settings = ["char-LM\n(1.1MB, 0.4M params)", "tuned, lite\n(n_iters=1)",
            "tuned, implicit\n(n_iters=3)", "bigger\n(enwik8 25MB, 5.5M)"]
sm_vals  = [2.528, 2.507, 2.507, 2.349]
sm_err   = [0.024, 0.020, 0.020, 0.0]
ib_vals  = [2.524, 2.501, 2.489, 2.359]
ib_err   = [0.013, 0.019, 0.027, 0.0]
x = range(len(settings)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], sm_vals, w, yerr=sm_err, capsize=4, label="standard FFN", color=SM)
b2 = ax.bar([i + w/2 for i in x], ib_vals, w, yerr=ib_err, capsize=4, label="IBNN neuron", color=IBNN)
_labels(ax, b1); _labels(ax, b2)
ax.set_xticks(list(x)); ax.set_xticklabels(settings)
ax.set_ylabel("held-out bits / char  (lower is better)")
ax.set_ylim(2.2, 2.62)
ax.set_title("IBNN neuron vs standard transformer FFN — a tie at every scale\n"
             "identical models, only the FFN neuron changes (+1 scalar/layer)", fontsize=12.5)
ax.legend(loc="upper right", framealpha=0.95)
ax.text(0.012, 0.03, "bigger run is single-seed (no error bar); others are mean ± std over 3 seeds",
        transform=ax.transAxes, fontsize=8.5, style="italic", color="#555")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_ibnn_vs_sm.png")); plt.close(fig)


# ---------------------------------------------------------------- Figure 2: the decisive test
# Does breaking the mean-field symmetry (learned w_ik) help? Param-matched control = sm_wide.
fig, ax = plt.subplots(figsize=(9, 5.2))
names  = ["standard FFN\n(0.42M)", "wide FFN\n(0.62M, control)",
          "IBNN mean-field\n(0.42M)", "IBNN learned w\n(0.62M)"]
vals   = [2.524, 2.468, 2.543, 2.549]
errs   = [0.018, 0.043, 0.014, 0.011]
colors = [SM, CTRL, IBNN, IBNN]
bars = ax.bar(range(len(names)), vals, 0.62, yerr=errs, capsize=4, color=colors)
bars[1].set_hatch("//"); bars[3].set_hatch("//")   # hatch the 0.62M-param models
_labels(ax, bars)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_ylabel("held-out bits / char  (lower is better)")
ax.set_ylim(2.40, 2.60)
ax.axhline(2.468, ls="--", lw=1, color=CTRL, alpha=0.8)
ax.set_title("The decisive test: structure doesn't rescue it\n"
             "a learned coupling is WORSE than spending the same parameters on plain FFN width",
             fontsize=12.5)
ax.annotate("same parameter budget →\nplain width wins by 0.08 bpc",
            xy=(3, 2.549), xytext=(1.75, 2.575), fontsize=9.5, color="#333",
            arrowprops=dict(arrowstyle="->", color="#666"))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_coupling.png")); plt.close(fig)


# ---------------------------------------------------------------- Figure 3: architecture context
# At ~420k params, the architecture matters far more than the neuron model.
fig, ax = plt.subplots(figsize=(7.5, 5.2))
names = ["char-LSTM", "standard FFN\nTransformer", "IBNN\nTransformer"]
vals  = [2.255, 2.524, 2.543]
errs  = [0.015, 0.018, 0.014]
bars = ax.bar(range(len(names)), vals, 0.6, yerr=errs, capsize=4, color=[LSTM, SM, IBNN])
_labels(ax, bars)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_ylabel("held-out bits / char  (lower is better)")
ax.set_ylim(2.1, 2.62)
ax.set_title("Context: at ~0.42M params, architecture > neuron model\n"
             "a plain LSTM beats both Transformers; the two Transformers tie each other",
             fontsize=12.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_architecture.png")); plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("  figures/" + f)
