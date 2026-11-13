"""Regenerate test_metrics_overview.pdf and oversmoothing_depth_trend.pdf at thesis size from the evaluation CSV."""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
FAMILY = {"Baseline": "#52514e", "U-Net": "#2a78d6", "GCN": "#eb6834", "SNN": "#1baf7a"}
TEXT_W = 6.06

def family(m):
    return "Baseline" if m in ("LinearInterp", "PeriodicInterp", "ZeroFilling") else "U-Net" if m.startswith("UNet") else m.split("_")[0]

m = pd.read_csv(repo / "results/evaluation/model_comparison_metrics.csv")
res = m[m["Sample"] >= 0]
side = re.compile(r"_(P\d|PL\d|S\d|N\d+|E\d+|Full)(_|$)")
raw = [x for x in res["Model"].unique() if not side.search(x)]
g = res[res["Model"].isin(raw)].groupby("Model")
mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
order = list(mean["mse_wedge"].sort_values().index)
print("models:", order)

# ---- test_metrics_overview.pdf
METRICS = [("psnr_wedge", "PSNR, missing views (dB)"), ("ssim_wedge", "SSIM, missing views"),
           ("img_psnr", "PSNR after FBP (dB)"), ("img_ssim", "SSIM after FBP")]
y = np.arange(len(order)); colors = [FAMILY[family(x)] for x in order]
fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, 0.19 * len(order) + 1.0), sharey=True, layout="constrained")
for ax, (col, title) in zip(axes, METRICS):
    ax.barh(y, mean.loc[order, col], xerr=std.loc[order, col], height=0.62, color=colors,
            error_kw={"elinewidth": 0.7, "ecolor": "#898781", "capsize": 1.5})
    ax.set_title(title); ax.grid(axis="x", color="#e1e0d9", linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=2)
axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=6.5); axes[0].invert_yaxis()
present = [f for f in FAMILY if f in {family(x) for x in order}]
fig.legend([plt.Rectangle((0, 0), 1, 1, color=FAMILY[f]) for f in present], present,
           loc="outside lower center", ncol=len(present), frameon=False)
fig.savefig(out / "test_metrics_overview.png", dpi=300)

# ---- oversmoothing_depth_trend.pdf
DEPTHS = [6, 12, 18]
fig, ax = plt.subplots(figsize=(0.62 * TEXT_W, 2.4), layout="constrained")
ax.axvspan(3, 9, color="#f0efec")
ax.text(6, 0.03, "reach-limited", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#52514e")
for fam, dodge in (("GCN", -0.4), ("SNN", 0.4)):
    vals = [res[res["Model"] == f"{fam}_L{L}"]["mse_wedge"] for L in DEPTHS]
    ax.errorbar([L + dodge for L in DEPTHS], [v.mean() for v in vals], yerr=[1.96 * v.sem() for v in vals],
                color=FAMILY[fam], marker="o", markersize=4, linewidth=1.2, capsize=2, label=fam)
ax.set_xticks(DEPTHS); ax.set_xlim(3, 21)
ax.set_xlabel("number of layers"); ax.set_ylabel("wedge MSE (mean ± 95 % CI)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
ax.grid(axis="y", color="#e1e0d9", linewidth=0.6); ax.set_axisbelow(True); ax.legend(frameon=False)
fig.savefig(out / "oversmoothing_depth_trend.pdf")
print("ok")
