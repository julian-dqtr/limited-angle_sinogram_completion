"""Regenerate angular_reach.pdf and angular_reach_p4.pdf at thesis text width (6.06 in) with 7-8 pt text."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "lines.linewidth": 1.3, "pdf.fonttype": 42, "axes.unicode_minus": True,
})
INK, SECONDARY, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
FAMILY = {"Baseline": "#52514e", "U-Net": "#2a78d6", "GCN": "#eb6834", "SNN": "#1baf7a"}
DEPTH_STYLE = {6: "-", 12: "--", 18: ":"}
P4_COLOR = "#4a3aa7"
TEXT_W = 6.06
K, STEP, WIN = 12, 1.0, 25.0

mse = pd.read_csv(repo / "results/evaluation/per_view_mse.csv")
var = pd.read_csv(repo / "results/evaluation/per_view_var.csv")
ang = mse["angle_deg"].to_numpy()
wedge = ~mse["acquired"].to_numpy(dtype=bool)
dist = np.rint(np.maximum(-WIN - ang, ang - WIN)[wedge] / STEP) * STEP
fold = lambda v: pd.Series(np.asarray(v)[wedge]).groupby(dist).mean()
floor = fold(var["GroundTruth"])
d_max = dist.max()

def family(m):
    return "Baseline" if m in ("LinearInterp", "PeriodicInterp", "ZeroFilling") else "U-Net" if m.startswith("UNet") else m.split("_")[0]
def depth(m):
    for tag in m.split("_"):
        if tag.startswith("L") and tag[1:].isdigit(): return int(tag[1:])
    return None
def grid(ax):
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_xlim(0, d_max + STEP)
def reach_line(ax, L, label=True, y=0.04):
    r = L * (K // 2) * STEP
    ax.axvline(r + STEP / 2, color=SECONDARY, linewidth=0.8, linestyle=(0, (1, 2)))
    if label:
        ax.text(r + STEP / 2, y, f" reach of {L} layers ({r:.0f}°)", transform=ax.get_xaxis_transform(),
                va="bottom", fontsize=6.5, color=SECONDARY)

# ---- Figure 6.2: angular_reach.pdf
models = ["UNet2D", "LinearInterp", "GCN_L6", "GCN_L12", "GCN_L18", "SNN_L6", "SNN_L12", "SNN_L18"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, 2.75), layout="constrained")
for m in models:
    st = dict(color=FAMILY[family(m)], linestyle=DEPTH_STYLE.get(depth(m), "-"), label=m)
    c = fold(mse[m]); a1.plot(c.index, c.to_numpy(), **st)
    rho = np.sqrt(fold(var[m]) / floor); a2.plot(rho.index, rho.to_numpy(), **st)
a1.plot(floor.index, floor.to_numpy(), color=INK, linestyle="-.", linewidth=1.0, label="floor (variance of the ground truth)")
reach_line(a1, 6); reach_line(a2, 6, y=0.42)
for ax in (a1, a2): grid(ax); ax.set_xlabel("distance to the acquired window $d$ (deg)")
a1.set_yscale("log"); a1.set_ylabel("MSE per view")
a2.set_ylabel(r"$\rho(d)$"); a2.set_ylim(0, 1.05); a2.axhline(1.0, color=AXIS, linewidth=0.8)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, columnspacing=1.2, handlelength=2.2)
fig.savefig(out / "angular_reach.pdf")

# ---- Figure C.11: angular_reach_p4.pdf, one column per network; top MSE and floor, bottom rho
NETS = ["GCN_L6", "GCN_L12", "GCN_L18", "SNN_L6", "SNN_L12", "SNN_L18"]
fig, axes = plt.subplots(2, 6, figsize=(TEXT_W, 3.0), sharex=True, sharey="row", layout="constrained")
for col, raw in enumerate(NETS):
    fam, L = raw.split("_")[0], depth(raw)
    ax_m, ax_r = axes[0, col], axes[1, col]
    for name, color, ls in ((raw, FAMILY[fam], "-"), (raw + "_P4", P4_COLOR, "--")):
        c = fold(mse[name]); ax_m.plot(c.index, c.to_numpy(), color=color, linestyle=ls, linewidth=1.0)
        rho = np.sqrt(fold(var[name]) / floor); ax_r.plot(rho.index, rho.to_numpy(), color=color, linestyle=ls, linewidth=1.0)
    ax_m.plot(floor.index, floor.to_numpy(), color=INK, linestyle="-.", linewidth=0.8)
    ax_m.set_yscale("log"); ax_r.set_ylim(0, 1.05); ax_r.axhline(1.0, color=AXIS, linewidth=0.6)
    ax_m.set_title(raw, fontsize=7.5)
    for ax in (ax_m, ax_r):
        grid(ax); ax.set_xticks([0, 30, 60]); ax.tick_params(labelsize=6.5, length=2, pad=1)
        if L * K // 2 < d_max: reach_line(ax, L, label=False)
    ax_r.set_xlabel("$d$ (deg)", fontsize=7)
axes[0, 0].set_ylabel("MSE per view", fontsize=7.5); axes[1, 0].set_ylabel(r"$\rho(d)$", fontsize=7.5)
h = [Line2D([], [], color=FAMILY["GCN"]), Line2D([], [], color=FAMILY["SNN"]),
     Line2D([], [], color=P4_COLOR, linestyle="--"), Line2D([], [], color=INK, linestyle="-.", linewidth=0.8),
     Line2D([], [], color=SECONDARY, linewidth=0.8, linestyle=(0, (1, 2)))]
l = ["GCN alone", "SNN alone", "with moment regression", "floor", "reach"]
fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, fontsize=7)
fig.savefig(out / "angular_reach_p4.pdf")
print("ok")
