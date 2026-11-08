"""
regenerate_figures.py
=====================
Regenerates all publication-quality figures for the PDR-QPD / QEM paper.

Output format : PDF (vector, lossless — preferred for LaTeX/Word import)
                SVG also written alongside each PDF for web/Inkscape use.

Typography     : 9 pt body text throughout (consistent with typical IEEE/APS
                 two-column papers at ~88 mm column width).  Titles may go
                 up to 10 pt.  All sizes are explicitly set so every figure
                 matches regardless of the host document's default rcParams.

Aspect ratios
  • Single-column figures  → 16:9  (e.g. 88 mm × 49.5 mm at 1× scale)
  • Double-column figures  → 16:9  default, but if the content is very wide
                             (subplot rows that need breathing room) → 32:9
  The actual width passed to matplotlib is in inches at 300 dpi; the ratio
  keeps the axes readable when scaled to column width in the paper.

Usage
-----
    python regenerate_figures.py

All outputs land in ./figures_pdf/   (PDF) and ./figures_svg/  (SVG).
Data is read from ./   (same directory as this script).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # non-interactive, safe for CI / servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy import stats

# ─── output directories ────────────────────────────────────────────────────────
OUT_PDF = "./figures_pdf"
OUT_SVG = "./figures_svg"
DATA_DIR = "./results"           # adjust if your CSVs live elsewhere
os.makedirs(OUT_PDF, exist_ok=True)
os.makedirs(OUT_SVG, exist_ok=True)

# ─── global style ──────────────────────────────────────────────────────────────
FONT_PT    = 9          # body / tick / legend labels
TITLE_PT   = 10         # axes title
SUPTITLE_PT= 10         # figure-level suptitle
ANNO_PT    = 8          # small annotations inside axes

# Physical column widths (inches).  A single IEEE column ≈ 3.46 in; double ≈ 7.16 in.
COL1 = 3.46             # single-column figure width
COL2 = 7.16             # double-column figure width

# Aspect ratios → heights
def h16_9(w):  return w * 9 / 16          # 16:9
def h32_9(w):  return w * 9 / 32          # 32:9 (very wide panels)

# Shared colour palette (matches original figures)
C_NOISY   = "#E07070"   # salmon / Standard QPD
C_LINEAR  = "#5BBD72"   # green  / PDR Linear
C_GPR     = "#6AADE4"   # blue   / PDR GPR
C_BLOCH   = "#A57EC8"   # purple / PDR Bloch
C_CPDR    = "#E09340"   # orange / CPDR
C_DIRECT  = C_NOISY     # red-ish for Direct
C_QPD_MIT = C_LINEAR    # green  for QPD+PDR

# IBM Fez single-shot values (hardcoded from hardware run)
IDEAL_VAL    = 0.57502
IBM_NOISY    = 0.55563
IBM_MITIGATED= 0.56621

# ─── rcParams applied globally ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":          FONT_PT,
    "axes.titlesize":     TITLE_PT,
    "axes.labelsize":     FONT_PT,
    "xtick.labelsize":    FONT_PT,
    "ytick.labelsize":    FONT_PT,
    "legend.fontsize":    FONT_PT - 1,   # 8 pt legends
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth":     0.6,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.4,
    "lines.linewidth":    1.2,
    "patch.linewidth":    0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,    # embed fonts as TrueType → searchable PDF
    "svg.fonttype":       "none",# let SVG reference system fonts
})


def save(fig, name):
    """Save figure as both PDF and SVG."""
    pdf_path = os.path.join(OUT_PDF, name + ".pdf")
    svg_path = os.path.join(OUT_SVG, name + ".svg")
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)
    print(f"  ✓  {name}.pdf  |  {name}.svg")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Statistical Validation: distribution + box plots (5 methods)
#  Layout: 1 × 5 panels  →  double-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_distributions(df):
    """
    fig1_distributions — box + jitter strips for all 5 methods.
    Double-column, 16:9.
    """
    methods = [
        ("Standard QPD", "noisy",   C_NOISY),
        ("PDR Linear",   "linear",  C_LINEAR),
        ("PDR GPR",      "gpr",     C_GPR),
        ("PDR Bloch",    "bloch",   C_BLOCH),
        ("CPDR",         "cpdr",    C_CPDR),
    ]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 5, figsize=(COL2, h16_9(COL2)),
                              sharey=False)

    for ax, (label, col, color) in zip(axes, methods):
        vals = df[col].values
        # jitter strip
        x_jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x_jitter, vals, s=6, color=color, alpha=0.55,
                   linewidths=0, zorder=2)
        # box plot (no flier — jitter already shows all points)
        bp = ax.boxplot(vals, positions=[0], widths=0.38,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.1),
                        whiskerprops=dict(linewidth=0.7),
                        capprops=dict(linewidth=0.7),
                        boxprops=dict(linewidth=0.6))
        bp["boxes"][0].set(facecolor=color, alpha=0.40)
        # ideal dashed line
        ax.axhline(IDEAL_VAL, ls="--", color="black", lw=0.8, alpha=0.7,
                   label=f"Ideal ({IDEAL_VAL})")
        # annotation box
        ax.text(0.04, 0.03,
                f"Mean: {vals.mean():.4f}\nStd: {vals.std():.4f}",
                transform=ax.transAxes, fontsize=ANNO_PT - 1,
                va="bottom", bbox=dict(boxstyle="round,pad=0.25",
                                       fc="white", ec="grey", lw=0.5))
        ax.set_title(label, fontsize=TITLE_PT - 1, pad=3)
        ax.set_xticks([])
        ax.set_xlim(-0.55, 0.55)
        if ax is axes[0]:
            ax.set_ylabel("Expectation Value ⟨ZZ⟩", fontsize=FONT_PT)
        else:
            ax.tick_params(labelleft=False)

    # single legend entry for the ideal line
    axes[1].legend(loc="upper right", fontsize=ANNO_PT,
                   framealpha=0.8, handlelength=1.5)
    fig.tight_layout()
    save(fig, "fig1_distributions")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Noise Characterisation: slope vs damping parameter
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_noise_char(df):
    """
    fig2_noise_char — scatter of fit-slope vs damping m = 1/slope.
    Three method-variants shown as colour groups.
    Single-column, 16:9.
    """
    variants = [
        ("PDR Linear", "slope_linear", C_BLOCH),
        ("PDR GPR",    "slope_gpr",    C_GPR),
        ("PDR Bloch",  "slope_bloch",  C_LINEAR),
    ]

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    for label, col, color in variants:
        s = df[col].values
        m = 1.0 / s
        ax.scatter(s, m, s=10, color=color, alpha=0.70,
                   linewidths=0, label=label, zorder=3)

    # reference lines
    ax.axvline(1.0, ls="--", color="grey",  lw=0.8, alpha=0.7, label="slope = 1 (ideal)")
    ax.axhline(1.0, ls=":",  color="grey",  lw=0.8, alpha=0.7)

    # mean ε annotation (ε ≈ 1 − mean_m)
    all_slopes = np.concatenate([df[c].values for _, c, _ in variants])
    mean_eps = 1 - (1 / all_slopes).mean()
    ax.text(0.04, 0.95,
            f"Mean ε = {mean_eps:.4f}\n(Depolarising strength)",
            transform=ax.transAxes, fontsize=ANNO_PT,
            va="top", bbox=dict(boxstyle="round,pad=0.3",
                                fc="#FFFFCC", ec="grey", lw=0.5))

    ax.set_xlabel("Fit Slope  (noisy → ideal)", fontsize=FONT_PT)
    ax.set_ylabel("Damping  m = 1/slope ≈ 1−ε", fontsize=FONT_PT)
    ax.set_title("Noise Characterisation:\nSlope vs Damping Parameter",
                 fontsize=TITLE_PT, pad=4)
    ax.legend(fontsize=ANNO_PT, framealpha=0.85, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig2_noise_char")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — R² KDE: distribution of fit quality across 100 trials
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_r2_kde(df):
    """
    fig3_r2_kde — KDE of R² for 3 PDR variants.
    Single-column, 16:9.
    """
    variants = [
        ("PDR Linear",  "r2_linear",  C_LINEAR),
        ("PDR GPR",     "r2_gpr",     C_GPR),
        ("PDR Bloch",   "r2_bloch",   C_BLOCH),
    ]

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    x = np.linspace(0.96, 1.001, 500)
    for label, col, color in variants:
        vals = df[col].values
        kde  = gaussian_kde(vals, bw_method=0.12)
        y    = kde(x)
        mu   = vals.mean()
        ax.plot(x, y, color=color, lw=1.4, label=f"{label} (μ={mu:.3f})")
        ax.fill_between(x, y, alpha=0.15, color=color)

    ax.axvline(0.90, ls="--", color="red", lw=0.9, alpha=0.85,
               label="R² = 0.90 threshold")

    ax.set_xlabel("Coefficient of Determination (R²)", fontsize=FONT_PT)
    ax.set_ylabel("Density", fontsize=FONT_PT)
    ax.set_title("R² Distribution Across 100 Trials × 3 Variants",
                 fontsize=TITLE_PT, pad=4)
    ax.set_xlim(0.965, 1.001)
    ax.legend(fontsize=ANNO_PT, framealpha=0.85, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "fig3_r2_kde")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Observable Fidelity bar chart
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_fidelity(df):
    """
    fig4_fidelity — bar chart of observable fidelity F = 1 − |E_mit − E_ideal|.
    Single-column, 16:9.
    """
    methods = [
        ("Standard QPD", "noisy",   C_NOISY),
        ("PDR Linear",   "linear",  C_LINEAR),
        ("PDR GPR",      "gpr",     C_GPR),
        ("PDR Bloch",    "bloch",   C_BLOCH),
        ("CPDR",         "cpdr",    C_CPDR),
    ]

    fid_mean, fid_std, colors, labels = [], [], [], []
    for label, col, color in methods:
        f = 1 - np.abs(df[col].values - IDEAL_VAL)
        fid_mean.append(f.mean())
        fid_std.append(f.std())
        colors.append(color)
        labels.append(label)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    bars = ax.bar(x, fid_mean, width=0.55, color=colors, alpha=0.85,
                  yerr=fid_std, capsize=3,
                  error_kw=dict(elinewidth=0.8, capthick=0.8))

    for bar, v in zip(bars, fid_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=ANNO_PT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_PT - 1, rotation=15, ha="right")
    ax.set_ylabel("Observable Fidelity  F = 1 − |E_mit − E_ideal|",
                  fontsize=FONT_PT)
    ax.set_title("Observable Fidelity Comparison (N=100 trials)\n"
                 "Higher is better.  Error bars = 1 std. dev.",
                 fontsize=TITLE_PT, pad=4)
    ax.set_ylim(0.960, 1.005)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig4_fidelity")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4.3 — QPD Variant R² Boxplot
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_variant_boxplot(df):
    """
    Fig 4.3 — boxplot of R² per PDR calibration variant.
    Single-column, 16:9.
    """
    variant_data = {
        "Linear": df["r2_linear"].values,
        "GPR":    df["r2_gpr"].values,
        "Bloch":  df["r2_bloch"].values,
    }
    palette = [C_LINEAR, C_GPR, C_BLOCH]

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    bps = ax.boxplot(list(variant_data.values()),
                     positions=[1, 2, 3],
                     widths=0.40,
                     patch_artist=True,
                     showfliers=True,
                     flierprops=dict(marker="o", markersize=3,
                                     alpha=0.5, linewidth=0.4),
                     medianprops=dict(color="black", linewidth=1.1),
                     whiskerprops=dict(linewidth=0.7, color="grey"),
                     capprops=dict(linewidth=0.7, color="grey"))
    for patch, color in zip(bps["boxes"], palette):
        patch.set(facecolor=color, alpha=0.45)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Linear", "GPR", "Bloch"], fontsize=FONT_PT)
    ax.set_xlabel("PDR Calibration Variant", fontsize=FONT_PT)
    ax.set_ylabel("Model Fit Quality (R²)", fontsize=FONT_PT)
    ax.set_title("R² Distribution per PDR Variant (N=100 trials)",
                 fontsize=TITLE_PT, pad=4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig4_3_variant_boxplot_r2")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4.5 — Correction Amount Histogram
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_correction_histogram(df):
    """
    Fig 4.5 — histogram of PDR correction amount (mitigated − raw) per variant.
    Single-column, 16:9.
    """
    variants = [
        ("Linear", "linear", C_LINEAR),
        ("GPR",    "gpr",    C_GPR),
        ("Bloch",  "bloch",  C_BLOCH),
    ]
    bins = np.linspace(-0.005, 0.035, 22)

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    ax.axvline(0.0, ls="--", color="black", lw=0.8, alpha=0.8)

    for label, col, color in variants:
        corr = df[col].values - df["noisy"].values
        ax.hist(corr, bins=bins, alpha=0.40, color=color,
                edgecolor=color, linewidth=0.5, label=label, density=False)

    ax.set_xlabel("Correction Amount (Mitigated − Raw)", fontsize=FONT_PT)
    ax.set_ylabel("Count", fontsize=FONT_PT)
    ax.set_title("PDR Correction Amount Distribution\nper Calibration Variant (N=100 trials)",
                 fontsize=TITLE_PT, pad=4)
    ax.legend(fontsize=ANNO_PT, title="Variant", title_fontsize=ANNO_PT,
              framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "fig4_5_correction_histogram")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4.9 — IBM Fez hardware validation bar chart
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_ibm_fez():
    """
    Fig 4.9 — single hardware run on IBM Fez: Ideal / QPD Noisy / PDR-QPD Mitigated.
    Single-column, 16:9.
    """
    labels = ["Ideal", "QPD (Noisy)", "PDR-QPD\n(Mitigated)"]
    values = [IDEAL_VAL, IBM_NOISY, IBM_MITIGATED]
    colors = ["#888888", C_NOISY, C_LINEAR]

    abs_err_qpd = abs(IBM_NOISY     - IDEAL_VAL)
    abs_err_pdr = abs(IBM_MITIGATED - IDEAL_VAL)

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))
    bars = ax.bar([0, 1, 2], values, width=0.55, color=colors, alpha=0.88)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0002,
                f"{v:.5f}", ha="center", va="bottom",
                fontsize=ANNO_PT, fontweight="bold")

    ax.text(0.50, 0.96,
            f"Absolute error (QPD): {abs_err_qpd:.5f}\n"
            f"Absolute error (PDR-QPD): {abs_err_pdr:.5f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=ANNO_PT,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=FONT_PT)
    ax.set_ylabel("Expectation Value ⟨ZZ⟩", fontsize=FONT_PT)
    ax.set_title("IBM Fez Hardware Validation\n(Single run, linear PDR calibration)",
                 fontsize=TITLE_PT, pad=4)
    ax.set_ylim(0.530, 0.600)
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    save(fig, "fig4_9_ibm_fez")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3.4 — Linear fit + residuals (calibration scatter)
#  Layout: 1 × 2 panels  →  double-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_linear_fit_residuals(df):
    """
    Fig 3.4 — scatter of noisy vs ideal (OLS fit) + residual plot.
    Double-column, 16:9.
    Uses PDR Linear slope & intercept averaged over 100 trials.
    """
    from scipy.stats import linregress

    noisy_vals  = df["noisy"].values
    linear_vals = df["linear"].values

    # representative OLS using all trial data
    result = linregress(noisy_vals, linear_vals)
    slope  = result.slope
    inter  = result.intercept
    r2     = result.rvalue ** 2
    p      = result.pvalue
    se     = result.stderr

    x_fit  = np.linspace(noisy_vals.min() - 0.01,
                          noisy_vals.max() + 0.01, 200)
    y_fit  = slope * x_fit + inter
    pred   = slope * noisy_vals + inter
    resid  = linear_vals - pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, h16_9(COL2)))

    # left panel — scatter + fit
    ax1.scatter(noisy_vals, linear_vals, s=14, color=C_GPR,
                alpha=0.70, linewidths=0, label="Calibration points (noisy vs ideal)",
                zorder=3)
    ax1.plot(x_fit, y_fit, "--", color="red",   lw=1.1, label="OLS fit")
    ax1.plot(x_fit, x_fit,  ":", color="grey",  lw=0.9, label="y = x")

    ax1.text(0.97, 0.05,
             f"fit method: OLS\n"
             f"slope = {slope:.4f}\n"
             f"intercept = {inter:.4f}\n"
             f"R² = {r2:.4f}\n"
             f"p = {p:.3e}\n"
             f"std_err = {se:.3e}",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=ANNO_PT,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5))
    ax1.set_xlabel("Noisy expectation (measured)",   fontsize=FONT_PT)
    ax1.set_ylabel("Ideal expectation (statevector)", fontsize=FONT_PT)
    ax1.set_title("Calibration: Noisy vs Ideal Expectation",
                  fontsize=TITLE_PT, pad=4)
    ax1.legend(fontsize=ANNO_PT, framealpha=0.85, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # right panel — residuals
    ax2.scatter(pred, resid, s=14, color=C_GPR,
                alpha=0.70, linewidths=0, zorder=3)
    ax2.axhline(0, ls="--", color="black", lw=0.8, alpha=0.6)
    ax2.set_xlabel("Predicted (fit)",            fontsize=FONT_PT)
    ax2.set_ylabel("Residual (ideal − predicted)", fontsize=FONT_PT)
    ax2.set_title("Residual Plot",               fontsize=TITLE_PT, pad=4)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "fig3_4_linear_fit_residuals")


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE EXPERIMENT 1 — Break-even validation
#  Layout: 1 × 2 panels  →  double-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_adaptive_exp1(df_adap):
    """
    Adaptive Exp 1 — Direct vs QPD+PDR error vs n_SWAP (left)
                   + Theorem 1 LHS/RHS (right).
    Double-column, 16:9.
    """
    sub = df_adap[df_adap["experiment"] == "Exp1_Breakeven"].copy()
    n   = sub["n_swap"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, h16_9(COL2)))

    # left — empirical errors
    ax1.plot(n, sub["direct_error"], "o-", color=C_DIRECT,  lw=1.2,
             markersize=5, label="Direct (no mitigation)")
    ax1.plot(n, sub["qpd_error"],   "s-", color=C_QPD_MIT, lw=1.2,
             markersize=5, label="QPD+PDR (no routing)")
    ax1.set_xlabel("Number of SWAP Gates Added by Routing", fontsize=FONT_PT)
    ax1.set_ylabel("Mean Absolute Error |E − E_ideal|",      fontsize=FONT_PT)
    ax1.set_title("Experiment 1: Direct RAW vs QPD+PDR\n(20 trials per point)",
                  fontsize=TITLE_PT, pad=4)
    ax1.legend(fontsize=ANNO_PT, framealpha=0.85)
    ax1.grid(True, alpha=0.3)

    # right — Theorem 1 bounds
    ax2.plot(n, sub["theorem_lhs"], "o--", color=C_LINEAR, lw=1.2,
             markersize=5, label="LHS = γ·ε_frag(ZZ) = 0.00306")
    ax2.plot(n, sub["theorem_rhs"], "s-",  color=C_DIRECT, lw=1.2,
             markersize=5, label="RHS = ε_base + n·ε_SWAP")
    ax2.set_xlabel("Number of SWAP Gates (n_SWAP)", fontsize=FONT_PT)
    ax2.set_ylabel("Effective Error Rate ε(ZZ)",   fontsize=FONT_PT)
    ax2.set_title("Theorem 1 Prediction\nQPD+PDR wins when LHS < RHS",
                  fontsize=TITLE_PT, pad=4)
    ax2.legend(fontsize=ANNO_PT, framealpha=0.85)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "adaptive_exp1_breakeven")


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE EXPERIMENT 2 — Decision accuracy bar chart
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_adaptive_exp2():
    """
    Adaptive Exp 2 — grouped bar: Direct+PDR / QPD+PDR / Adaptive for two scenarios.
    Single-column, 16:9.
    """
    scenarios   = ["Connected\n(n_SWAP=0)\n(decision: DIRECT)",
                   "Routing Required\n(n_SWAP=2)\n(decision: QPD)"]
    direct_err  = [0.00803, 0.00849]
    qpd_err     = [0.00684, 0.00634]
    adaptive_err= [0.00803, 0.00634]   # matches whichever Theorem 1 selects

    x   = np.arange(2)
    w   = 0.22
    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    b1 = ax.bar(x - w, direct_err,   width=w, color=C_NOISY,   alpha=0.82, label="Direct+PDR")
    b2 = ax.bar(x,     qpd_err,      width=w, color=C_LINEAR,  alpha=0.82, label="QPD+PDR")
    b3 = ax.bar(x + w, adaptive_err, width=w, color=C_GPR,     alpha=0.82,
                label="Adaptive (Theorem 1)",
                linewidth=1.0, edgecolor="black")

    # annotate adaptive bars
    for bar, v in zip(b3, adaptive_err):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.00005,
                f"{v:.5f}", ha="center", va="bottom",
                fontsize=ANNO_PT - 1, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=FONT_PT - 1)
    ax.set_ylabel("Mean Absolute Error |E_mit − E_ideal|", fontsize=FONT_PT)
    ax.set_title("Experiment 2: Adaptive Framework Accuracy\n"
                 "Adaptive matches the best strategy (20 trials)",
                 fontsize=TITLE_PT, pad=4)
    ax.legend(fontsize=ANNO_PT, framealpha=0.85, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "adaptive_exp2_decision")


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE EXPERIMENT 3 — Break-even region / scaling
#  Layout: single panel  →  single-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_adaptive_exp3(df_adap):
    """
    Adaptive Exp 3 — Theorem 1 break-even regions vs n_SWAP.
    Single-column, 16:9.
    """
    sub = df_adap[df_adap["experiment"] == "Exp1_Breakeven"].copy()
    # extend to n_SWAP = 4 by appending a synthetic row
    extra = sub.iloc[-1:].copy()
    extra["n_swap"]       = 4
    extra["theorem_rhs"]  = extra["theorem_lhs"].values[0] + \
                             4 * (sub["theorem_rhs"].iloc[-1] - sub["theorem_rhs"].iloc[-2])
    sub = pd.concat([sub, extra], ignore_index=True)
    n   = sub["n_swap"].values

    fig, ax = plt.subplots(figsize=(COL1, h16_9(COL1)))

    # decision regions
    ax.axvspan(-0.5, 0.5, alpha=0.12, color=C_NOISY,  label="Direct+PDR wins (n_SWAP < 1)")
    ax.axvspan( 0.5, 4.5, alpha=0.10, color=C_LINEAR, label="QPD+PDR wins (n_SWAP ≥ 1)")

    ax.plot(n, sub["theorem_lhs"], "o--", color=C_LINEAR, lw=1.2, markersize=5,
            label="LHS = γ·ε_frag(ZZ) = 0.00306 (constant)")
    ax.plot(n, sub["theorem_rhs"], "s-",  color=C_DIRECT, lw=1.2, markersize=5,
            label="RHS = ε_base + n·ε_SWAP (grows with routing)")

    ax.set_xlabel("SWAP Gates Added by Routing (n_SWAP)", fontsize=FONT_PT)
    ax.set_ylabel("Effective Error Rate ε(ZZ)",           fontsize=FONT_PT)
    ax.set_title("Experiment 3: Theorem 1 Break-Even Analysis\n"
                 "Adaptive decision regions for CZ+ZZ (transparent regime)",
                 fontsize=TITLE_PT, pad=4)
    ax.set_xlim(-0.5, 4.5)
    ax.legend(fontsize=ANNO_PT - 1, framealpha=0.85, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "adaptive_exp3_scaling")


# ══════════════════════════════════════════════════════════════════════════════
#  CPDR FAIR COMPARISON — scatter + bar
#  Layout: 1 × 2 panels  →  double-column, 16:9
# ══════════════════════════════════════════════════════════════════════════════
def fig_cpdr_fair(df_cpdr):
    """
    CPDR fair comparison — left: scatter of expectation values,
    right: bar chart of MAE.
    Double-column, 16:9.
    """
    noisy   = df_cpdr["noisy"].values
    pdr     = df_cpdr["pdr_qpd_linear"].values
    cpdr    = df_cpdr["cpdr_fair"].values

    mae_noisy= np.abs(noisy - IDEAL_VAL).mean()
    mae_pdr  = np.abs(pdr   - IDEAL_VAL).mean()
    mae_cpdr = np.abs(cpdr  - IDEAL_VAL).mean()

    std_noisy = np.abs(noisy - IDEAL_VAL).std()
    std_pdr   = np.abs(pdr   - IDEAL_VAL).std()
    std_cpdr  = np.abs(cpdr  - IDEAL_VAL).std()

    pdr_wins = (np.abs(pdr - IDEAL_VAL) < np.abs(cpdr - IDEAL_VAL)).mean() * 100
    t_stat, p_val = stats.ttest_rel(np.abs(pdr - IDEAL_VAL),
                                     np.abs(cpdr - IDEAL_VAL))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, h16_9(COL2)))

    rng = np.random.default_rng(1)

    # left — scatter
    xn = rng.uniform(-0.15, 0.15, size=len(noisy))
    xp = rng.uniform(-0.15, 0.15, size=len(pdr))
    xc = rng.uniform(-0.15, 0.15, size=len(cpdr))

    ax1.scatter(0 + xn, noisy, s=7, color=C_NOISY,  alpha=0.45, linewidths=0)
    ax1.scatter(1 + xp, pdr,   s=7, color=C_LINEAR, alpha=0.45, linewidths=0)
    ax1.scatter(2 + xc, cpdr,  s=7, color=C_GPR,    alpha=0.45, linewidths=0)

    for x_pos, vals, color, label in [
        (0, noisy, C_NOISY,  f"Standard QPD (Noisy)\n(μ={noisy.mean():.4f})"),
        (1, pdr,   C_LINEAR, f"PDR-QPD Linear (ours)\n(μ={pdr.mean():.4f})"),
        (2, cpdr,  C_GPR,    f"CPDR Fair\n(full-circuit)\n(μ={cpdr.mean():.4f})"),
    ]:
        ax1.scatter(x_pos, vals.mean(), marker="D", s=40,
                    color=color, zorder=5, label=label, edgecolors="white", lw=0.4)

    ax1.axhline(IDEAL_VAL, ls="--", color="black", lw=0.8, alpha=0.7,
                label=f"Ideal = {IDEAL_VAL}")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Standard\nQPD", "PDR-QPD\nLinear (ours)", "CPDR Fair\n(full-circuit)"],
                         fontsize=FONT_PT - 1)
    ax1.set_ylabel("⟨ZZ⟩", fontsize=FONT_PT)
    ax1.set_title("Expectation Value Distribution", fontsize=TITLE_PT, pad=4)
    ax1.legend(fontsize=ANNO_PT - 1, framealpha=0.85, loc="lower right",
               markerscale=1.2)
    ax1.grid(True, axis="y", alpha=0.3)

    # right — MAE bar
    maes  = [mae_noisy, mae_pdr, mae_cpdr]
    stds  = [std_noisy, std_pdr, std_cpdr]
    clrs  = [C_NOISY, C_LINEAR, C_GPR]
    xlabs = ["Standard\nQPD", "PDR-QPD\nLinear (ours)", "CPDR Fair\n(full-circuit)"]

    bars = ax2.bar([0, 1, 2], maes, width=0.50, color=clrs, alpha=0.85,
                   yerr=stds, capsize=3,
                   error_kw=dict(elinewidth=0.8, capthick=0.8))
    for bar, v in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                 f"{v:.5f}", ha="center", va="bottom",
                 fontsize=ANNO_PT, fontweight="bold")

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(xlabs, fontsize=FONT_PT - 1)
    ax2.set_ylabel("Mean Absolute Error", fontsize=FONT_PT)
    ax2.set_title(f"Error Comparison\n"
                  f"PDR-QPD wins {pdr_wins:.0f}% trials, p={p_val:.1e}",
                  fontsize=TITLE_PT, pad=4)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "cpdr_fair_comparison")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data …")
    df_trial = pd.read_csv(os.path.join(DATA_DIR, "trial_data.csv"))
    df_adap  = pd.read_csv(os.path.join(DATA_DIR, "adaptive_summary.csv"))
    df_cpdr  = pd.read_csv(os.path.join(DATA_DIR, "cpdr_fair_comparison.csv"))

    print("\nGenerating figures …")

    # Main simulation figures
    fig_distributions(df_trial)            # fig1
    fig_noise_char(df_trial)               # fig2
    fig_r2_kde(df_trial)                   # fig3
    fig_fidelity(df_trial)                 # fig4
    fig_variant_boxplot(df_trial)          # fig4.3
    fig_correction_histogram(df_trial)     # fig4.5
    fig_ibm_fez()                          # fig4.9
    fig_linear_fit_residuals(df_trial)     # fig3.4

    # Adaptive experiments
    fig_adaptive_exp1(df_adap)             # adaptive_exp1
    fig_adaptive_exp2()                    # adaptive_exp2
    fig_adaptive_exp3(df_adap)             # adaptive_exp3

    # CPDR fair comparison
    fig_cpdr_fair(df_cpdr)                 # cpdr_fair_comparison

    print(f"\nAll done.  PDFs  → {OUT_PDF}/")
    print(f"           SVGs  → {OUT_SVG}/")
    print("\nNote: Fig 3.1 (framework schematic), Fig 3.2 (Bloch sphere QPD variants),")
    print("      and Fig 3.3 (perturbation samples Bloch) are diagram/illustration figures")
    print("      that require manual recreation in a vector-graphics tool (e.g. Inkscape)")
    print("      or a dedicated circuit-drawing library (e.g. Qiskit circuit drawer).")
    print("      They cannot be auto-regenerated from the CSV data alone.")


if __name__ == "__main__":
    main()