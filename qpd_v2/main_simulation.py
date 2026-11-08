"""
main_simulation.py — PDR-QPD v2 Full 100-Trial Simulation
══════════════════════════════════════════════════════════
Compares five methods head-to-head on the same FakeFez noise model:

    1. Standard QPD      — unmitigated baseline (QPD without regression)
    2. PDR-QPD Linear    — TheilSen + OLS regression per variant
    3. PDR-QPD GPR       — Gaussian Process regression per variant
    4. PDR-QPD Bloch     — Bloch-matrix (Choi-inspired) per-qubit correction
    5. CPDR              — Clifford Perturbation Data Regression (fair comparison,
                           same noise backend, Ridge regression, no QPD)

Output:
    results/trial_data.csv           — per-trial raw results
    results/fig1_distributions.png  — expectation value distributions
    results/fig2_noise_char.png     — slope vs intercept scatter
    results/fig3_r2_kde.png         — R² kernel density
    results/fig4_fidelity.png       — observable fidelity comparison
    results/summary_table.txt       — formatted comparison table

Run:
    python main_simulation.py
"""
import os, time, csv, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')
np.random.seed(42)

from config     import (THETA_0, THETA_1, CUT_PAIR, IDEAL_ZZ, N_TRIALS,
                         SHOTS, SEED, OUTPUT_DIR, FIDELITY_BOUND)
from qpd_lib    import build_target_circuit, build_qpd_variants, reconstruct_expectation
from executor   import QuantumExecutor
from calibration import generate_calibration_data
from pdr_regression import PDRRegressor
from cpdr_baseline  import CPDRBaseline
from metrics    import summarize, print_summary_table, observable_fidelity

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
})

METHOD_COLORS = {
    'Standard QPD' : '#e74c3c',
    'PDR Linear'   : '#2ecc71',
    'PDR GPR'      : '#3498db',
    'PDR Bloch'    : '#9b59b6',
    'CPDR'         : '#e67e22',
}


# ── One trial ────────────────────────────────────────────────────────────────

def run_one_trial(trial_idx: int, executor: QuantumExecutor,
                   cpdr: CPDRBaseline) -> dict:
    """
    Execute one full trial for all methods.

    Returns a dict with all results needed for statistics and figures.
    """
    rng  = np.random.default_rng(SEED + trial_idx)
    target_qc = build_target_circuit(THETA_0, THETA_1)
    variants  = build_qpd_variants(target_qc, CUT_PAIR)
    # variants: list of (QuantumCircuit, float) — 4 tuples

    weights   = [w for _, w in variants]

    # ── Per-variant calibration + target measurement ──────────────────────────
    noisy_raw    = []          # Raw (unmitigated) ZZ per variant
    corr_linear  = []          # PDR-Linear corrected ZZ per variant
    corr_gpr     = []          # PDR-GPR corrected ZZ per variant
    corr_bloch   = []          # PDR-Bloch corrected ZZ per variant

    slopes_linear = []
    slopes_gpr    = []
    slopes_bloch  = []
    r2_linear     = []
    r2_gpr        = []
    r2_bloch      = []

    for v_idx, (v_qc, weight) in enumerate(variants):
        v_seed = SEED + trial_idx * 10 + v_idx

        # ── Calibration data (scalar mode) ───────────────────────────────────
        cal_scalar = generate_calibration_data(
            v_qc, executor, mode='scalar', seed=v_seed)

        # ── Calibration data (bloch mode) ────────────────────────────────────
        cal_bloch = generate_calibration_data(
            v_qc, executor, mode='bloch', seed=v_seed + 100)

        # ── Fit regression models ─────────────────────────────────────────────
        reg_lin = PDRRegressor(mode='linear', seed=v_seed).fit(cal_scalar)
        reg_gpr = PDRRegressor(mode='gpr',    seed=v_seed).fit(cal_scalar)
        reg_bloch = PDRRegressor(mode='bloch', seed=v_seed).fit(cal_bloch)

        slopes_linear.append(reg_lin.fit_slope)
        slopes_gpr.append(reg_gpr.fit_slope)
        slopes_bloch.append(reg_bloch.fit_slope)
        r2_linear.append(reg_lin.r2)
        r2_gpr.append(reg_gpr.r2)
        r2_bloch.append(reg_bloch.r2)

        # ── Target circuit measurement for this variant ───────────────────────
        z_counts = executor.batch_run([v_qc])[0]
        z0, z1   = executor._counts_to_z_per_qubit(z_counts, v_qc.num_qubits)
        noisy_zz = z0 * z1    # Product state: ZZ = Z_q0 × Z_q1

        # ── Linear / GPR correction ───────────────────────────────────────────
        corr_linear.append(reg_lin.predict(noisy_zz))
        corr_gpr.append(reg_gpr.predict(noisy_zz))

        # ── Bloch correction: need X, Y per qubit for the TARGET variant ──────
        x_rots = [[{'qubit': 0, 'basis': 'X'}, {'qubit': 1, 'basis': 'X'}]]
        y_rots = [[{'qubit': 0, 'basis': 'Y'}, {'qubit': 1, 'basis': 'Y'}]]
        x_counts = executor.batch_run([v_qc], basis_rotations=x_rots)[0]
        y_counts = executor.batch_run([v_qc], basis_rotations=y_rots)[0]

        x0, x1 = executor._counts_to_z_per_qubit(x_counts, v_qc.num_qubits)
        y0, y1 = executor._counts_to_z_per_qubit(y_counts, v_qc.num_qubits)

        bv_q0 = np.array([x0, y0, z0])
        bv_q1 = np.array([x1, y1, z1])
        corr_bloch.append(reg_bloch.predict_bloch(bv_q0, bv_q1))

        noisy_raw.append(noisy_zz)

    # ── QPD Reconstruction (weighted sum) ────────────────────────────────────
    noisy_recon  = reconstruct_expectation(noisy_raw,   weights)
    linear_recon = reconstruct_expectation(corr_linear, weights)
    gpr_recon    = reconstruct_expectation(corr_gpr,    weights)
    bloch_recon  = reconstruct_expectation(corr_bloch,  weights)

    # ── CPDR baseline (full circuit, same noise) ──────────────────────────────
    cpdr_result = cpdr.run_trial(target_qc, executor)

    return {
        'trial'         : trial_idx,
        # Expectation values
        'noisy'         : noisy_recon,
        'linear'        : linear_recon,
        'gpr'           : gpr_recon,
        'bloch'         : bloch_recon,
        'cpdr'          : cpdr_result['cpdr_zz'],
        # Regression slopes (average over 4 variants)
        'slope_linear'  : float(np.mean(slopes_linear)),
        'slope_gpr'     : float(np.mean(slopes_gpr)),
        'slope_bloch'   : float(np.mean(slopes_bloch)),
        # R² scores (average over 4 variants)
        'r2_linear'     : float(np.mean(r2_linear)),
        'r2_gpr'        : float(np.mean(r2_gpr)),
        'r2_bloch'      : float(np.mean(r2_bloch)),
        # All per-variant R² for KDE plot
        'r2_linear_all' : r2_linear,
        'r2_gpr_all'    : r2_gpr,
        'r2_bloch_all'  : r2_bloch,
        # Per-variant slopes for scatter plot
        'slopes_linear_all': slopes_linear,
    }


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_distributions(trial_data: list, ideal: float):
    """Figure 1: Expectation value distributions (like paper Fig 4.6 but all methods)."""
    methods = {
        'Standard QPD' : [d['noisy']  for d in trial_data],
        'PDR Linear'   : [d['linear'] for d in trial_data],
        'PDR GPR'      : [d['gpr']    for d in trial_data],
        'PDR Bloch'    : [d['bloch']  for d in trial_data],
        'CPDR'         : [d['cpdr']   for d in trial_data],
    }
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    for ax, (name, vals) in zip(axes, methods.items()):
        arr = np.array(vals)
        ax.scatter(np.zeros(len(arr)) + np.random.normal(0, 0.03, len(arr)),
                   arr, alpha=0.4, color=METHOD_COLORS[name], s=12, zorder=2)
        ax.boxplot(arr, widths=0.4, patch_artist=True,
                   boxprops=dict(facecolor=METHOD_COLORS[name], alpha=0.3),
                   medianprops=dict(color='black', linewidth=2), zorder=3)
        ax.axhline(ideal, color='#2c3e50', linestyle='--',
                   linewidth=1.5, label=f'Ideal ({ideal:.4f})', zorder=4)
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Expectation Value ⟨ZZ⟩' if ax == axes[0] else '')
        ax.set_xticks([])
        ci = np.percentile(arr, [2.5, 97.5])
        ax.annotate(f'Mean: {arr.mean():.4f}\nStd: {arr.std():.4f}',
                    xy=(0.5, 0.02), xycoords='axes fraction',
                    ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Statistical Validation (N={len(trial_data)}),  '
                 f'Ideal = {ideal:.5f}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_distributions.png'),
                bbox_inches='tight')
    plt.close()
    print("[Figure 1] Distribution plot saved.")


def plot_noise_characterization(trial_data: list):
    """Figure 2: Slope vs intercept scatter (noise profile), linear variant."""
    fig, ax = plt.subplots(figsize=(7, 5))
    slopes = []
    for d in trial_data:
        s = d['slopes_linear_all']
        slopes.extend(s)
    slopes = np.array(slopes)

    # For intercepts: approximate from slope and mean data
    # (intercepts are not stored per-variant — use slope range plot)
    damping = 1.0 / np.where(slopes != 0, slopes, np.nan)
    epsilons = 1.0 - damping

    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for v in range(4):
        v_slopes = np.array([d['slopes_linear_all'][v] for d in trial_data])
        v_damp   = 1.0 / v_slopes
        ax.scatter(v_slopes, v_damp, color=colors[v], alpha=0.5,
                   s=20, label=f'Variant {v+1}')

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7, label='m = 1 (ideal)')
    ax.axhline(1.0, color='gray', linestyle=':',  alpha=0.7)
    ax.set_xlabel('Fit Slope  (noisy → ideal)', fontweight='bold')
    ax.set_ylabel('Damping m = 1/slope  ≈ 1−ε', fontweight='bold')
    ax.set_title('Noise Characterisation: Slope vs Damping Parameter\n'
                 'Depolarising regime: slope > 1, damping m < 1',
                 fontweight='bold')
    ax.legend(fontsize=9)
    # Annotate mean epsilon
    eps_mean = float(np.nanmean(epsilons))
    ax.annotate(f'Mean ε = {eps_mean:.4f}\n(Depolarising strength)',
                xy=(0.05, 0.85), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9),
                fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_noise_char.png'),
                bbox_inches='tight')
    plt.close()
    print("[Figure 2] Noise characterisation saved.")


def plot_r2_kde(trial_data: list):
    """Figure 3: KDE of R² scores across 100 trials, all three PDR modes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    mode_data = {
        'PDR Linear' : np.concatenate([d['r2_linear_all'] for d in trial_data]),
        'PDR GPR'    : np.concatenate([d['r2_gpr_all']    for d in trial_data]),
        'PDR Bloch'  : np.concatenate([d['r2_bloch_all']  for d in trial_data]),
    }
    x_grid = np.linspace(0.7, 1.0, 300)
    for name, vals in mode_data.items():
        vals_clean = vals[np.isfinite(vals)]
        if len(vals_clean) > 5:
            kde = gaussian_kde(vals_clean, bw_method=0.15)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.3,
                            color=METHOD_COLORS[name])
            ax.plot(x_grid, kde(x_grid), lw=2,
                    color=METHOD_COLORS[name], label=f'{name} (μ={vals_clean.mean():.3f})')

    ax.axvline(0.90, color='red', linestyle='--', alpha=0.7, label='R² = 0.90 threshold')
    ax.set_xlabel('Coefficient of Determination (R²)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('R² Distribution Across 100 Trials × 4 Variants\n'
                 'Validates the linear noise assumption of PDR',
                 fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_r2_kde.png'),
                bbox_inches='tight')
    plt.close()
    print("[Figure 3] R² KDE saved.")


def plot_fidelity(trial_data: list, ideal: float):
    """Figure 4 (NEW): Observable fidelity comparison bar chart."""
    methods = {
        'Standard QPD' : [d['noisy']  for d in trial_data],
        'PDR Linear'   : [d['linear'] for d in trial_data],
        'PDR GPR'      : [d['gpr']    for d in trial_data],
        'PDR Bloch'    : [d['bloch']  for d in trial_data],
        'CPDR'         : [d['cpdr']   for d in trial_data],
    }
    names, means, errs = [], [], []
    for name, vals in methods.items():
        fids = [observable_fidelity(v, ideal) for v in vals]
        names.append(name)
        means.append(np.mean(fids))
        errs.append(np.std(fids))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=errs, capsize=5,
                  color=[METHOD_COLORS[n] for n in names],
                  alpha=0.8, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Observable Fidelity  F = 1 − |E_mit − E_ideal|',
                  fontweight='bold')
    ax.set_title('Observable Fidelity Comparison (N=100 trials)\n'
                 'Higher is better. Error bars = 1 std. dev.',
                 fontweight='bold')
    ax.set_ylim(0.9, 1.01)
    for bar, m, e in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 0.001,
                f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_fidelity.png'),
                bbox_inches='tight')
    plt.close()
    print("[Figure 4] Fidelity comparison saved.")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(trial_data: list):
    path = os.path.join(OUTPUT_DIR, 'trial_data.csv')
    fieldnames = ['trial', 'noisy', 'linear', 'gpr', 'bloch', 'cpdr',
                  'slope_linear', 'slope_gpr', 'slope_bloch',
                  'r2_linear', 'r2_gpr', 'r2_bloch']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(trial_data)
    print(f"[CSV] Trial data saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PDR-QPD v2 — Full Simulation")
    print(f"Ideal <ZZ> = cos({THETA_0})·cos({THETA_1}) = {IDEAL_ZZ:.5f}")
    print(f"Fidelity bound (Gentle Measurement Lemma): {FIDELITY_BOUND:.4f}")
    print(f"Shots: {SHOTS} | Trials: {N_TRIALS} | Seed: {SEED}")
    print("=" * 60)

    executor = QuantumExecutor(shots=SHOTS, seed=SEED)
    cpdr     = CPDRBaseline(seed=SEED)

    trial_data = []
    t0 = time.time()

    for t in range(N_TRIALS):
        trial_result = run_one_trial(t, executor, cpdr)
        trial_data.append(trial_result)

        if (t + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (t + 1) * (N_TRIALS - t - 1)
            print(f"  Trial {t+1:3d}/{N_TRIALS}  |  "
                  f"Noisy: {trial_result['noisy']:.4f}  "
                  f"Linear: {trial_result['linear']:.4f}  "
                  f"GPR: {trial_result['gpr']:.4f}  "
                  f"Bloch: {trial_result['bloch']:.4f}  "
                  f"CPDR: {trial_result['cpdr']:.4f}  "
                  f"| ETA: {eta:.0f}s")

    print(f"\nAll {N_TRIALS} trials complete in {time.time()-t0:.1f}s")

    # ── Compute statistics ────────────────────────────────────────────────────
    noisy_arr  = np.array([d['noisy']  for d in trial_data])
    linear_arr = np.array([d['linear'] for d in trial_data])
    gpr_arr    = np.array([d['gpr']    for d in trial_data])
    bloch_arr  = np.array([d['bloch']  for d in trial_data])
    cpdr_arr   = np.array([d['cpdr']   for d in trial_data])
    sl_arr     = np.array([d['slope_linear'] for d in trial_data])
    sg_arr     = np.array([d['slope_gpr']    for d in trial_data])
    sb_arr     = np.array([d['slope_bloch']  for d in trial_data])

    all_summaries = [
        summarize(noisy_arr,  noisy_arr, sl_arr, IDEAL_ZZ, 'Standard QPD'),
        summarize(linear_arr, noisy_arr, sl_arr, IDEAL_ZZ, 'PDR Linear'),
        summarize(gpr_arr,    noisy_arr, sg_arr, IDEAL_ZZ, 'PDR GPR'),
        summarize(bloch_arr,  noisy_arr, sb_arr, IDEAL_ZZ, 'PDR Bloch'),
        summarize(cpdr_arr,   noisy_arr,
                  np.array([d.get('slope_linear', 1.0) for d in trial_data]),
                  IDEAL_ZZ, 'CPDR'),
    ]

    print_summary_table(all_summaries)

    # Save summary to text
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary_table(all_summaries)
    with open(os.path.join(OUTPUT_DIR, 'summary_table.txt'), 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())

    # ── Generate all figures ──────────────────────────────────────────────────
    print("Generating figures...")
    plot_distributions(trial_data, IDEAL_ZZ)
    plot_noise_characterization(trial_data)
    plot_r2_kde(trial_data)
    plot_fidelity(trial_data, IDEAL_ZZ)

    save_csv(trial_data)

    print("\nDone. All outputs in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
