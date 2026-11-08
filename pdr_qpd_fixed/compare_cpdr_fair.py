"""
compare_cpdr_fair.py — PDR-QPD vs CPDR: Fair Head-to-Head Comparison
══════════════════════════════════════════════════════════════════════
Run:  python compare_cpdr_fair.py

WHY THIS REPLACES compare_cpdr_proper.py
─────────────────────────────────────────
The previous "proper CPDR" script calibrated near CLIFFORD points (0, π/2).
For our circuit (RX(0.6), RX(0.8)):
    RX(0.6): nearest Clifford = 0.0,   distance = 0.60 rad
    RX(0.8): nearest Clifford = π/2,   distance = 0.77 rad

This forces CPDR to extrapolate ~0.7 rad outside its training domain,
making the comparison unfair — CPDR was designed for circuits already
near Clifford points.

THIS SCRIPT implements a fair comparison where BOTH methods use the
SAME perturbation strategy (near target angles, δθ ~ U(-0.2, 0.2)).
The ONLY structural difference is:
    PDR-QPD : trains on QPD variant circuits (local, separable)
    CPDR    : trains on the FULL 2-qubit circuit (no QPD)

This isolates the contribution of QPD decomposition + per-variant
calibration as PDR-QPD's distinctive contribution.

Metrics reported (Zhang et al. comparison style):
    - Mean absolute error (MAE)
    - 95% CI containing ideal
    - Fraction of trials PDR-QPD wins
    - Paired t-test p-value
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, csv, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

from config      import THETA_0, THETA_1, IDEAL_ZZ, SHOTS, SEED, OUTPUT_DIR
from qpd_lib     import build_target_circuit, build_qpd_variants, reconstruct_expectation
from executor    import QuantumExecutor
from calibration import generate_calibration_data
from pdr_regression import PDRRegressor

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: perturb full circuit near TARGET angles ───────────────────────────

def _perturb_near_target(qc: QuantumCircuit,
                          rng: np.random.Generator,
                          delta_max: float = 0.2) -> QuantumCircuit:
    """Perturb all RX gates by δθ ~ U(-delta_max, +delta_max)."""
    n_rx   = sum(1 for i in qc.data if i.operation.name == 'rx')
    deltas = rng.uniform(-delta_max, delta_max, size=max(n_rx, 1))
    new_qc = QuantumCircuit(qc.num_qubits)
    rx_c   = 0
    for inst in qc.data:
        gate   = inst.operation
        try:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
        except AttributeError:
            qubits = [qc.qubits.index(q) for q in inst.qubits]
        if gate.name == 'rx' and rx_c < len(deltas):
            new_qc.append(RXGate(gate.params[0] + deltas[rx_c]), qubits)
            rx_c += 1
        else:
            new_qc.append(gate, qubits)
    return new_qc


# ── CPDR (fair): full-circuit calibration near target angles ─────────────────

def run_cpdr_fair(target_qc: QuantumCircuit,
                   executor,
                   K: int = 20,
                   seed: int = 42) -> tuple:
    """
    Fair CPDR: same perturbation range as PDR-QPD, Ridge regression,
    but on the FULL 2-qubit circuit (no QPD decomposition).

    Returns (mitigated_zz, noisy_zz)
    """
    rng      = np.random.default_rng(seed)
    cal_qcs  = [_perturb_near_target(target_qc, rng) for _ in range(K)]
    ideal_tr = np.array([executor.get_ideal_zz(qc) for qc in cal_qcs])

    counts_list = executor.batch_run(cal_qcs)
    noisy_tr    = np.array([
        executor._counts_to_zz(c, target_qc.num_qubits)
        for c in counts_list])

    model = Ridge(alpha=0.1)
    model.fit(noisy_tr.reshape(-1, 1), ideal_tr)

    target_counts = executor.batch_run([target_qc])[0]
    noisy_target  = executor._counts_to_zz(target_counts, target_qc.num_qubits)
    mitigated     = float(model.predict([[noisy_target]])[0])

    return mitigated, noisy_target


# ── PDR-QPD (Linear): per-variant calibration ────────────────────────────────

def run_pdr_qpd(target_qc: QuantumCircuit,
                 executor,
                 K: int = 20,
                 seed: int = 42) -> tuple:
    """
    PDR-QPD with Linear regression per QPD variant.
    Returns (mitigated_zz, noisy_zz_reconstructed)
    """
    variants = build_qpd_variants(target_qc, (0, 1))
    weights  = [w for _, w in variants]
    noisy_r, mit_r = [], []

    for v_idx, (v_qc, _) in enumerate(variants):
        cal  = generate_calibration_data(
            v_qc, executor, mode='scalar', K=K, seed=seed + v_idx)
        reg  = PDRRegressor('linear', seed=seed + v_idx).fit(cal)

        counts   = executor.batch_run([v_qc])[0]
        z0, z1   = executor._counts_to_z_per_qubit(counts, v_qc.num_qubits)
        noisy_zz = z0 * z1

        noisy_r.append(noisy_zz)
        mit_r.append(reg.predict(noisy_zz))

    return (reconstruct_expectation(mit_r, weights),
            reconstruct_expectation(noisy_r, weights))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(N: int = 100):
    print("=" * 65)
    print("PDR-QPD vs CPDR — Fair Comparison (same perturbation strategy)")
    print(f"Ideal <ZZ> = {IDEAL_ZZ:.5f}  |  N={N}  |  Shots={SHOTS}")
    print()
    print("Both methods: δθ ~ U(-0.2, 0.2) near TARGET angles")
    print("Difference:   PDR-QPD uses QPD decomposition + per-variant")
    print("              calibration; CPDR uses full circuit + Ridge")
    print("=" * 65)

    executor = QuantumExecutor(shots=SHOTS, seed=SEED)
    noisy_v, pdr_v, cpdr_v = [], [], []

    for t in range(N):
        target   = build_target_circuit(THETA_0, THETA_1)
        seed_t   = SEED + t

        mit_pdr,  noisy_pdr  = run_pdr_qpd(target,  executor, K=20, seed=seed_t)
        mit_cpdr, _          = run_cpdr_fair(target, executor, K=20, seed=seed_t + 500)

        noisy_v.append(noisy_pdr)
        pdr_v.append(mit_pdr)
        cpdr_v.append(mit_cpdr)

        if (t + 1) % 10 == 0:
            pe = np.mean(np.abs(np.array(pdr_v)  - IDEAL_ZZ))
            ce = np.mean(np.abs(np.array(cpdr_v) - IDEAL_ZZ))
            print(f"  Trial {t+1:3d}/{N}  |  PDR-QPD={pe:.5f}  CPDR(fair)={ce:.5f}")

    noisy = np.array(noisy_v)
    pdr   = np.array(pdr_v)
    cpdr  = np.array(cpdr_v)

    ne = np.abs(noisy - IDEAL_ZZ)
    pe = np.abs(pdr   - IDEAL_ZZ)
    ce = np.abs(cpdr  - IDEAL_ZZ)

    t_stat, p_val = stats.ttest_rel(ce, pe)
    ci_pdr   = stats.t.interval(0.95, len(pdr)-1,  np.mean(pdr),  stats.sem(pdr))
    ci_cpdr  = stats.t.interval(0.95, len(cpdr)-1, np.mean(cpdr), stats.sem(cpdr))
    pdr_wins = np.mean(pe < ce)

    print("\n" + "=" * 65)
    print(f"{'Method':<24} {'MAE':>9} {'Std':>9} {'CI∋Ideal':>10} {'Improve':>9}")
    print("-" * 65)
    for nm, arr, err, ci in [
        ('Standard QPD (Noisy)',    noisy, ne, (None, None)),
        ('PDR-QPD Linear (ours)',   pdr,   pe, ci_pdr),
        ('CPDR Fair (full-circuit)', cpdr,  ce, ci_cpdr),
    ]:
        if ci[0] is None:
            ci_ok = '-'
            imp   = '-'
        else:
            ci_ok = str(ci[0] <= IDEAL_ZZ <= ci[1])
            imp   = f"{(1 - err.mean() / ne.mean()) * 100:.1f}%"
        print(f"  {nm:<22} {err.mean():>9.5f} {err.std():>9.5f} "
              f"{ci_ok:>10} {imp:>9}")

    print(f"\nPDR-QPD beats CPDR: {pdr_wins * 100:.0f}% of trials")
    print(f"Paired t-test: t={t_stat:.2f}, p={p_val:.2e}")
    print("→ Statistically significant (p < 0.05)" if p_val < 0.05
          else "→ Not significant")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"PDR-QPD vs CPDR — Fair Comparison (N={N}, Shots={SHOTS})\n"
        f"Both methods: δθ ~ U(−0.2, 0.2) near target angles",
        fontsize=11, fontweight='bold')

    rng_plot = np.random.default_rng(0)
    for arr, lbl, col in [
        (noisy, 'Standard QPD (Noisy)',      '#e74c3c'),
        (pdr,   'PDR-QPD Linear (ours)',     '#2ecc71'),
        (cpdr,  'CPDR Fair\n(full-circuit)', '#3498db'),
    ]:
        jitter = rng_plot.normal(0, 0.025, len(arr))
        ax1.scatter(jitter, arr, alpha=0.25, color=col, s=10)
        ax1.plot([0], [arr.mean()], 'D', color=col, ms=11,
                 label=f'{lbl}\n(μ={arr.mean():.4f})')

    ax1.axhline(IDEAL_ZZ, color='black', ls='--', lw=2,
                label=f'Ideal = {IDEAL_ZZ:.5f}')
    ax1.set_title('Expectation Value Distribution', fontweight='bold')
    ax1.set_ylabel('⟨ZZ⟩')
    ax1.set_xticks([])
    ax1.legend(fontsize=8)

    methods = ['Standard\nQPD', 'PDR-QPD\nLinear (ours)', 'CPDR Fair\n(full-circuit)']
    means   = [ne.mean(), pe.mean(), ce.mean()]
    stds    = [ne.std(),  pe.std(),  ce.std()]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']
    x       = np.arange(3)
    bars    = ax2.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title(
        f'Error Comparison\n'
        f'PDR-QPD wins {pdr_wins*100:.0f}% trials, p={p_val:.1e}',
        fontweight='bold')
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0002,
                 f'{m:.5f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_fig = os.path.join(OUTPUT_DIR, 'cpdr_fair_comparison.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Figure] → {out_fig}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, 'cpdr_fair_comparison.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'noisy', 'pdr_qpd_linear', 'cpdr_fair'])
        for t, (n, p, c) in enumerate(zip(noisy_v, pdr_v, cpdr_v)):
            w.writerow([t, n, p, c])
    print(f"[CSV]    → {out_csv}")


if __name__ == "__main__":
    main()
