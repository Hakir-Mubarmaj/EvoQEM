"""
compare_cpdr_proper.py — PDR-QPD vs Proper CPDR (Zhang et al. 2024)
════════════════════════════════════════════════════════════════════
Drop this file into your existing qpd_v2 folder. Run:
    python compare_cpdr_proper.py

What this does:
    Implements CPDR correctly — calibration near CLIFFORD points (0, π/2)
    NOT near target angles. Then compares against PDR-QPD Linear.

    For RX(0.6): nearest Clifford = 0  → CPDR trains on RX(0 ± δ)
    For RX(0.8): nearest Clifford = 0  → CPDR trains on RX(0 ± δ)

    PDR-QPD trains on RX(0.6 ± δ) — much closer to the target.
    This is why PDR-QPD should outperform proper CPDR.

Output:
    results/cpdr_comparison.png
    results/cpdr_comparison.csv
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, csv, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate

from config      import THETA_0, THETA_1, IDEAL_ZZ, SHOTS, SEED, OUTPUT_DIR
from qpd_lib     import build_target_circuit, build_qpd_variants, reconstruct_expectation
from executor    import QuantumExecutor
from calibration import generate_calibration_data
from pdr_regression import PDRRegressor

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────

def nearest_clifford(theta: float) -> float:
    """Nearest multiple of π/2 to theta."""
    candidates = [k * np.pi / 2 for k in range(-4, 5)]
    return min(candidates, key=lambda c: abs(theta - c))


def run_cpdr_proper(target_qc, executor, K=20, seed=42):
    """
    Proper CPDR: calibrate near Clifford points (Zhang et al. 2024).
    Uses Ridge regression on full circuit ZZ measurements.
    Returns (mitigated_zz, noisy_zz).
    """
    rng = np.random.default_rng(seed)

    # Clifford center for each RX gate
    centers = [nearest_clifford(inst.operation.params[0])
               for inst in target_qc.data if inst.operation.name == 'rx']

    noisy_tr, ideal_tr, cal_qcs = [], [], []
    for _ in range(K):
        delta  = rng.uniform(-0.2, 0.2, size=len(centers))
        new_qc = QuantumCircuit(target_qc.num_qubits)
        rx_c   = 0
        for inst in target_qc.data:
            gate   = inst.operation
            qubits = [target_qc.find_bit(q).index for q in inst.qubits]
            if gate.name == 'rx' and rx_c < len(centers):
                new_qc.append(RXGate(centers[rx_c] + delta[rx_c]), qubits)
                rx_c += 1
            else:
                new_qc.append(gate, qubits)
        cal_qcs.append(new_qc)
        ideal_tr.append(executor.get_ideal_zz(new_qc))

    counts_list = executor.batch_run(cal_qcs)
    for c in counts_list:
        noisy_tr.append(executor._counts_to_zz(c, target_qc.num_qubits))

    model = Ridge(alpha=0.1)
    model.fit(np.array(noisy_tr).reshape(-1, 1), np.array(ideal_tr))

    tc = executor.batch_run([target_qc])[0]
    noisy_t = executor._counts_to_zz(tc, target_qc.num_qubits)
    return float(model.predict([[noisy_t]])[0]), noisy_t


def run_pdr_qpd(target_qc, executor, K=20, seed=42):
    """PDR-QPD Linear: per-variant calibration near target angles."""
    variants = build_qpd_variants(target_qc, (0, 1))
    weights  = [w for _, w in variants]
    noisy_r, mit_r = [], []
    for v_idx, (v_qc, _) in enumerate(variants):
        cal = generate_calibration_data(v_qc, executor, mode='scalar',
                                         K=K, seed=seed + v_idx)
        reg = PDRRegressor('linear', seed=seed + v_idx).fit(cal)
        counts = executor.batch_run([v_qc])[0]
        z0, z1 = executor._counts_to_z_per_qubit(counts, v_qc.num_qubits)
        nzz    = z0 * z1
        noisy_r.append(nzz)
        mit_r.append(reg.predict(nzz))
    return (reconstruct_expectation(mit_r, weights),
            reconstruct_expectation(noisy_r, weights))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N = 100
    print("=" * 60)
    print("PDR-QPD vs Proper CPDR (Zhang et al. 2024)")
    print(f"Ideal <ZZ> = {IDEAL_ZZ:.5f}  |  N={N}  |  Shots={SHOTS}")

    tgt = build_target_circuit(THETA_0, THETA_1)
    print(f"\nClifford centers (why CPDR extrapolates further):")
    for inst in tgt.data:
        if inst.operation.name == 'rx':
            θ = inst.operation.params[0]
            cf = nearest_clifford(θ)
            print(f"  RX({θ:.1f}): nearest Clifford={cf:.4f}  distance={abs(θ-cf):.4f} rad")
    print("=" * 60)

    executor = QuantumExecutor(shots=SHOTS, seed=SEED)
    noisy_v, pdr_v, cpdr_v = [], [], []

    for t in range(N):
        target = build_target_circuit(THETA_0, THETA_1)
        seed_t = SEED + t
        mit_pdr,  noisy_pdr  = run_pdr_qpd(target, executor, K=20, seed=seed_t)
        mit_cpdr, _          = run_cpdr_proper(target, executor, K=20, seed=seed_t + 500)
        noisy_v.append(noisy_pdr)
        pdr_v.append(mit_pdr)
        cpdr_v.append(mit_cpdr)
        if (t + 1) % 10 == 0:
            pe = np.mean(np.abs(np.array(pdr_v)  - IDEAL_ZZ))
            ce = np.mean(np.abs(np.array(cpdr_v) - IDEAL_ZZ))
            print(f"  Trial {t+1:3d}/{N}  |  PDR-QPD={pe:.5f}  CPDR={ce:.5f}")

    noisy = np.array(noisy_v)
    pdr   = np.array(pdr_v)
    cpdr  = np.array(cpdr_v)
    ne, pe, ce = np.abs(noisy-IDEAL_ZZ), np.abs(pdr-IDEAL_ZZ), np.abs(cpdr-IDEAL_ZZ)

    t_stat, p_val = stats.ttest_rel(ce, pe)   # H0: CPDR err = PDR err
    ci_pdr  = stats.t.interval(0.95, len(pdr)-1, np.mean(pdr),  stats.sem(pdr))
    ci_cpdr = stats.t.interval(0.95, len(cpdr)-1, np.mean(cpdr), stats.sem(cpdr))
    pdr_wins = np.mean(pe < ce)

    print("\n" + "=" * 60)
    print(f"{'Method':<22} {'Err Mean':>10} {'Err Std':>9} "
          f"{'CI∋Ideal':>10} {'Improve':>10}")
    print("-" * 60)
    for nm, arr, err, ci in [
        ('Standard QPD',       noisy, ne, (0, 0)),
        ('PDR-QPD Linear',     pdr,   pe, ci_pdr),
        ('Proper CPDR',        cpdr,  ce, ci_cpdr),
    ]:
        ci_ok = ci[0] <= IDEAL_ZZ <= ci[1] if nm != 'Standard QPD' else '-'
        imp   = f"{(1-err.mean()/ne.mean())*100:.1f}%" if nm != 'Standard QPD' else '-'
        print(f"  {nm:<20} {err.mean():>10.5f} {err.std():>9.5f} "
              f"{str(ci_ok):>10} {imp:>10}")
    print(f"\nPDR-QPD beats CPDR: {pdr_wins*100:.0f}% of trials")
    print(f"Paired t-test: t={t_stat:.2f}, p={p_val:.2e}")
    if p_val < 0.05:
        print("→ Statistically significant (p < 0.05)")

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for arr, lbl, col in [
        (noisy, 'Standard QPD',             '#e74c3c'),
        (pdr,   'PDR-QPD Linear (ours)',     '#2ecc71'),
        (cpdr,  'Proper CPDR\n(Zhang 2024)', '#3498db'),
    ]:
        ax1.scatter(np.random.normal(0, .03, len(arr)), arr,
                    alpha=0.25, color=col, s=10)
        ax1.plot([0], [arr.mean()], 'D', color=col, ms=11, label=f'{lbl} (μ={arr.mean():.4f})')
    ax1.axhline(IDEAL_ZZ, color='black', ls='--', lw=2, label=f'Ideal {IDEAL_ZZ:.5f}')
    ax1.set_title(f'Expectation Value Distribution (N={N})', fontweight='bold')
    ax1.set_ylabel('⟨ZZ⟩'); ax1.set_xticks([]); ax1.legend(fontsize=8)

    methods = ['Standard\nQPD', 'PDR-QPD\nLinear (ours)', 'Proper CPDR\n(Zhang 2024)']
    means   = [ne.mean(), pe.mean(), ce.mean()]
    stds    = [ne.std(),  pe.std(),  ce.std()]
    colors  = ['#e74c3c', '#2ecc71', '#3498db']
    x       = np.arange(3)
    bars    = ax2.bar(x, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title(f'Error Comparison\nPDR-QPD wins {pdr_wins*100:.0f}% trials, p={p_val:.1e}',
                  fontweight='bold')
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.0002, f'{m:.5f}',
                 ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'cpdr_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Figure] → {out}")

    csv_out = os.path.join(OUTPUT_DIR, 'cpdr_comparison.csv')
    with open(csv_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trial', 'noisy', 'pdr_qpd', 'cpdr_proper'])
        for t, (n, p, c) in enumerate(zip(noisy_v, pdr_v, cpdr_v)):
            w.writerow([t, n, p, c])
    print(f"[CSV]    → {csv_out}")


if __name__ == "__main__":
    main()
