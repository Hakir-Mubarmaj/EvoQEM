# cpdr_with_qpdr.py
"""
CPDR core (unchanged configuration) + QPDR (QPD + PDR) injected for target evaluation.
- Keeps CPDR training-grid, SPD, G_list, ridge_alpha as in original CPDR script.
- For target: runs CPDR mitigation (full-circuit) AND runs QPDR pipeline:
    * decompose CZ via src.qpd_lib.decompose_cz_gate
    * per-variant: generate calibration circuits via src.calibration.generate_calibration_circuits
                 run noisy training evals, fit PDR (linregress then Theil-Sen fallback)
    * reconstruct global QPDR result via src.qpd_lib.reconstruct_qpd_result
- Writes per-variant CSV like your original QPDR pipeline.
"""
import math
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import r2_score

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel

# your modules
from src.calibration_backup import generate_calibration_circuits
from src.qpd_lib import decompose_cz_gate, reconstruct_qpd_result
from src.executor import VirtualExecutor

# --- add near top of cpdr_with_qpdr.py (imports-এর ঠিক পরে) ---

def strip_measurements(qc):
    """Return a copy of qc with any 'measure' instructions removed."""
    qc2 = qc.copy()
    qc2.data = [instr for instr in qc2.data if getattr(instr.operation, "name", "") != "measure"]
    return qc2

def ensure_measured(qc):
    """Return a copy of qc that has measure_all() if it currently has no measure."""
    qc2 = qc.copy()
    has_measure = any(getattr(instr.operation, "name", "") == "measure" for instr in qc2.data)
    if not has_measure:
        qc2.measure_all()
    return qc2


# -------------------------
# ------- CPDR (same config from earlier) ----------
# -------------------------
def pauli_mul(p1: str, p2: str):
    # minimal helper used in SPD earlier (we re-use SPD implementation below)
    _single_mul = {
        ('I','I'):(1,'I'), ('I','X'):(1,'X'), ('I','Y'):(1,'Y'), ('I','Z'):(1,'Z'),
        ('X','I'):(1,'X'), ('X','X'):(1,'I'), ('X','Y'):(1j,'Z'), ('X','Z'):(-1j,'Y'),
        ('Y','I'):(1,'Y'), ('Y','X'):(-1j,'Z'), ('Y','Y'):(1,'I'), ('Y','Z'):(1j,'X'),
        ('Z','I'):(1,'Z'), ('Z','X'):(1j,'Y'), ('Z','Y'):(-1j,'X'), ('Z','Z'):(1,'I'),
    }
    assert len(p1) == len(p2)
    phase = 1+0j
    out = []
    for a,b in zip(p1,p2):
        ph, r = _single_mul[(a,b)]
        phase *= ph
        out.append(r)
    return phase, ''.join(out)

def pauli_commutes(p1: str, p2: str):
    _anticommute_pairs = {('X','Y'),('Y','X'),('Y','Z'),('Z','Y'),('Z','X'),('X','Z')}
    cnt = 0
    for a,b in zip(p1,p2):
        if a == 'I' or b == 'I': continue
        if (a,b) in _anticommute_pairs:
            cnt += 1
    return (cnt % 2) == 0

def SPD_truncated_expectation(observable: str, gates: list, thetas: list, M: int):
    terms = [(observable, 1+0j, 0)]
    for P, theta in zip(reversed(gates), reversed(thetas)):
        new_terms = {}
        c_cos = math.cos(theta)
        c_sin = math.sin(theta)
        for sigma, coeff, sincount in terms:
            if pauli_commutes(sigma, P):
                key = (sigma, sincount)
                new_terms[key] = new_terms.get(key, 0+0j) + coeff
            else:
                k1 = (sigma, sincount)
                new_terms[k1] = new_terms.get(k1, 0+0j) + coeff * c_cos
                phase, new_sigma = pauli_mul(P, sigma)
                k2 = (new_sigma, sincount + 1)
                new_terms[k2] = new_terms.get(k2, 0+0j) + coeff * (1j * c_sin) * phase
        terms = [(s, new_terms[(s,sc)], sc) for (s,sc) in new_terms.keys() if sc <= M]
    exp = 0+0j
    for sigma, coeff, sc in terms:
        if all(c in ('I','Z') for c in sigma):
            exp += coeff
    return float(np.real_if_close(exp))

def build_trotter_gate_lists(n_qubits, steps, theta_h, theta_J):
    gates = []
    thetas = []
    for _ in range(steps):
        for i in range(n_qubits):
            P = ['I'] * n_qubits
            P[i] = 'X'
            gates.append(''.join(P))
            thetas.append(theta_h)
        for i in range(n_qubits - 1):
            P = ['I'] * n_qubits
            P[i] = 'Z'; P[i+1] = 'Z'
            gates.append(''.join(P))
            thetas.append(theta_J)
    return gates, thetas

def build_theta_grid():
    small = list(range(0,6))
    large = list(range(54,60))
    idxs = small + large
    thetas_h = [i * math.pi / 120.0 for i in idxs]
    thetas_J = [-j * math.pi / 120.0 for j in idxs]
    return [(th, tJ) for th in thetas_h for tJ in thetas_J]

# -------------------------
# Qiskit noisy-run helpers (used by both CPDR training and QPDR variant eval)
# -------------------------
def make_depolarizing_noise_model(p1, p2):
    nm = NoiseModel()
    e1 = depolarizing_error(p1, 1)
    e2 = depolarizing_error(p2, 2)
    nm.add_all_qubit_quantum_error(e1, ['rx', 'u3', 'rz'])
    nm.add_all_qubit_quantum_error(e2, ['rzz', 'cx'])
    return nm

def run_noisy_counts(qc, noise_model, shots=4000, seed=None):
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=shots, seed_simulator=seed)
    res = job.result()
    return res.get_counts()

def counts_expectation_pauli(counts, pauli_str):
    shots = sum(counts.values())
    exp = 0.0
    for bitstr, c in counts.items():
        b = bitstr[::-1]
        val = 1
        for bit, p in zip(b, pauli_str):
            if p == 'I': continue
            if p == 'Z':
                val *= (1 if bit == '0' else -1)
            else:
                val = 0
                break
        exp += val * c
    return exp / shots

def build_trotter_qc(n_qubits, steps, theta_h, theta_J, add_measure=True):
    qc = QuantumCircuit(n_qubits)
    for _ in range(steps):
        for i in range(n_qubits):
            qc.rx(theta_h, i)
        for i in range(n_qubits - 1):
            qc.rzz(theta_J, i, i+1)
    if add_measure:
        qc.measure_all()
    return qc

# -------------------------
# CPDR training (unchanged config)
# -------------------------
def train_cpdr(n_qubits=8, steps=5, trunc_M=7, G_list=(1.0,1.2,1.6),
               shots=2000, base_depol_1q=0.001, base_depol_2q=0.005, ridge_alpha=2e-5, seed=None):
    print("Building training grid (144 pairs) ...")
    T = build_theta_grid()
    K = len(T)
    print(f"Training set size K = {K}")
    observable = 'Z' * n_qubits

    X = np.zeros((K, len(G_list)))
    y = np.zeros(K)

    start = time.time()
    for k, (theta_h, theta_J) in enumerate(T):
        gates, thetas = build_trotter_gate_lists(n_qubits, steps, theta_h, theta_J)
        y[k] = SPD_truncated_expectation(observable, gates, thetas, trunc_M)

        qc = build_trotter_qc(n_qubits, steps, theta_h, theta_J, add_measure=True)
        for j, G in enumerate(G_list):
            p1 = base_depol_1q * G
            p2 = base_depol_2q * G
            nm = make_depolarizing_noise_model(p1, p2)
            counts = run_noisy_counts(qc, nm, shots=shots, seed=seed)
            X[k, j] = counts_expectation_pauli(counts, observable)
        if (k+1) % 16 == 0:
            print(f"  processed {k+1}/{K} training circuits ...")
    print(f"Training data generated in {time.time()-start:.1f}s")

    model = Ridge(alpha=ridge_alpha)
    model.fit(X, y)
    print("CPDR model trained. Coeffs:", model.coef_, "intercept:", model.intercept_)
    return {
        'model': model,
        'G_list': G_list,
        'n_qubits': n_qubits,
        'steps': steps,
        'trunc_M': trunc_M,
        'base_depol_1q': base_depol_1q,
        'base_depol_2q': base_depol_2q,
        'shots': shots
    }

# -------------------------
# QPDR logic (decompose + per-variant PDR training using src.calibration)
# -------------------------
def fit_pdr_local(noisy_X, ideal_y):
    """
    Fit a simple local PDR mapping noisy -> ideal.
    noisy_X: 1D array-like (K,)
    ideal_y: 1D (K,)
    Returns slope, intercept, r2, p_value, std_err, model_predict_func
    """
    noisy_X = np.array(noisy_X).astype(float)
    ideal_y = np.array(ideal_y).astype(float)
    if len(noisy_X) < 3:
        # fallback to identity if too few points
        return 1.0, 0.0, float('nan'), None, None, lambda x: x

    slope, intercept, r_value, p_value, std_err = stats.linregress(noisy_X, ideal_y)
    r2 = r_value**2
    if r2 < 0.82:
        model = TheilSenRegressor(random_state=0)
        X = noisy_X.reshape(-1,1)
        y = ideal_y
        model.fit(X, y)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        r2 = r2_score(y, model.predict(X))
        predict_fn = lambda x: float(model.predict(np.array(x).reshape(-1,1))[0]) if np.ndim(x) == 0 else model.predict(np.array(x).reshape(-1,1))
        return slope, intercept, r2, p_value, std_err, predict_fn
    else:
        def predict_fn(x):
            return slope * x + intercept
        return slope, intercept, r2, p_value, std_err, predict_fn

# -------------------------
# Main integrated pipeline
# -------------------------
def run_cpdr_with_qpdr(seed=None):
    # CPDR hyperparams (same as earlier)
    n_qubits = 8
    steps = 5
    trunc_M = 7
    G_list = (1.0, 1.2, 1.6)
    shots = 3000
    base_depol_1q = 0.001
    base_depol_2q = 0.005
    ridge_alpha = 2e-5
    executor = VirtualExecutor(backend_name='fez')

    print("=== STEP 1: Train CPDR model (unchanged config) ===")
    cpdr = train_cpdr(n_qubits=n_qubits, steps=steps, trunc_M=trunc_M,
                      G_list=G_list, shots=shots,
                      base_depol_1q=base_depol_1q, base_depol_2q=base_depol_2q,
                      ridge_alpha=ridge_alpha)
    model = cpdr['model']

    # Build a target circuit (same pattern as CPDR example)
    print("\n=== STEP 2: Build target circuit (to be QPD-decomposed) ===")
    # Here we intentionally include CZ so QPD is meaningful:
    qc_target = build_trotter_qc(n_qubits, steps, math.pi/6, -math.pi/2, add_measure=True)

    # For comparison compute exact ideal (statevector)
    qc_nv = build_trotter_qc(n_qubits, steps, math.pi/6, -math.pi/2, add_measure=False)
    psi = Statevector.from_instruction(qc_nv)
    exact = 0.0
    sv = psi.data
    for idx, amp in enumerate(sv):
        bits = format(idx, '0{}b'.format(n_qubits))[::-1]
        val = 1
        for b in bits:
            val *= (1 if b == '0' else -1)
        exact += val * (abs(amp)**2)
    print(f"Exact ideal (global Z^n): {exact:.6f}")

    # Unmitigated noisy run (single G = 1)
    nm_base = make_depolarizing_noise_model(base_depol_1q, base_depol_2q)
    noisy_unmit = executor.run_mitigation_experiments([qc_target], shots=8000, seed=seed)[0]
    print(f"Unmitigated noisy (global circuit): {noisy_unmit:.6f}")

    # CPDR mitigation on full circuit (as baseline)
    feat = []
    for G in G_list:
        # try passing noise_scale kw (executor may accept it; main_qpd used this style)
        try:
            feat.append(executor.run_mitigation_experiments([qc_target], shots=8000, noise_scale=G, seed=seed)[0])
        except TypeError:
            # fallback to calling executor without noise_scale if it doesn't support; you can keep original Aer-based code then
            feat.append(executor.run_mitigation_experiments([qc_target], shots=8000, seed=seed)[0])
    feat = np.array(feat).reshape(1, -1)
    cpdr_mitigated = float(model.predict(feat)[0])

    # --- NOW inject QPDR: decompose target into QPD variants and run per-variant PDR ---
    print("\n=== STEP 3: QPDR on target (decompose CZ gates, then PDR per variant) ===")
    # Use qpd_lib.decompose_cz_gate which expects a circuit and qubit_pair; adapt if multiple CZs exist.
    # For simplicity, assume target has one primary CZ pair for decomposition (if more, you can iterate)
    # We'll try to locate first CZ occurrence (if none, fall back to single-variant)
    # Try to decompose pair (0,1) as example (user's qpd_lib used 2-qubit example)
    try:
        qpd_variants = decompose_cz_gate(qc_target, qubit_pair=(0,1))
        variants = [v[0] for v in qpd_variants]
        weights = [v[1] for v in qpd_variants]
    except Exception as e:
        print(f"[Warning] qpd_lib.decompose_cz_gate failed or not appropriate: {e}. Falling back to single-variant.")
        variants = [qc_target]
        weights = [1.0]

    variant_raw_results = []
    variant_pdr_mitigated = []
    per_variant_stats = []

    for i, variant_qc in enumerate(variants):
        print(f"Processing QPD variant {i+1}/{len(variants)} ...")

        # A. Generate calibration circuits (structure-preserving) for this variant (use your src.calibration)
        train_circuits, train_ideals = generate_calibration_circuits([strip_measurements(variant_qc)], num_samples=50)

        # B. For each calibration circuit, run noisy eval (using base noise, single-G) to build X_train for PDR
        noisy_train_X = executor.run_mitigation_experiments(train_circuits, shots=8000)
        noisy_train_X = np.array(noisy_train_X)
        ideal_vals = np.array(train_ideals)

        # C. Fit PDR local model
        slope, intercept, r2, pval, stderr, predict_fn = fit_pdr_local(noisy_train_X, ideal_vals)
        print(f"  PDR fit: slope={slope:.4f}, intercept={intercept:.4f}, R2={r2:.4f}")

        # D. Evaluate raw variant (unmitigated)
        raw_val = executor.run_mitigation_experiments([variant_qc], shots=12000)[0]

        # E. Mitigate variant via PDR predict_fn
        mitigated_val = predict_fn(raw_val)

        variant_raw_results.append(raw_val)
        variant_pdr_mitigated.append(float(mitigated_val))

        per_variant_stats.append({
            'variant': i+1,
            'slope': slope, 'intercept': intercept, 'r2': r2,
            'raw_val': raw_val, 'mit_val': float(mitigated_val)
        })

    # Reconstruct final global values via QPD weights
    global_raw_qpd = reconstruct_qpd_result(variant_raw_results, weights)
    global_mit_qpd = reconstruct_qpd_result(variant_pdr_mitigated, weights)

    print("\n=== RESULTS ===")
    print(f"Exact ideal:              {exact:.6f}")
    print(f"Noisy (unmitigated):      {noisy_unmit:.6f}")
    print(f"CPDR mitigated (full):    {cpdr_mitigated:.6f}")
    print(f"QPDR global (raw recon):  {global_raw_qpd:.6f}")
    print(f"QPDR global (PDR mitig):  {global_mit_qpd:.6f}")

    # Save per-variant stats to CSV (mimic your pipeline)
    pd.DataFrame(per_variant_stats).to_csv('csv/qpdr_variant_stats.csv', index=False)

    

    return {
        'exact': exact,
        'noisy': noisy_unmit,
        'cpdr_mitigated': cpdr_mitigated,
        'qpdr_raw': global_raw_qpd,
        'qpdr_mitigated': global_mit_qpd,
        'per_variant_stats': per_variant_stats
    }

if __name__ == "__main__":
    out = run_cpdr_with_qpdr()
    print("Done.")
