#!/usr/bin/env python3
"""
CPDR-ZNE reproduction (compact, runnable)
Follows: "Clifford Perturbation Approximation for Quantum Error Mitigation"
(see uploaded PDF). Implements:
 - Trotter Ising circuit builder (RX + RZZ layers)
 - SPD approximation (Algorithm 1) with truncation M
 - Generate training set T as in paper (theta grid)
 - Noisy simulations with Aer + configurable depolarizing noise scaled by G
 - Train Ridge on noisy expectations at multiple Gs (G_list) -> predict ideal
"""
import math
import itertools
import time
import numpy as np
from sklearn.linear_model import Ridge
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel

# -------------------------
# Utility: Pauli string ops
# -------------------------
_single_mul = {
    ('I','I'):(1,'I'), ('I','X'):(1,'X'), ('I','Y'):(1,'Y'), ('I','Z'):(1,'Z'),
    ('X','I'):(1,'X'), ('X','X'):(1,'I'), ('X','Y'):(1j,'Z'), ('X','Z'):(-1j,'Y'),
    ('Y','I'):(1,'Y'), ('Y','X'):(-1j,'Z'), ('Y','Y'):(1,'I'), ('Y','Z'):(1j,'X'),
    ('Z','I'):(1,'Z'), ('Z','X'):(1j,'Y'), ('Z','Y'):(-1j,'X'), ('Z','Z'):(1,'I'),
}
_anticommute_pairs = {('X','Y'),('Y','X'),('Y','Z'),('Z','Y'),('Z','X'),('X','Z')}

def pauli_mul(p1: str, p2: str):
    """Multiply two equal-length Pauli strings. Return (phase (complex), result_str)."""
    assert len(p1) == len(p2)
    phase = 1+0j
    out = []
    for a,b in zip(p1, p2):
        ph, r = _single_mul[(a,b)]
        phase *= ph
        out.append(r)
    return phase, ''.join(out)

def pauli_commutes(p1: str, p2: str):
    """Return True if p1 commutes with p2 (multi-qubit Pauli strings)."""
    # count positions with anti-commuting single-qubit Paulis
    cnt = 0
    for a,b in zip(p1, p2):
        if a == 'I' or b == 'I': continue
        if (a,b) in _anticommute_pairs:
            cnt += 1
    return (cnt % 2) == 0

# -------------------------
# SPD: Heisenberg evolution
# -------------------------
def SPD_truncated_expectation(observable: str, gates: list, thetas: list, M: int):
    """
    Compute approximate expectation <O>^(M) for initial state |0>^n
    using SPD truncation.
    - observable: Pauli string e.g. 'ZIZI...'
    - gates: list of Pauli strings (P_k) applied in forward order
             each P_k is same length string, e.g. 'X...'
    - thetas: list of rotation angles corresponding to gates (same length)
    - M: truncation order (max number of 'sin' factors allowed)
    Returns real approximate expectation (float).
    Implementation of Algorithm 1 (Heisenberg propagation & truncate by sin-count).
    """
    n = len(observable)
    # list of (pauli_str, coeff (complex), sin_count)
    terms = [(observable, 1+0j, 0)]
    # apply gates in reverse (Heisenberg)
    for P, theta in zip(reversed(gates), reversed(thetas)):
        new_terms = {}
        c_cos = math.cos(theta)
        c_sin = math.sin(theta)
        for sigma, coeff, sincount in terms:
            if pauli_commutes(sigma, P):
                # sigma -> sigma (coeff unchanged)
                key = (sigma, sincount)
                new_terms[key] = new_terms.get(key, 0+0j) + coeff
            else:
                # sigma -> cos(theta) * sigma   + i sin(theta) * P * sigma
                # term 1:
                k1 = (sigma, sincount)
                new_terms[k1] = new_terms.get(k1, 0+0j) + coeff * c_cos
                # term 2:
                phase, new_sigma = pauli_mul(P, sigma)
                k2 = (new_sigma, sincount + 1)
                new_terms[k2] = new_terms.get(k2, 0+0j) + coeff * (1j * c_sin) * phase
        # prune by M
        terms = [(s, new_terms[(s,sc)], sc) for (s,sc) in new_terms.keys() if sc <= M]
    # Now compute expectation over |0>^n: only Pauli strings with only I and Z contribute.
    exp = 0+0j
    for sigma, coeff, sc in terms:
        # tr{|0><0|^{⊗n} sigma} = 1 if sigma has only I/Z (and Z acts with +1), else 0
        if all(c in ('I','Z') for c in sigma):
            exp += coeff  # each such Pauli has expectation 1 on |0>^n
    return float(np.real_if_close(exp))

# -------------------------
# Build Trotter ising gates list (Pauli strings)
# -------------------------
def build_trotter_gate_lists(n_qubits, steps, theta_h, theta_J):
    """
    Return lists:
      gates_list: list of Pauli strings for each rotation in forward order
      thetas: corresponding angles
    Our Trotter step: RX on every qubit with angle theta_h, then RZZ between neighbour pairs with theta_J.
    For SPD we represent RX as Pauli 'X' on that qubit, RZZ as 'Z' on the two qubits.
    """
    gates = []
    thetas = []
    for _ in range(steps):
        # RX layer: single-qubit rotations for each qubit
        for i in range(n_qubits):
            P = ['I'] * n_qubits
            P[i] = 'X'
            gates.append(''.join(P))
            thetas.append(theta_h)
        # RZZ layer: neighbour interactions (linear chain)
        for i in range(n_qubits - 1):
            P = ['I'] * n_qubits
            P[i] = 'Z'; P[i+1] = 'Z'
            gates.append(''.join(P))
            thetas.append(theta_J)
    return gates, thetas

# -------------------------
# Qiskit circuit builder (with measurements)
# -------------------------
def build_trotter_qc(n_qubits, steps, theta_h, theta_J, add_measure=True):
    qc = QuantumCircuit(n_qubits)
    # start in |0>^n
    for _ in range(steps):
        for i in range(n_qubits):
            qc.rx(theta_h, i)
        for i in range(n_qubits - 1):
            qc.rzz(theta_J, i, i+1)
    if add_measure:
        qc.measure_all()
    return qc

# -------------------------
# Noisy simulator helper
# -------------------------
def make_depolarizing_noise_model(p1, p2):
    """Return a NoiseModel with single/two-qubit depolarizing errors."""
    nm = NoiseModel()
    e1 = depolarizing_error(p1, 1)
    e2 = depolarizing_error(p2, 2)
    # attach to all common basis gates (rx, rzz composed by cx/zz etc); for simplicity attach to 'rx' and 'rzz'
    nm.add_all_qubit_quantum_error(e1, ['rx','u3'])
    nm.add_all_qubit_quantum_error(e2, ['rzz','cx'])  # rzz mapping may use cx in transpile
    return nm

def run_noisy_counts(qc, noise_model, shots=10000, seed=None):
    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=shots, seed_simulator=seed)
    res = job.result()
    return res.get_counts()

# -------------------------
# Expectation from counts for a given Pauli observable (multi-qubit)
# -------------------------
def counts_expectation_pauli(counts, pauli_str):
    # pauli_str like 'ZIZZ...': compute eigenvalue +-1 per outcome
    shots = sum(counts.values())
    exp = 0.0
    for bitstr, c in counts.items():
        # Qiskit little-endian in counts: reverse to align qubit0 as leftmost in our pauli_str design
        b = bitstr[::-1]
        val = 1
        for bit, p in zip(b, pauli_str):
            if p == 'I': continue
            if p == 'Z':
                val *= (1 if bit == '0' else -1)
            else:
                # if measuring X or Y we'd need basis change; here SPD uses Pauli observable O (we restrict to Z-magnetization or Z-strings)
                # if non-Z present we conservatively return 0 for measurement-based estimation
                val = 0
                break
        exp += val * c
    return exp / shots

# -------------------------
# Build training grid T as in paper (Eq. in Sec IV)
# -------------------------
def build_theta_grid():
    # paper uses indices i = 0..5 and 54..59 for both axes -> 12 values * 12 values = 144 pairs
    small = list(range(0,6))
    large = list(range(54,60))
    idxs = small + large
    thetas_h = [i * math.pi / 120.0 for i in idxs]
    thetas_J = [-j * math.pi / 120.0 for j in idxs]
    grid = [(th, tJ) for th in thetas_h for tJ in thetas_J]
    return grid

# -------------------------
# Main reproduce pipeline
# -------------------------
def cpdr_zne_pipeline(
    n_qubits=8,
    steps=5,
    G_list=(1.0, 1.2, 1.6),
    shots=10000,
    trunc_M=7,
    ridge_alpha=2e-5,
    base_depol_1q=0.001,
    base_depol_2q=0.005,
    seed=None
):
    """
    - Build training set T according to paper (144 theta pairs)
    - For each theta_k in T:
        - compute SPD ideal <O>^(M)
        - run noisy simulations at G in G_list (scale depol params by G)
    - Fit Ridge: X_train shape (K, len(G_list)), y_train shape (K,)
    - Apply to a target theta (choose some) and report results.
    """
    print("Building training grid (144 pairs) ...")
    T = build_theta_grid()
    K = len(T)
    print(f"Training set size K = {K}")
    # choose observable: for demo use global Z^{⊗n} (paper uses various weight Pauli; change here as needed)
    observable = 'Z' * n_qubits

    X = np.zeros((K, len(G_list)))
    y = np.zeros(K)

    # instantiate ideal SPD using gate lists (no qiskit)
    start = time.time()
    for k, (theta_h, theta_J) in enumerate(T):
        # SPD ideal
        gates, thetas = build_trotter_gate_lists(n_qubits, steps, theta_h, theta_J)
        spd_val = SPD_truncated_expectation(observable, gates, thetas, trunc_M)
        y[k] = spd_val

        # build a qiskit circuit (with measurements)
        qc = build_trotter_qc(n_qubits, steps, theta_h, theta_J, add_measure=True)

        # noisy expectations for each G
        for j, G in enumerate(G_list):
            p1 = base_depol_1q * G
            p2 = base_depol_2q * G
            nm = make_depolarizing_noise_model(p1, p2)
            counts = run_noisy_counts(qc, nm, shots=shots, seed=seed)
            X[k, j] = counts_expectation_pauli(counts, observable)
        if (k+1) % 16 == 0:
            print(f"  processed {k+1}/{K} training circuits ...")
    dur = time.time() - start
    print(f"Training data generated in {dur:.1f}s")

    # Train Ridge regression per Eq.(14)
    print("Training Ridge regression (CPDR-ZNE)...")
    model = Ridge(alpha=ridge_alpha)
    model.fit(X, y)
    print("Model trained. Coeffs:", model.coef_, "intercept:", model.intercept_)

    # Pick a target theta (e.g., theta_h mid between 0 and pi/2)
    target_theta_h = math.pi / 6
    target_theta_J = -math.pi / 2  # paper often uses -pi/2
    qc_target = build_trotter_qc(n_qubits, steps, target_theta_h, target_theta_J, add_measure=True)

    # compute true ideal (for small n) via Statevector
    qc_target_nv = build_trotter_qc(n_qubits, steps, target_theta_h, target_theta_J, add_measure=False)
    psi = Statevector.from_instruction(qc_target_nv)
    # exact ideal expectation for global Z
    exact_op = None  # we can compute directly from statevector
    # compute expectation explicitly
    # for |0> basis, expectation of Z^.. = sum |amp|^2 * parity_sign
    exact = 0.0
    sv = psi.data
    nstates = len(sv)
    for idx, amp in enumerate(sv):
        bits = format(idx, '0{}b'.format(n_qubits))[::-1]
        val = 1
        for b in bits:
            val *= (1 if b == '0' else -1)
        exact += val * (abs(amp)**2)
    print(f"Exact (Statevector) target value: {exact:.6f}")

    # Unmitigated noisy run with base G=1
    nm_base = make_depolarizing_noise_model(base_depol_1q, base_depol_2q)
    counts_unmit = run_noisy_counts(qc_target, nm_base, shots=shots)
    noisy_unmit = counts_expectation_pauli(counts_unmit, observable)
    print(f"Noisy (unmitigated): {noisy_unmit:.6f}")

    # Build noisy feature vector at G_list and predict
    feat = []
    for G in G_list:
        nm = make_depolarizing_noise_model(base_depol_1q * G, base_depol_2q * G)
        counts = run_noisy_counts(qc_target, nm, shots=shots)
        feat.append(counts_expectation_pauli(counts, observable))
    feat = np.array(feat).reshape(1, -1)
    mitigated = model.predict(feat)[0]
    print(f"CPDR-ZNE mitigated estimate: {mitigated:.6f}")
    print(f"Absolute errors (noisy vs mitigated): {abs(exact - noisy_unmit):.6f} , {abs(exact - mitigated):.6f}")
    return {
        'model': model, 'X_train': X, 'y_train': y,
        'target': {'exact': exact, 'noisy': noisy_unmit, 'mitigated': mitigated}
    }

if __name__ == "__main__":
    out = cpdr_zne_pipeline(n_qubits=8, steps=5, shots=2000, trunc_M=7)
    print("Done.")
