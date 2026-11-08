"""
calibration.py — Structure-Aware Perturbation Calibrator
══════════════════════════════════════════════════════════
Implements calibration data generation with three modes:

Mode 'scalar':
    Perturb RX angles (δθ ~ U(−0.2, 0.2)).
    Measure ZZ in Z-basis only.
    Output: K pairs (noisy_ZZ, ideal_ZZ) per variant.
    → Used for Linear and GPR regression.

Mode 'bloch':
    Perturb both RX and RY angles for full Bloch sphere coverage.
    Measure each qubit in Z, X, Y bases (3 runs per perturbation).
    Output: K Bloch vectors (noisy and ideal) per qubit per variant.
    → Used for Bloch-matrix (Choi-inspired) correction.

Gentle Measurement Lemma:
    δθ ∈ [−0.2, 0.2] guarantees fidelity ≥ cos²(0.1) ≈ 0.9900
    with the target circuit state — calibration stays in local noise manifold.

Qiskit 2.x compatibility:
    - find_bit() replaced with safe _get_qubit_idx() helper
    - RXGate, RYGate import unchanged
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate

from config import K_CAL_SCALAR, K_CAL_BLOCH, DELTA_MAX, SEED


# ── Qubit index helper (Qiskit 1.x / 2.x safe) ──────────────────────────────

def _get_qubit_idx(qc: QuantumCircuit, qubit) -> int:
    try:
        return qc.find_bit(qubit).index
    except AttributeError:
        return qc.qubits.index(qubit)


# ── Circuit perturbation helpers ─────────────────────────────────────────────

def _perturb_rx(qc: QuantumCircuit, deltas: np.ndarray) -> QuantumCircuit:
    """Apply angle perturbations δθ to all RX gates in order."""
    new_qc   = QuantumCircuit(qc.num_qubits)
    rx_count = 0
    for inst in qc.data:
        gate   = inst.operation
        qubits = [_get_qubit_idx(qc, q) for q in inst.qubits]
        if gate.name == 'rx' and rx_count < len(deltas):
            new_qc.append(RXGate(gate.params[0] + deltas[rx_count]), qubits)
            rx_count += 1
        else:
            new_qc.append(gate, qubits)
    return new_qc


def _perturb_rx_ry(qc: QuantumCircuit,
                    deltas_x: np.ndarray,
                    deltas_y: np.ndarray) -> QuantumCircuit:
    """
    Apply RX perturbations (δx) AND add small RY rotations (δy) per qubit.
    RY perturbations create X-component variance on Bloch sphere,
    enabling full 3D noise characterisation.
    """
    new_qc   = QuantumCircuit(qc.num_qubits)
    rx_count = 0
    for inst in qc.data:
        gate   = inst.operation
        qubits = [_get_qubit_idx(qc, q) for q in inst.qubits]
        if gate.name == 'rx' and rx_count < len(deltas_x):
            q = qubits[0]
            new_qc.append(RXGate(gate.params[0] + deltas_x[rx_count]), [q])
            if rx_count < len(deltas_y):
                new_qc.append(RYGate(deltas_y[rx_count]), [q])
            rx_count += 1
        else:
            new_qc.append(gate, qubits)
    return new_qc


def _count_rx(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.name == 'rx')


# ── Main calibration function ────────────────────────────────────────────────

def generate_calibration_data(variant_qc: QuantumCircuit,
                               executor,
                               mode: str = 'scalar',
                               K: int = None,
                               delta_max: float = DELTA_MAX,
                               seed: int = SEED) -> dict:
    """
    Generate calibration data for ONE QPD variant circuit.

    Parameters
    ----------
    variant_qc : QuantumCircuit  (QPD variant, no measurements)
    executor   : QuantumExecutor
    mode       : 'scalar' or 'bloch'
    K          : number of calibration circuits (default from config)
    delta_max  : perturbation bound (default 0.2)
    seed       : RNG seed

    Returns (scalar mode)
    ----------------------
    {'noisy_zz': ndarray(K,), 'ideal_zz': ndarray(K,)}

    Returns (bloch mode, additional keys)
    --------------------------------------
    + 'noisy_bloch_q0/q1' : ndarray(K,3)
    + 'ideal_bloch_q0/q1' : ndarray(K,3)
    """
    K = K or (K_CAL_SCALAR if mode == 'scalar' else K_CAL_BLOCH)

    rng  = np.random.default_rng(seed)
    n_rx = _count_rx(variant_qc)

    # ── Generate K perturbed circuits ────────────────────────────────────────
    perturbed = []
    for _ in range(K):
        d_x = rng.uniform(-delta_max, delta_max, size=max(n_rx, 1))
        if mode == 'bloch':
            d_y = rng.uniform(-delta_max, delta_max, size=max(n_rx, 1))
            perturbed.append(_perturb_rx_ry(variant_qc, d_x, d_y))
        else:
            perturbed.append(_perturb_rx(variant_qc, d_x))

    # ── Ideal values (noiseless Statevector) ─────────────────────────────────
    ideal_zz       = np.array([executor.get_ideal_zz(qc) for qc in perturbed])
    ideal_bloch_q0 = (np.array([executor.get_ideal_bloch(qc, 0)
                                 for qc in perturbed]) if mode == 'bloch' else None)
    ideal_bloch_q1 = (np.array([executor.get_ideal_bloch(qc, 1)
                                 for qc in perturbed]) if mode == 'bloch' else None)

    # ── Noisy measurements (single batch job) ────────────────────────────────
    if mode == 'scalar':
        all_counts = executor.batch_run(perturbed)
        noisy_zz   = np.array([
            executor._counts_to_zz(c, variant_qc.num_qubits)
            for c in all_counts])
        return {'noisy_zz': noisy_zz, 'ideal_zz': ideal_zz}

    else:  # bloch: Z, X, Y per qubit
        n_qb = variant_qc.num_qubits

        x_rots = [[{'qubit': 0, 'basis': 'X'}, {'qubit': 1, 'basis': 'X'}]] * K
        y_rots = [[{'qubit': 0, 'basis': 'Y'}, {'qubit': 1, 'basis': 'Y'}]] * K

        counts_z = executor.batch_run(perturbed)
        counts_x = executor.batch_run(perturbed, basis_rotations=x_rots)
        counts_y = executor.batch_run(perturbed, basis_rotations=y_rots)

        noisy_bloch_q0 = np.zeros((K, 3))
        noisy_bloch_q1 = np.zeros((K, 3))
        noisy_zz       = np.zeros(K)

        for k in range(K):
            z0, z1 = executor._counts_to_z_per_qubit(counts_z[k], n_qb)
            x0, x1 = executor._counts_to_z_per_qubit(counts_x[k], n_qb)
            y0, y1 = executor._counts_to_z_per_qubit(counts_y[k], n_qb)

            noisy_bloch_q0[k] = [x0, y0, z0]
            noisy_bloch_q1[k] = [x1, y1, z1]
            noisy_zz[k]       = z0 * z1

        return {
            'noisy_zz'        : noisy_zz,
            'ideal_zz'        : ideal_zz,
            'noisy_bloch_q0'  : noisy_bloch_q0,
            'ideal_bloch_q0'  : ideal_bloch_q0,
            'noisy_bloch_q1'  : noisy_bloch_q1,
            'ideal_bloch_q1'  : ideal_bloch_q1,
        }
