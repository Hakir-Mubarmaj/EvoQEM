"""
calibration.py — Structure-Aware Perturbation Calibrator
══════════════════════════════════════════════════════════
Implements the calibration data generation protocol with two modes:

Mode 'scalar':
    Perturb RX angles (δθ ~ U(−0.2, 0.2)) on all RX gates.
    Measure ZZ in Z-basis only.
    Output: K pairs (noisy_ZZ, ideal_ZZ) per variant.
    → Used for Linear and GPR regression.

Mode 'bloch':
    Perturb both RX and RY angles for full Bloch sphere coverage.
    Measure each qubit in Z, X, Y bases (3 circuit runs per perturbation).
    Output: K Bloch vectors (noisy and ideal) per qubit per variant.
    → Used for Bloch-matrix (Choi-inspired) correction.

Gentle Measurement Lemma:
    Perturbations δθ ∈ [−0.2, 0.2] guarantee fidelity ≥ cos²(0.1) ≈ 0.995
    with the target circuit state, keeping calibration circuits in the
    local noise manifold of the target.

Efficiency:
    All K circuits per mode are run in a SINGLE batch_run() call,
    minimising sampler job-submission overhead.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate

from config import K_CAL_SCALAR, K_CAL_BLOCH, DELTA_MAX, SEED


# ── Circuit perturbation helpers ─────────────────────────────────────────────

def _perturb_rx(qc: QuantumCircuit, deltas: np.ndarray) -> QuantumCircuit:
    """
    Apply angle perturbations δθ to all RX gates in a circuit.
    δθ values are consumed in the order gates appear.
    """
    new_qc   = QuantumCircuit(qc.num_qubits)
    rx_count = 0
    for inst in qc.data:
        gate    = inst.operation
        qubits  = [qc.find_bit(q).index for q in inst.qubits]
        if gate.name == 'rx' and rx_count < len(deltas):
            new_qc.append(RXGate(gate.params[0] + deltas[rx_count]), qubits)
            rx_count += 1
        else:
            new_qc.append(gate, qubits)
    return new_qc


def _perturb_rx_ry(qc: QuantumCircuit, deltas_x: np.ndarray,
                    deltas_y: np.ndarray) -> QuantumCircuit:
    """
    Apply RX perturbations (δx) AND add small RY rotations (δy) per qubit.
    RY perturbations create variance in the X component of the Bloch vector,
    enabling full 3D noise characterisation beyond the Y-Z plane.
    """
    new_qc   = QuantumCircuit(qc.num_qubits)
    rx_count = 0
    for inst in qc.data:
        gate   = inst.operation
        qubits = [qc.find_bit(q).index for q in inst.qubits]
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
    variant_qc : QuantumCircuit
        A QPD variant circuit (without measurements).
    executor : QuantumExecutor
        Execution engine for noisy measurements.
    mode : str
        'scalar' → ZZ only, K_CAL_SCALAR circuits.
        'bloch'  → XYZ per qubit, K_CAL_BLOCH circuits × 3 bases.
    K : int, optional
        Override default K (K_CAL_SCALAR or K_CAL_BLOCH).
    delta_max : float
        Perturbation magnitude bound.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys depending on mode:
      Scalar mode:
        'noisy_zz'   : np.ndarray (K,) — noisy ZZ per calibration circuit
        'ideal_zz'   : np.ndarray (K,) — ideal ZZ per calibration circuit

      Bloch mode (adds):
        'noisy_bloch_q0' : np.ndarray (K, 3) — noisy [X,Y,Z] for qubit 0
        'ideal_bloch_q0' : np.ndarray (K, 3)
        'noisy_bloch_q1' : np.ndarray (K, 3) — noisy [X,Y,Z] for qubit 1
        'ideal_bloch_q1' : np.ndarray (K, 3)
    """
    if mode == 'scalar':
        K = K or K_CAL_SCALAR
    else:
        K = K or K_CAL_BLOCH

    rng    = np.random.default_rng(seed)
    n_rx   = _count_rx(variant_qc)
    n_qb   = variant_qc.num_qubits

    # ── Generate all K perturbed circuits ────────────────────────────────────
    perturbed_circuits = []
    for _ in range(K):
        d_x = rng.uniform(-delta_max, delta_max, size=max(n_rx, 1))
        if mode == 'bloch':
            d_y = rng.uniform(-delta_max, delta_max, size=max(n_rx, 1))
            cal_qc = _perturb_rx_ry(variant_qc, d_x, d_y)
        else:
            cal_qc = _perturb_rx(variant_qc, d_x)
        perturbed_circuits.append(cal_qc)

    # ── Compute ideal values (noiseless Statevector — no shots needed) ────────
    ideal_zz        = np.array([executor.get_ideal_zz(qc)
                                 for qc in perturbed_circuits])
    ideal_bloch_q0  = np.array([executor.get_ideal_bloch(qc, qubit=0)
                                 for qc in perturbed_circuits]) if mode=='bloch' else None
    ideal_bloch_q1  = np.array([executor.get_ideal_bloch(qc, qubit=1)
                                 for qc in perturbed_circuits]) if mode=='bloch' else None

    # ── Run ALL noisy measurements in ONE batch ───────────────────────────────
    if mode == 'scalar':
        # One batch: Z-basis only
        all_counts = executor.batch_run(perturbed_circuits)
        noisy_zz   = np.array([
            executor._counts_to_zz(c, n_qb) for c in all_counts])
        return {'noisy_zz': noisy_zz, 'ideal_zz': ideal_zz}

    else:  # mode == 'bloch': need Z, X, Y for each qubit
        # Build 3K circuits: Z-basis, X-basis (H on both), Y-basis (Sdg·H on both)
        z_circuits = perturbed_circuits                         # K circuits
        x_rotations = [[{'qubit': 0, 'basis': 'X'},
                         {'qubit': 1, 'basis': 'X'}]] * K
        y_rotations = [[{'qubit': 0, 'basis': 'Y'},
                         {'qubit': 1, 'basis': 'Y'}]] * K

        counts_z = executor.batch_run(z_circuits)
        counts_x = executor.batch_run(z_circuits, basis_rotations=x_rotations)
        counts_y = executor.batch_run(z_circuits, basis_rotations=y_rotations)

        noisy_bloch_q0 = np.zeros((K, 3))
        noisy_bloch_q1 = np.zeros((K, 3))
        noisy_zz       = np.zeros(K)

        for k in range(K):
            z0_z, z1_z = executor._counts_to_z_per_qubit(counts_z[k], n_qb)
            x0,   x1   = executor._counts_to_z_per_qubit(counts_x[k], n_qb)
            y0,   y1   = executor._counts_to_z_per_qubit(counts_y[k], n_qb)

            noisy_bloch_q0[k] = [x0, y0, z0_z]   # (X, Y, Z) for qubit 0
            noisy_bloch_q1[k] = [x1, y1, z1_z]   # (X, Y, Z) for qubit 1
            noisy_zz[k]       = z0_z * z1_z       # Product-state ZZ

        return {
            'noisy_zz'       : noisy_zz,
            'ideal_zz'       : ideal_zz,
            'noisy_bloch_q0' : noisy_bloch_q0,
            'ideal_bloch_q0' : ideal_bloch_q0,
            'noisy_bloch_q1' : noisy_bloch_q1,
            'ideal_bloch_q1' : ideal_bloch_q1,
        }
