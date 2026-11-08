"""
executor.py — Unified Quantum Execution Engine
═══════════════════════════════════════════════
Uses SamplerV2 (Qiskit ≥ 1.3) exclusively.
Key design: batch_run() sends ALL circuits in one job → major speed-up.

Bit-ordering convention (Qiskit):
    Bitstring is big-endian in print order but stored little-endian:
    rightmost character in counts key = qubit 0.
    e.g. "10" means qubit 1 = |1⟩, qubit 0 = |0⟩.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from config import SHOTS, SEED


class QuantumExecutor:
    """
    Unified quantum execution engine.

    All noisy measurements use SamplerV2 on an AerSimulator with FakeFezV2
    noise model (or a custom depolarising noise model as fallback).

    Batch execution: call batch_run() with a list of circuits to minimise
    job-submission overhead — critical for the 240+ circuit calibration loops.
    """

    def __init__(self, shots: int = SHOTS, seed: int = SEED):
        self.shots  = shots
        self.seed   = seed
        self.backend = self._init_backend()
        self.sampler = SamplerV2.from_backend(self.backend)

    # ── Backend initialisation ───────────────────────────────────────────────

    def _init_backend(self) -> AerSimulator:
        """
        Attempt FakeFezV2 noise model; fall back to custom depolarising model.
        """
        try:
            from qiskit_ibm_runtime.fake_provider import FakeFez
            from qiskit_aer.noise import NoiseModel
            fake    = FakeFez()
            nm      = NoiseModel.from_backend(fake)
            backend = AerSimulator(noise_model=nm)   # No fixed seed → genuine random noise per job
            print("[Executor] Using FakeFezV2 noise model.")
            return backend
        except Exception as e:
            print(f"[Executor] FakeFezV2 unavailable ({e}). "
                  "Using custom depolarising noise.")
            return self._custom_noise_backend()

    def _custom_noise_backend(self) -> AerSimulator:
        from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                       ReadoutError)
        nm = NoiseModel()
        # Single-qubit gate error
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.003, 1), ['rx', 'ry', 'rz', 'h', 'z', 'sdg'])
        # Two-qubit gate error
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.015, 2), ['cx', 'cz'])
        # Readout error
        p_err = 0.02
        nm.add_all_qubit_readout_error(ReadoutError([[1-p_err, p_err],
                                                      [p_err, 1-p_err]]))
        return AerSimulator(noise_model=nm)   # No fixed seed → genuine random noise

    # ── Batch execution (key efficiency method) ───────────────────────────────

    def batch_run(self, circuits: list,
                  basis_rotations: list = None) -> list:
        """
        Run a list of circuits in ONE sampler job (batched).

        Parameters
        ----------
        circuits : list of QuantumCircuit
        basis_rotations : list of list of dict, optional
            Per-circuit basis rotation specs, e.g.:
            [[{'qubit': 0, 'basis': 'X'}, {'qubit': 1, 'basis': 'Y'}], ...]
            Supported bases: 'Z' (none), 'X' (H gate), 'Y' (Sdg·H gate).

        Returns
        -------
        list of dict : counts dictionaries, one per circuit.
        """
        meas_circuits = []
        for i, qc in enumerate(circuits):
            mq = qc.copy()
            # Apply basis rotations if provided
            if basis_rotations and i < len(basis_rotations):
                for rot in basis_rotations[i]:
                    qubit = rot['qubit']
                    basis = rot.get('basis', 'Z')
                    if basis == 'X':
                        mq.h(qubit)
                    elif basis == 'Y':
                        mq.sdg(qubit)
                        mq.h(qubit)
            mq.measure_all()
            meas_circuits.append(
                transpile(mq, self.backend, optimization_level=0))

        job    = self.sampler.run(meas_circuits, shots=self.shots)
        result = job.result()
        return [result[i].data.meas.get_counts()
                for i in range(len(meas_circuits))]

    # ── Single-circuit helpers ────────────────────────────────────────────────

    def measure_zz(self, qc: QuantumCircuit) -> float:
        """Measure ⟨ZZ⟩ from a single circuit (Z-basis, both qubits)."""
        counts = self.batch_run([qc])[0]
        return self._counts_to_zz(counts, qc.num_qubits)

    def measure_z_per_qubit(self, qc: QuantumCircuit) -> tuple:
        """
        Measure ⟨Z⟩ per qubit from a single circuit run.
        Returns (z_q0, z_q1).
        """
        counts = self.batch_run([qc])[0]
        return self._counts_to_z_per_qubit(counts, qc.num_qubits)

    def measure_pauli_qubit(self, qc: QuantumCircuit,
                             qubit: int, basis: str) -> float:
        """Measure ⟨P⟩ for a single qubit in basis X, Y, or Z."""
        rot = [{'qubit': qubit, 'basis': basis}]
        counts = self.batch_run([qc], basis_rotations=[rot])[0]
        return self._counts_to_z_qubit(counts, qubit, qc.num_qubits)

    def get_ideal_zz(self, qc: QuantumCircuit) -> float:
        """Noiseless ⟨ZZ⟩ via Statevector (exact, no shots)."""
        sv  = Statevector(qc)
        val = sv.expectation_value(SparsePauliOp("ZZ"))
        return float(val.real)

    def get_ideal_bloch(self, qc: QuantumCircuit,
                         qubit: int) -> np.ndarray:
        """
        Noiseless Bloch vector (X, Y, Z) for one qubit via Statevector.
        """
        sv   = Statevector(qc)
        ops  = ["IX", "IY", "IZ"] if qubit == 0 else ["XI", "YI", "ZI"]
        vals = [sv.expectation_value(SparsePauliOp(op)).real for op in ops]
        return np.array(vals, dtype=float)

    # ── Counts → expectation value converters ────────────────────────────────

    def _counts_to_zz(self, counts: dict, num_qubits: int) -> float:
        """
        ⟨ZZ⟩ from measurement counts.
        ZZ eigenvalue: +1 for |00⟩ or |11⟩, −1 for |01⟩ or |10⟩.
        = (−1)^(b₀ ⊕ b₁)  where b₀, b₁ ∈ {0,1}.
        """
        total, exp = sum(counts.values()), 0.0
        for bitstring, cnt in counts.items():
            bits = bitstring.zfill(num_qubits)
            # Qiskit: rightmost bit = qubit 0
            b0 = int(bits[-1])
            b1 = int(bits[-2]) if num_qubits >= 2 else 0
            exp += ((-1) ** (b0 ^ b1)) * cnt
        return exp / total

    def _counts_to_z_per_qubit(self, counts: dict,
                                 num_qubits: int) -> tuple:
        """
        Per-qubit ⟨Z⟩ from measurement counts.
        ⟨Zᵢ⟩ = P(0 on qubit i) − P(1 on qubit i).
        """
        total   = sum(counts.values())
        z_sum   = np.zeros(num_qubits)
        for bitstring, cnt in counts.items():
            bits = bitstring.zfill(num_qubits)
            for q in range(num_qubits):
                b = int(bits[-(q + 1)])   # rightmost bit = qubit 0
                z_sum[q] += (1 - 2 * b) * cnt
        z_vals = z_sum / total
        return tuple(z_vals)

    def _counts_to_z_qubit(self, counts: dict,
                             target_qubit: int, num_qubits: int) -> float:
        """⟨Z⟩ for a single qubit from counts."""
        return self._counts_to_z_per_qubit(counts, num_qubits)[target_qubit]
