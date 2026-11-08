"""
executor.py — Unified Quantum Execution Engine
═══════════════════════════════════════════════
Qiskit 2.2.1 compatible.

Key design: batch_run() sends ALL circuits in one job → major speed-up.

Bit-ordering convention (Qiskit):
    Bitstring is big-endian in print order but stored little-endian:
    rightmost character in counts key = qubit 0.
    e.g. "10" means qubit 1 = |1⟩, qubit 0 = |0⟩.

Qiskit 2.x changes handled here:
    - FakeFez import path: qiskit_ibm_runtime.fake_provider (still valid in 2.x)
    - generate_preset_pass_manager replaces direct transpile() for ISA circuits
    - SamplerV2.from_backend() unchanged
    - find_bit() still valid in 2.2.1
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

from config import SHOTS, SEED


class QuantumExecutor:
    """
    Unified quantum execution engine for Qiskit 2.x.

    All noisy measurements use SamplerV2 on an AerSimulator with FakeFezV2
    noise model (or a custom depolarising noise model as fallback).

    Batch execution: call batch_run() with a list of circuits to minimise
    job-submission overhead — critical for the 240+ circuit calibration loops.
    """

    def __init__(self, shots: int = SHOTS, seed: int = SEED):
        self.shots   = shots
        self.seed    = seed
        self.backend = self._init_backend()
        self.sampler = SamplerV2.from_backend(self.backend)
        # Pass manager for ISA transpilation (Qiskit 2.x requirement)
        self._pm = generate_preset_pass_manager(
            optimization_level=0, backend=self.backend)

    # ── Backend initialisation ───────────────────────────────────────────────

    def _init_backend(self) -> AerSimulator:
        """
        Try FakeFez noise model first; fall back to custom depolarising model.
        FakeFez import works in both qiskit-ibm-runtime 0.29–0.35+.
        """
        try:
            from qiskit_ibm_runtime.fake_provider import FakeFez
            from qiskit_aer.noise import NoiseModel
            fake    = FakeFez()
            nm      = NoiseModel.from_backend(fake)
            backend = AerSimulator(noise_model=nm)
            print("[Executor] Using FakeFezV2 noise model.")
            return backend
        except Exception as e:
            print(f"[Executor] FakeFez unavailable ({e}). "
                  "Using custom depolarising noise.")
            return self._custom_noise_backend()

    def _custom_noise_backend(self) -> AerSimulator:
        from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                       ReadoutError)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.003, 1), ['rx', 'ry', 'rz', 'h', 'z', 'sdg'])
        nm.add_all_qubit_quantum_error(
            depolarizing_error(0.015, 2), ['cx', 'cz'])
        p_err = 0.02
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_err, p_err], [p_err, 1 - p_err]]))
        return AerSimulator(noise_model=nm)

    # ── Transpile to ISA (Qiskit 2.x requires ISA circuits for SamplerV2) ────

    def _to_isa(self, qc: QuantumCircuit) -> QuantumCircuit:
        """
        Convert a logical circuit to ISA (Instruction Set Architecture) form
        required by Qiskit 2.x SamplerV2. Adds measurements if absent.
        """
        mq = qc.copy()
        mq.measure_all()
        return self._pm.run(mq)

    def _to_isa_with_basis(self, qc: QuantumCircuit,
                            basis_rots: list = None) -> QuantumCircuit:
        """
        Apply optional basis rotations then transpile to ISA.
        basis_rots: list of {'qubit': int, 'basis': 'X'|'Y'|'Z'}
        """
        mq = qc.copy()
        if basis_rots:
            for rot in basis_rots:
                qubit = rot['qubit']
                basis = rot.get('basis', 'Z')
                if basis == 'X':
                    mq.h(qubit)
                elif basis == 'Y':
                    mq.sdg(qubit)
                    mq.h(qubit)
        mq.measure_all()
        return self._pm.run(mq)

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

        Returns
        -------
        list of dict : counts dictionaries, one per circuit.
        """
        isa_circuits = []
        for i, qc in enumerate(circuits):
            rots = (basis_rotations[i]
                    if basis_rotations and i < len(basis_rotations)
                    else None)
            isa_circuits.append(self._to_isa_with_basis(qc, rots))

        job    = self.sampler.run(isa_circuits, shots=self.shots)
        result = job.result()

        counts_list = []
        for i in range(len(isa_circuits)):
            pub_result = result[i]
            # Qiskit 2.x: data is a DataBin, classical register named 'meas'
            try:
                counts = pub_result.data.meas.get_counts()
            except AttributeError:
                # Fallback: try first classical register
                creg_name = list(pub_result.data)[0]
                counts = getattr(pub_result.data, creg_name).get_counts()
            counts_list.append(counts)

        return counts_list

    # ── Ideal (noiseless) helpers ─────────────────────────────────────────────

    def get_ideal_zz(self, qc: QuantumCircuit) -> float:
        """Noiseless ⟨ZZ⟩ via Statevector (exact, no shots)."""
        sv  = Statevector(qc)
        val = sv.expectation_value(SparsePauliOp("ZZ"))
        return float(val.real)

    def get_ideal_bloch(self, qc: QuantumCircuit,
                         qubit: int) -> np.ndarray:
        """
        Noiseless Bloch vector (X, Y, Z) for one qubit via Statevector.
        Note: SparsePauliOp string is in reversed qubit order (q0 = rightmost).
        """
        sv = Statevector(qc)
        if qubit == 0:
            ops = ["IX", "IY", "IZ"]   # acts on qubit 0
        else:
            ops = ["XI", "YI", "ZI"]   # acts on qubit 1
        vals = [float(sv.expectation_value(SparsePauliOp(op)).real)
                for op in ops]
        return np.array(vals, dtype=float)

    # ── Single-circuit helpers ────────────────────────────────────────────────

    def measure_zz(self, qc: QuantumCircuit) -> float:
        """Measure ⟨ZZ⟩ from a single circuit (Z-basis, both qubits)."""
        counts = self.batch_run([qc])[0]
        return self._counts_to_zz(counts, qc.num_qubits)

    def measure_z_per_qubit(self, qc: QuantumCircuit) -> tuple:
        """Measure ⟨Z⟩ per qubit. Returns (z_q0, z_q1)."""
        counts = self.batch_run([qc])[0]
        return self._counts_to_z_per_qubit(counts, qc.num_qubits)

    def measure_pauli_qubit(self, qc: QuantumCircuit,
                             qubit: int, basis: str) -> float:
        """Measure ⟨P⟩ for a single qubit in basis X, Y, or Z."""
        rot = [{'qubit': qubit, 'basis': basis}]
        counts = self.batch_run([qc], basis_rotations=[rot])[0]
        return self._counts_to_z_qubit(counts, qubit, qc.num_qubits)

    # ── Counts → expectation value converters ────────────────────────────────

    def _counts_to_zz(self, counts: dict, num_qubits: int) -> float:
        """
        ⟨ZZ⟩ from measurement counts.
        ZZ eigenvalue: +1 for |00⟩ or |11⟩, −1 for |01⟩ or |10⟩.
        Qiskit: rightmost bit = qubit 0.
        """
        total, exp_val = sum(counts.values()), 0.0
        for bitstring, cnt in counts.items():
            bits = bitstring.zfill(num_qubits)
            b0 = int(bits[-1])
            b1 = int(bits[-2]) if num_qubits >= 2 else 0
            exp_val += ((-1) ** (b0 ^ b1)) * cnt
        return exp_val / total if total > 0 else 0.0

    def _counts_to_z_per_qubit(self, counts: dict,
                                 num_qubits: int) -> tuple:
        """
        Per-qubit ⟨Z⟩ from measurement counts.
        ⟨Zᵢ⟩ = P(0 on qubit i) − P(1 on qubit i).
        """
        total  = sum(counts.values())
        z_sum  = np.zeros(num_qubits)
        for bitstring, cnt in counts.items():
            bits = bitstring.zfill(num_qubits)
            for q in range(num_qubits):
                b = int(bits[-(q + 1)])
                z_sum[q] += (1 - 2 * b) * cnt
        z_vals = z_sum / total if total > 0 else z_sum
        return tuple(z_vals)

    def _counts_to_z_qubit(self, counts: dict,
                             target_qubit: int, num_qubits: int) -> float:
        """⟨Z⟩ for a single qubit from counts."""
        return self._counts_to_z_per_qubit(counts, num_qubits)[target_qubit]
