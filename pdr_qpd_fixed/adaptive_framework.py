"""
adaptive_framework.py — Adaptive QPD-PDR Decision Framework
════════════════════════════════════════════════════════════
Implements Theorem 1 (Section III) of the paper:

    QPD+PDR wins iff:  γ · ε_fragment(O) < ε_full_circuit(O)

    where ε_full_circuit(O) = ε_circuit_base(O) + n_SWAP · ε_SWAP
          ε_fragment(O)      = Σ_{g ∈ fragment} ε_g · I(g, O)
          I(g, O) ∈ {0,1}    observable susceptibility indicator

The framework:
    1. Calibrates ε_1q, ε_2q from the backend (once per session)
    2. Estimates n_SWAP using the Qiskit transpiler
    3. Applies Theorem 1 to decide: 'qpd' or 'direct'
    4. Executes the chosen strategy and returns the mitigated result

Key insight (Remark 1):
    The decision is INDEPENDENT of β — it depends only on
    hardware noise rates and routing topology.
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

warnings.filterwarnings('ignore')


# ── Observable Susceptibility Table ─────────────────────────────────────────
# I(gate_name, observable) ∈ {0, 1}
# 0 = gate noise is transparent to observable (commutes with all Kraus ops)
# 1 = gate noise perturbs the observable expectation value
#
# Derivation:
#   CZ is diagonal → its dephasing noise commutes with ZZ → I(CZ, ZZ) = 0
#   Z gate is diagonal → I(Z, ZZ) = 0
#   RX noise (depolarising) has X,Y components → I(RX, ZZ) = 1
#   CNOT noise (depolarising) has X,Y → does not commute with ZZ → I(CX, ZZ) = 1
#   SWAP = 3 × CNOT → I(SWAP, ZZ) = 1

SUSCEPTIBILITY = {
    # (gate_name, observable) → I value
    ('rx',   'ZZ'): 1.0,   # RX noise affects Z-basis
    ('ry',   'ZZ'): 1.0,
    ('rz',   'ZZ'): 0.0,   # RZ is diagonal → transparent
    ('x',    'ZZ'): 1.0,
    ('y',    'ZZ'): 1.0,
    ('z',    'ZZ'): 0.0,   # Z is diagonal → transparent
    ('h',    'ZZ'): 1.0,
    ('s',    'ZZ'): 0.0,
    ('sdg',  'ZZ'): 0.0,
    ('cz',   'ZZ'): 0.0,   # CZ diagonal → transparent to ZZ
    ('cx',   'ZZ'): 1.0,   # CNOT: X on target → non-transparent
    ('ecr',  'ZZ'): 1.0,
    ('swap', 'ZZ'): 1.0,   # SWAP = 3 CNOT → non-transparent
    # For XX observable: CZ is non-transparent (affects X basis)
    ('cz',   'XX'): 1.0,
    ('cx',   'XX'): 1.0,
    ('rx',   'XX'): 0.0,
    ('z',    'XX'): 1.0,
}

GAMMA = 2.0  # QPD 1-norm for single CZ cut


# ── Data container for calibration results ───────────────────────────────────

@dataclass
class HardwareProfile:
    """Noise parameters extracted from a backend."""
    eps_1q:    float   # Mean single-qubit gate error
    eps_2q:    float   # Mean two-qubit gate error
    eps_swap:  float   # SWAP error = 1-(1-eps_2q)^3
    gamma:     float   # QPD 1-norm (fixed = 2.0 for single cut)
    backend_name: str  # For logging

    def summary(self) -> str:
        return (f"Backend: {self.backend_name} | "
                f"ε_1q={self.eps_1q:.6f} | "
                f"ε_2q={self.eps_2q:.6f} | "
                f"ε_SWAP={self.eps_swap:.6f}")


# ── Hardware Calibration ─────────────────────────────────────────────────────

def calibrate_hardware(backend) -> HardwareProfile:
    """
    Extract ε_1q and ε_2q from the backend's properties.
    ε_SWAP is derived analytically as 1 - (1 - ε_2q)³.

    Parameters
    ----------
    backend : AerSimulator or IBMBackend
        The target quantum backend.

    Returns
    -------
    HardwareProfile
    """
    try:
        props = backend.properties()

        # ── Single-qubit gate error (SX or X gate) ───────────────────────
        sq_errors = []
        for gate_name in ['sx', 'x', 'rx']:
            for q in range(backend.num_qubits):
                try:
                    err = props.gate_error(gate_name, q)
                    if err is not None and 0 < err < 1:
                        sq_errors.append(err)
                except Exception:
                    continue

        eps_1q = float(np.mean(sq_errors)) if sq_errors else 0.000764

        # ── Two-qubit gate error ──────────────────────────────────────────
        tq_errors = []
        for gate_name in ['cx', 'ecr', 'cz', 'rzz']:
            for gate in props.gates:
                if gate.gate == gate_name:
                    for param in gate.parameters:
                        if param.name == 'gate_error':
                            val = param.value
                            if val is not None and 0 < val < 1:
                                tq_errors.append(val)

        eps_2q = float(np.mean(tq_errors)) if tq_errors else 0.010157

    except Exception:
        # FakeFez or backends without .properties() — use known defaults
        eps_1q = 0.000764
        eps_2q = 0.010157

    eps_swap = 1.0 - (1.0 - eps_2q) ** 3
    name = getattr(backend, 'name', type(backend).__name__)

    return HardwareProfile(
        eps_1q=eps_1q,
        eps_2q=eps_2q,
        eps_swap=eps_swap,
        gamma=GAMMA,
        backend_name=name,
    )


# ── SWAP Count Estimation ────────────────────────────────────────────────────

def estimate_swap_count(circuit: QuantumCircuit, backend,
                        qubit_layout: Optional[list] = None) -> int:
    """
    Estimate the number of SWAP gates the transpiler would insert
    to execute `circuit` on `backend` with the given qubit layout.

    Parameters
    ----------
    circuit : QuantumCircuit
        The undecomposed logical circuit.
    backend : AerSimulator or IBMBackend
        Target backend (used for coupling map).
    qubit_layout : list of int, optional
        Physical qubit indices to map logical qubits onto.
        If None, transpiler chooses automatically.

    Returns
    -------
    int : Number of SWAP gates in the transpiled circuit.
    """
    try:
        pm_kwargs = dict(optimization_level=1, backend=backend)
        if qubit_layout is not None:
            pm_kwargs['initial_layout'] = qubit_layout

        pm         = generate_preset_pass_manager(**pm_kwargs)
        transpiled = pm.run(circuit)
        n_swap     = sum(
            1 for inst in transpiled.data
            if inst.operation.name in ('swap', 'SWAP')
        )
        return n_swap
    except Exception:
        return 0


# ── Observable-Specific Error Computation ────────────────────────────────────

def circuit_error(circuit: QuantumCircuit, profile: HardwareProfile,
                   observable: str = 'ZZ',
                   extra_swap: int = 0) -> float:
    """
    Compute the observable-specific effective gate error of a circuit:

        ε(O) = Σ_{g ∈ C} ε_g · I(g, O) + extra_swap · ε_SWAP

    Parameters
    ----------
    circuit : QuantumCircuit
    profile : HardwareProfile
    observable : str  e.g. 'ZZ', 'XX'
    extra_swap : int  Additional SWAP gates from routing

    Returns
    -------
    float : effective error rate
    """
    total = 0.0
    for inst in circuit.data:
        name = inst.operation.name
        n_qubits = len(inst.qubits)
        # Base gate error
        base_err = profile.eps_2q if n_qubits >= 2 else profile.eps_1q
        # Susceptibility
        key = (name, observable)
        susceptibility = SUSCEPTIBILITY.get(key, 1.0)  # default: non-transparent
        total += base_err * susceptibility

    # Add routing SWAP contribution (always non-transparent: I=1)
    total += extra_swap * profile.eps_swap
    return total


# ── Theorem 1 Decision Logic ─────────────────────────────────────────────────

@dataclass
class DecisionReport:
    """Full audit trail of the Theorem 1 decision."""
    decision:        str     # 'qpd' or 'direct'
    lhs:             float   # γ · ε_fragment(O)
    rhs:             float   # ε_full_circuit(O)
    n_swap:          int
    eps_fragment:    float
    eps_full:        float
    eps_circuit_base: float
    observable:      str
    gamma:           float
    margin:          float = 0.0   # rhs - lhs; positive → QPD clearly wins

    def __str__(self) -> str:
        margin_note = (
            "  ⚠ Margin is near zero (|margin| < ε_2q): "
            "empirical result may differ from Theorem 1 prediction "
            "(regression gain not captured by noise-propagation model)."
            if abs(self.margin) < 0.002 else ""
        )
        lines = [
            f"  Observable:    {self.observable}",
            f"  n_SWAP:        {self.n_swap}",
            f"  ε_fragment(O): {self.eps_fragment:.6f}",
            f"  ε_circuit_base:{self.eps_circuit_base:.6f}",
            f"  ε_full(O):     {self.eps_full:.6f}",
            f"  LHS = γ·ε_frag:{self.lhs:.6f}",
            f"  RHS = ε_full:  {self.rhs:.6f}",
            f"  Margin (RHS-LHS): {self.margin:+.6f}",
            f"  Decision:      {self.decision.upper()} "
            f"({'LHS < RHS' if self.decision == 'qpd' else 'LHS ≥ RHS'})",
        ]
        if margin_note:
            lines.append(margin_note)
        return "\n".join(lines)


def theorem1_decide(full_circuit: QuantumCircuit,
                    fragment_circuit: QuantumCircuit,
                    profile: HardwareProfile,
                    n_swap: int,
                    observable: str = 'ZZ') -> DecisionReport:
    """
    Apply Theorem 1 to decide: use QPD+PDR or Direct+PDR?

    Parameters
    ----------
    full_circuit    : The complete (uncut) logical circuit
    fragment_circuit: A representative QPD variant circuit
    profile         : HardwareProfile from calibrate_hardware()
    n_swap          : SWAP count from estimate_swap_count()
    observable      : Pauli observable string, e.g. 'ZZ'

    Returns
    -------
    DecisionReport
    """
    eps_frag  = circuit_error(fragment_circuit, profile, observable, extra_swap=0)
    eps_base  = circuit_error(full_circuit,     profile, observable, extra_swap=0)
    eps_full  = eps_base + n_swap * profile.eps_swap

    lhs = profile.gamma * eps_frag
    rhs = eps_full

    decision = 'qpd' if lhs < rhs else 'direct'

    return DecisionReport(
        decision=decision,
        lhs=lhs,
        rhs=rhs,
        n_swap=n_swap,
        eps_fragment=eps_frag,
        eps_full=eps_full,
        eps_circuit_base=eps_base,
        observable=observable,
        gamma=profile.gamma,
        margin=rhs - lhs,
    )


# ── Full Adaptive Execution Pipeline ─────────────────────────────────────────

class AdaptiveQPDPDR:
    """
    End-to-end adaptive QPD-PDR framework.

    Usage
    -----
    framework = AdaptiveQPDPDR(executor, backend)
    result = framework.run(target_circuit, qubit_layout=[0, 5])

    Result contains:
        mitigated_value : float
        decision        : 'qpd' or 'direct'
        report          : DecisionReport (full audit trail)
    """

    def __init__(self, executor, backend, observable: str = 'ZZ',
                 verbose: bool = True):
        """
        Parameters
        ----------
        executor   : QuantumExecutor from executor.py
        backend    : The AerSimulator/IBMBackend used for execution
        observable : Pauli string to measure (default 'ZZ')
        verbose    : Print decision report
        """
        self.executor   = executor
        self.backend    = backend
        self.observable = observable
        self.verbose    = verbose

        # Calibrate hardware once at init
        self.profile = calibrate_hardware(backend)
        if verbose:
            print(f"[AdaptiveQPDPDR] Hardware calibrated.")
            print(f"  {self.profile.summary()}")

    def run(self, target_circuit: QuantumCircuit,
            cut_qubit_pair: tuple = (0, 1),
            qubit_layout: Optional[list] = None) -> dict:
        """
        Full adaptive pipeline:

            1. Estimate SWAP count for direct execution
            2. Apply Theorem 1 decision
            3. Execute chosen strategy with PDR mitigation
            4. Return mitigated result + full audit trail

        Parameters
        ----------
        target_circuit : QuantumCircuit
            The full logical circuit (with entangling gate).
        cut_qubit_pair : tuple
            Qubits sharing the entangling gate for QPD.
        qubit_layout : list of int, optional
            Physical qubit mapping for routing estimation.

        Returns
        -------
        dict with keys:
            mitigated_value : float
            decision        : 'qpd' or 'direct'
            noisy_value     : float (before mitigation)
            report          : DecisionReport
            ideal_value     : float (noiseless, for validation)
        """
        from qpd_lib import build_qpd_variants, reconstruct_expectation
        from calibration import generate_calibration_data
        from pdr_regression import PDRRegressor

        # ── Step 1: Estimate routing overhead ────────────────────────────
        n_swap = estimate_swap_count(target_circuit, self.backend, qubit_layout)

        # ── Step 2: Get a representative fragment for error estimation ────
        variants = build_qpd_variants(target_circuit, cut_qubit_pair)
        rep_fragment = variants[0][0]  # Variant 1 (II) as representative

        # ── Step 3: Apply Theorem 1 ───────────────────────────────────────
        report = theorem1_decide(
            full_circuit=target_circuit,
            fragment_circuit=rep_fragment,
            profile=self.profile,
            n_swap=n_swap,
            observable=self.observable,
        )

        if self.verbose:
            print(f"\n[Theorem 1 Decision]")
            print(report)

        # ── Step 4: Execute chosen strategy ──────────────────────────────
        ideal_value = self.executor.get_ideal_zz(target_circuit)

        if report.decision == 'qpd':
            mitigated, noisy = self._run_qpd_pdr(
                target_circuit, variants)
        else:
            mitigated, noisy = self._run_direct_pdr(target_circuit)

        return {
            'mitigated_value': mitigated,
            'noisy_value'    : noisy,
            'decision'       : report.decision,
            'report'         : report,
            'ideal_value'    : ideal_value,
        }

    # ── QPD+PDR execution path ────────────────────────────────────────────

    def _run_qpd_pdr(self, target_circuit, variants) -> tuple:
        """Run QPD variants with per-variant PDR regression."""
        from calibration import generate_calibration_data
        from pdr_regression import PDRRegressor
        from qpd_lib import reconstruct_expectation

        noisy_vals    = []
        mitigated_vals = []
        weights       = [w for _, w in variants]

        for v_idx, (v_qc, weight) in enumerate(variants):
            # Calibration
            cal_data = generate_calibration_data(
                v_qc, self.executor, mode='scalar', K=20, seed=42 + v_idx)

            # Fit linear PDR
            reg = PDRRegressor(mode='linear', seed=42 + v_idx).fit(cal_data)

            # Target measurement
            counts = self.executor.batch_run([v_qc])[0]
            z0, z1 = self.executor._counts_to_z_per_qubit(
                counts, v_qc.num_qubits)
            noisy_zz = z0 * z1

            noisy_vals.append(noisy_zz)
            mitigated_vals.append(reg.predict(noisy_zz))

        noisy_recon    = reconstruct_expectation(noisy_vals, weights)
        mitigated_recon = reconstruct_expectation(mitigated_vals, weights)
        return mitigated_recon, noisy_recon

    # ── Direct+PDR execution path ─────────────────────────────────────────

    def _run_direct_pdr(self, target_circuit) -> tuple:
        """Run full circuit with PDR regression (no QPD)."""
        from calibration import generate_calibration_data
        from pdr_regression import PDRRegressor
        from qiskit.circuit.library import RXGate

        n_rx = sum(1 for i in target_circuit.data
                   if i.operation.name == 'rx')
        rng = np.random.default_rng(42)

        noisy_train, ideal_train = [], []
        cal_circuits = []
        for _ in range(20):
            delta = rng.uniform(-0.2, 0.2, size=max(n_rx, 1))
            new_qc = QuantumCircuit(target_circuit.num_qubits)
            rx_count = 0
            for inst in target_circuit.data:
                gate   = inst.operation
                try:
                    qubits = [target_circuit.find_bit(q).index
                              for q in inst.qubits]
                except AttributeError:
                    qubits = [target_circuit.qubits.index(q)
                              for q in inst.qubits]
                if gate.name == 'rx' and rx_count < len(delta):
                    new_qc.append(RXGate(gate.params[0] + delta[rx_count]),
                                   qubits)
                    rx_count += 1
                else:
                    new_qc.append(gate, qubits)
            cal_circuits.append(new_qc)
            ideal_train.append(self.executor.get_ideal_zz(new_qc))

        all_counts = self.executor.batch_run(cal_circuits)
        for counts in all_counts:
            noisy_train.append(
                self.executor._counts_to_zz(counts, target_circuit.num_qubits))

        cal_data = {
            'noisy_zz': np.array(noisy_train),
            'ideal_zz': np.array(ideal_train),
        }
        reg = PDRRegressor(mode='linear', seed=42).fit(cal_data)

        target_counts = self.executor.batch_run([target_circuit])[0]
        noisy_target  = self.executor._counts_to_zz(
            target_counts, target_circuit.num_qubits)
        mitigated     = reg.predict(noisy_target)

        return mitigated, noisy_target
