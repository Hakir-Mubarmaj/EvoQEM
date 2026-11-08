"""
qpd_lib.py — Quasi-Probability Decomposition (QPD) of the CZ gate
═══════════════════════════════════════════════════════════════════
Mathematical basis:
    CZ = e^{-iπ/4 (I−Z₀)(I−Z₁)}
       ≡ ½(I⊗I) + ½(Z⊗I) + ½(I⊗Z) − ½(Z⊗Z)

Each term is a local tensor-product operator with weight ωᵢ ∈ {+0.5, −0.5}.
The one-norm γ = Σᵢ|ωᵢ| = 2.0, giving a sampling overhead of γ² = 4×.

BUG FIX (vs original thesis code):
    The original code APPENDED Z gates after the full circuit.
    This code REBUILDS the circuit instruction-by-instruction, REMOVING
    the CZ gate and INSERTING the local Pauli replacement at the same position.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate, IGate
import numpy as np

# CZ Operator-Schmidt decomposition:
# (weight, op_on_q_left, op_on_q_right)
# q_left = first element of cut_qubit_pair
# q_right = second element of cut_qubit_pair
QPD_TERMS = [
    (+0.5, 'I', 'I'),   # Variant 1: Identity on both — weight +0.5
    (+0.5, 'Z', 'I'),   # Variant 2: Z on q0, I on q1 — weight +0.5
    (+0.5, 'I', 'Z'),   # Variant 3: I on q0, Z on q1 — weight +0.5
    (-0.5, 'Z', 'Z'),   # Variant 4: Z on q0, Z on q1 — weight −0.5
]

GAMMA = sum(abs(w) for w, _, _ in QPD_TERMS)   # One-norm = 2.0


def build_target_circuit(theta0: float = 0.6, theta1: float = 0.8) -> QuantumCircuit:
    """
    Build the two-qubit target circuit: RX(θ₀) ⊗ RX(θ₁) followed by CZ.
    Ideal <ZZ> = cos(θ₀)·cos(θ₁) [CZ is ZZ-transparent for Z-diagonal inputs].
    """
    qc = QuantumCircuit(2, name="target")
    qc.rx(theta0, 0)
    qc.rx(theta1, 1)
    qc.cz(0, 1)
    return qc


def build_qpd_variants(qc: QuantumCircuit,
                        cut_qubit_pair: tuple) -> list:
    """
    Properly decompose a CZ gate via QPD.

    The CZ gate on `cut_qubit_pair` is REMOVED from the circuit and replaced
    by local Pauli operations (I or Z) at exactly the same position.

    Parameters
    ----------
    qc : QuantumCircuit
        Original circuit containing exactly one CZ on cut_qubit_pair.
    cut_qubit_pair : tuple (q0_idx, q1_idx)
        Indices of the two qubits sharing the CZ gate.

    Returns
    -------
    list of (QuantumCircuit, float)
        Four (variant_circuit, weight) tuples, weights ∈ {+0.5, −0.5}.
    """
    q0_idx, q1_idx = cut_qubit_pair
    variants = []

    for weight, op0, op1 in QPD_TERMS:
        new_qc = QuantumCircuit(qc.num_qubits)
        cz_replaced = False

        for instruction in qc.data:
            gate      = instruction.operation
            qubit_ids = [qc.find_bit(q).index for q in instruction.qubits]

            # Detect the target CZ gate (replace only the FIRST occurrence)
            is_target_cz = (
                gate.name == 'cz'
                and set(qubit_ids) == {q0_idx, q1_idx}
                and not cz_replaced
            )

            if is_target_cz:
                cz_replaced = True
                # Insert local Pauli replacements at this position
                if op0 == 'Z':
                    new_qc.z(q0_idx)
                if op1 == 'Z':
                    new_qc.z(q1_idx)
                # If both are 'I', nothing is inserted (identity = no gate)
            else:
                new_qc.append(gate, qubit_ids)

        if not cz_replaced:
            raise ValueError(
                f"No CZ gate found on qubits {cut_qubit_pair}. "
                "Ensure the circuit contains exactly one CZ at the cut."
            )

        variants.append((new_qc, weight))

    return variants


def reconstruct_expectation(variant_results: list,
                             weights: list) -> float:
    """
    QPD reconstruction formula: <O>_original = Σᵢ ωᵢ · <O>_variant_i

    Parameters
    ----------
    variant_results : list of float
        Expectation value <O> measured on each variant circuit.
    weights : list of float
        QPD weights (same order as variants), ωᵢ ∈ {+0.5, −0.5}.

    Returns
    -------
    float : Reconstructed expectation value.
    """
    return float(sum(w * r for w, r in zip(weights, variant_results)))
