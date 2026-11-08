"""
qpd_lib.py — Quasi-Probability Decomposition (QPD) of the CZ gate
═══════════════════════════════════════════════════════════════════
Mathematical basis:
    CZ = e^{-iπ/4 (I−Z₀)(I−Z₁)}
       ≡ ½(I⊗I) + ½(Z⊗I) + ½(I⊗Z) − ½(Z⊗Z)

Each term is a local tensor-product operator with weight ωᵢ ∈ {+0.5, −0.5}.
The one-norm γ = Σᵢ|ωᵢ| = 2.0, giving a sampling overhead of γ² = 4×.

QPD reconstruction:
    ⟨O⟩_original = Σᵢ ωᵢ · ⟨O⟩_variant_i

Qiskit 2.x compatibility:
    - Uses qc.data[i].operation and qc.data[i].qubits (unchanged API)
    - find_bit() replaced with direct qubit index lookup for safety
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate, IGate
import numpy as np

# CZ Operator-Schmidt decomposition:
# (weight, op_on_q0, op_on_q1)
QPD_TERMS = [
    (+0.5, 'I', 'I'),   # Variant 1: I⊗I  weight +0.5
    (+0.5, 'Z', 'I'),   # Variant 2: Z⊗I  weight +0.5
    (+0.5, 'I', 'Z'),   # Variant 3: I⊗Z  weight +0.5
    (-0.5, 'Z', 'Z'),   # Variant 4: Z⊗Z  weight −0.5
]

GAMMA = sum(abs(w) for w, _, _ in QPD_TERMS)   # One-norm = 2.0


def _qubit_indices(qc: QuantumCircuit, instruction) -> list:
    """
    Return physical qubit indices for an instruction.
    Compatible with Qiskit 1.x and 2.x.
    """
    try:
        # Qiskit 1.x / 2.x: find_bit() still valid
        return [qc.find_bit(q).index for q in instruction.qubits]
    except AttributeError:
        # Fallback: qubit_indices directly
        return [qc.qubits.index(q) for q in instruction.qubits]


def build_target_circuit(theta0: float = 0.6,
                          theta1: float = 0.8) -> QuantumCircuit:
    """
    Build the two-qubit target circuit: RX(θ₀) ⊗ RX(θ₁) followed by CZ.
    Ideal ⟨ZZ⟩ = cos(θ₀)·cos(θ₁)  [CZ is ZZ-transparent for Z-diagonal inputs].
    """
    qc = QuantumCircuit(2, name="target")
    qc.rx(theta0, 0)
    qc.rx(theta1, 1)
    qc.cz(0, 1)
    return qc


def build_qpd_variants(qc: QuantumCircuit,
                        cut_qubit_pair: tuple) -> list:
    """
    Decompose the CZ gate on cut_qubit_pair via QPD.

    The CZ is REMOVED and replaced by local Pauli operations (I or Z)
    at the same position — not appended at the end.

    Parameters
    ----------
    qc : QuantumCircuit
        Original circuit with exactly one CZ on cut_qubit_pair.
    cut_qubit_pair : tuple (q0_idx, q1_idx)

    Returns
    -------
    list of (QuantumCircuit, float)
        Four (variant_circuit, weight) tuples.
    """
    q0_idx, q1_idx = cut_qubit_pair
    variants = []

    for weight, op0, op1 in QPD_TERMS:
        new_qc      = QuantumCircuit(qc.num_qubits)
        cz_replaced = False

        for instruction in qc.data:
            gate      = instruction.operation
            qubit_ids = _qubit_indices(qc, instruction)

            is_target_cz = (
                gate.name == 'cz'
                and set(qubit_ids) == {q0_idx, q1_idx}
                and not cz_replaced
            )

            if is_target_cz:
                cz_replaced = True
                if op0 == 'Z':
                    new_qc.z(q0_idx)
                if op1 == 'Z':
                    new_qc.z(q1_idx)
                # 'I' on both → no gate inserted (identity)
            else:
                new_qc.append(gate, qubit_ids)

        if not cz_replaced:
            raise ValueError(
                f"No CZ gate found on qubits {cut_qubit_pair}. "
                "Ensure the circuit has exactly one CZ at the cut position."
            )

        variants.append((new_qc, weight))

    return variants


def reconstruct_expectation(variant_results: list,
                             weights: list) -> float:
    """
    QPD reconstruction: ⟨O⟩ = Σᵢ ωᵢ · ⟨O⟩ᵢ

    Parameters
    ----------
    variant_results : list of float
    weights         : list of float  (same order, ωᵢ ∈ {+0.5, −0.5})

    Returns
    -------
    float
    """
    return float(sum(w * r for w, r in zip(weights, variant_results)))
