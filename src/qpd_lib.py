import numpy as np
from qiskit import QuantumCircuit

def decompose_cz_gate(qc_original, qubit_pair):
    """
    Decomposes a specific CZ gate in a circuit into 4 local variants 
    using Quasi-Probability Decomposition (QPD).
    
    Mathematical Formula for CZ decomposition:
    CZ = 0.5*II + 0.5*ZI + 0.5*IZ - 0.5*ZZ
    
    Args:
        qc_original (QuantumCircuit): The target circuit with the CZ gate removed.
        qubit_pair (tuple): Indices of the two qubits (control, target) where CZ was supposed to be.
        
    Returns:
        list: A list of tuples [(circuit_variant, weight), ...]
    """
    q1, q2 = qubit_pair
    variants = []
    
    # We create 4 branches (variants) based on the formula
    # Each variant applies local gates instead of the entangling CZ
    
    # Variant 1: Identity on both (I tensor I) -> Weight +0.5
    qc1 = qc_original.copy()
    # No operation added (Identity)
    variants.append((qc1, 0.5))
    
    # Variant 2: Z on Q1, Identity on Q2 (Z tensor I) -> Weight +0.5
    qc2 = qc_original.copy()
    qc2.z(q1)
    variants.append((qc2, 0.5))
    
    # Variant 3: Identity on Q1, Z on Q2 (I tensor Z) -> Weight +0.5
    qc3 = qc_original.copy()
    qc3.z(q2)
    variants.append((qc3, 0.5))
    
    # Variant 4: Z on both (Z tensor Z) -> Weight -0.5
    qc4 = qc_original.copy()
    qc4.z(q1)
    qc4.z(q2)
    variants.append((qc4, -0.5))
    
    return variants

def reconstruct_qpd_result(results, weights):
    """
    Combines the results of the variants using their weights.
    
    Formula: Final_Val = sum(weight_i * result_i)
    """
    final_val = 0.0
    for r, w in zip(results, weights):
        final_val += r * w
    return final_val