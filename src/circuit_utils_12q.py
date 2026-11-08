import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

def get_target_circuit():
    """
    Creates a larger 12-qubit target circuit for scalability testing.
    Structure: 6 pairs of entangled qubits (0-1, 2-3, ..., 10-11).
    """
    qc = QuantumCircuit(12)
    
    # Apply rotations to all qubits with varying angles to create a unique state
    # Angles range from 0.3 to 1.4 radians to ensure non-trivial results
    for i in range(12):
        angle = 0.3 + (i * 0.1) 
        qc.rx(angle, i)
    
    # Apply Entangling Gates (CZ) between pairs
    # This structure is specifically designed to be cleanly cut into 2-qubit fragments
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    for q1, q2 in pairs:
        qc.cz(q1, q2)
    
    return qc

def get_cut_subcircuits():
    """
    Manually cuts the 12-qubit circuit into 6 independent fragments.
    Each fragment is a 2-qubit circuit containing the specific angles for that pair.
    """
    subcircuits = []
    
    # We have 6 blocks. Each block corresponds to a pair of qubits in the original circuit.
    # Pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for i in range(6):
        # Calculate original qubit indices
        q1_idx = 2 * i
        q2_idx = 2 * i + 1
        
        # Calculate the specific angles used for these qubits
        angle1 = 0.3 + (q1_idx * 0.1)
        angle2 = 0.3 + (q2_idx * 0.1)
        
        # Create the fragment (Always mapped to qubits 0 and 1 locally)
        qc_frag = QuantumCircuit(2)
        qc_frag.rx(angle1, 0)
        qc_frag.rx(angle2, 1)
        qc_frag.cz(0, 1)
        
        subcircuits.append(qc_frag)
    
    return subcircuits

def get_true_ideal_value():
    """
    Calculates the exact theoretical expectation value for the full 12-qubit circuit.
    Measuring operator: <Z0 Z1 ... Z11>
    """
    qc = get_target_circuit()
    psi = Statevector.from_instruction(qc)
    
    # Expectation of Z tensor product 12 times (Z^12)
    op = Pauli('Z' * 12)
    return psi.expectation_value(op).real

def ideal_reconstruction(sub_results):
    """
    Reconstructs the global expectation value from sub-circuit results.
    Since the cuts are independent (tensor product), we multiply the results.
    
    Args:
        sub_results (list): List of expectation values from the 6 fragments.
    """
    final_val = 1.0
    for val in sub_results:
        final_val *= val
    return final_val