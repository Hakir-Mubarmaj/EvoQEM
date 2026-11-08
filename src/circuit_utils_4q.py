import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def get_target_circuit():
    """
    Creates a 4-qubit target circuit for demonstration.
    Structure: Single qubit rotations followed by CZ gates.
    """
    qc = QuantumCircuit(4)
    qc.rx(0.5, 0)
    qc.rx(0.6, 1)
    qc.rx(0.7, 2)
    qc.rx(0.8, 3)
    qc.cz(0, 1)
    qc.cz(2, 3)
    return qc

def get_cut_subcircuits():
    """
    Returns the two fragments manually for the demo.
    These are the independent components after cutting the circuit.
    """
    # Sub-circuit 1 (Qubits 0 and 1)
    qc1 = QuantumCircuit(2)
    qc1.rx(0.5, 0)
    qc1.rx(0.6, 1)
    qc1.cz(0, 1)
    
    # Sub-circuit 2 (Qubits 2 and 3 mapped to 0 and 1)
    qc2 = QuantumCircuit(2)
    qc2.rx(0.7, 0)
    qc2.rx(0.8, 1)
    qc2.cz(0, 1)
    
    return [qc1, qc2]

def get_true_ideal_value():
    """
    Calculates the exact theoretical expectation value of the full circuit.
    Assumes we are measuring <Z0 Z1 Z2 Z3>.
    """
    qc = get_target_circuit()
    psi = Statevector.from_instruction(qc)
    # Calculate expectation value of Z^4
    # Note: In Qiskit, operator order is reversed, but for all Z it doesn't matter.
    from qiskit.quantum_info import Operator, Pauli
    op = Pauli('ZZZZ')
    return psi.expectation_value(op).real

def ideal_reconstruction(sub_results):
    """
    Reconstructs the global expectation value from sub-circuit results.
    For independent fragments measuring <ZZ> each, the total <ZZZZ> is the product.
    
    Args:
        sub_results (list): List of expectation values from sub-circuits.
    
    Returns:
        float: Reconstructed global expectation value.
    """
    # Assuming the cut assumes a tensor product structure (multiplication of expectations)
    final_val = 1.0
    for val in sub_results:
        final_val *= val
    return final_val