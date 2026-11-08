import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

def get_target_circuit():
    """
    Creates a MORE COMPLEX 6-qubit target circuit.
    Structure: 
    - Layer 1: Random rotations (RX, RY, RZ) to create superposition.
    - Layer 2: Strong Entanglement (CNOT + CRX) within pairs.
    - Layer 3: Additional rotations to make it non-Clifford.
    
    Pairs: (0,1), (2,3), (4,5) [Still separable for validation, but harder to mitigate]
    """
    qc = QuantumCircuit(6)
    
    # --- BLOCK 1 (Q0, Q1) ---
    # Initial Superposition with non-trivial phases
    qc.rx(0.3, 0)
    qc.ry(0.4, 1)
    
    # Entanglement Layer (Multiple gates increase depth and noise)
    qc.cz(0, 1)          # Phase entangler
    qc.rx(0.5, 0)        # Mid-circuit rotation
    qc.rx(0.6, 1)
    qc.cz(0, 1)          # Second entangler (increases complexity)
    
    # Final Rotation
    qc.rz(0.2, 0)
    qc.rz(0.3, 1)

    # --- BLOCK 2 (Q2, Q3) ---
    qc.rx(0.7, 2)
    qc.ry(0.8, 3)
    
    qc.cz(2, 3)
    qc.rx(0.9, 2)
    qc.rx(1.0, 3)
    qc.cz(2, 3)
    
    qc.rz(0.4, 2)
    qc.rz(0.5, 3)

    # --- BLOCK 3 (Q4, Q5) ---
    qc.rx(1.1, 4)
    qc.ry(1.2, 5)
    
    qc.cz(4, 5)
    qc.rx(1.3, 4)
    qc.rx(1.4, 5)
    qc.cz(4, 5)
    
    qc.rz(0.6, 4)
    qc.rz(0.7, 5)
    
    return qc

def get_cut_subcircuits():
    """
    Manually cuts the complex 6-qubit circuit into 3 independent fragments.
    Each fragment mimics the exact operations of the blocks above.
    """
    subcircuits = []
    
    # Parameters matching the blocks above
    # Params format: [rx_0, ry_1, mid_rx_0, mid_rx_1, final_rz_0, final_rz_1]
    params_list = [
        [0.3, 0.4, 0.5, 0.6, 0.2, 0.3], # Fragment 1
        [0.7, 0.8, 0.9, 1.0, 0.4, 0.5], # Fragment 2
        [1.1, 1.2, 1.3, 1.4, 0.6, 0.7]  # Fragment 3
    ]
    
    for params in params_list:
        qc_frag = QuantumCircuit(2)
        
        # Reconstruct the exact gate sequence
        qc_frag.rx(params[0], 0)
        qc_frag.ry(params[1], 1)
        
        qc_frag.cz(0, 1)
        qc_frag.rx(params[2], 0)
        qc_frag.rx(params[3], 1)
        qc_frag.cz(0, 1)
        
        qc_frag.rz(params[4], 0)
        qc_frag.rz(params[5], 1)
        
        subcircuits.append(qc_frag)
    
    return subcircuits

def get_true_ideal_value():
    """
    Calculates the exact theoretical expectation value for the full circuit.
    Measuring operator: <Z0 Z1 Z2 Z3 Z4 Z5>
    """
    qc = get_target_circuit()
    psi = Statevector.from_instruction(qc)
    
    op = Pauli('ZZZZZZ')
    return psi.expectation_value(op).real

def ideal_reconstruction(sub_results):
    """
    Reconstructs the global expectation value from sub-circuit results.
    Math: <Global> = <Frag1> * <Frag2> * <Frag3>
    """
    return np.prod(sub_results)