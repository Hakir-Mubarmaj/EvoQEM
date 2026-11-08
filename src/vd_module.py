from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# NOTE: This module is deprecated for the main pipeline.
# The original implementation calculated Purity (Tr(rho^2)) via Swap Test,
# which cannot be directly mapped to Expectation Values (<Z>) for error mitigation.
# This file is kept for archival purposes or future purity studies.

def apply_vd_to_circuit(original_qc):
    """
    Takes a circuit (N qubits) and creates a VD circuit (2N + 1 qubits).
    Returns a circuit measuring the ancilla for Purity estimation.
    """
    n_qubits = original_qc.num_qubits
    
    # Registers: Ancilla (1), Copy A (N), Copy B (N)
    qr_ancilla = QuantumRegister(1, 'ancilla')
    qr_a = QuantumRegister(n_qubits, 'sys_a')
    qr_b = QuantumRegister(n_qubits, 'sys_b')
    cr = ClassicalRegister(1, 'c')
    
    vd_qc = QuantumCircuit(qr_ancilla, qr_a, qr_b, cr)
    
    # 1. Initialize Ancilla
    vd_qc.h(qr_ancilla) 
    
    # 2. Append original circuit to both copies
    # Copy A
    vd_qc.compose(original_qc, qubits=qr_a, inplace=True)
    # Copy B
    vd_qc.compose(original_qc, qubits=qr_b, inplace=True)
    
    # 3. Controlled-SWAP (Fredkin Gate)
    for i in range(n_qubits):
        vd_qc.cswap(qr_ancilla[0], qr_a[i], qr_b[i])
        
    # 4. Measure Ancilla (Basis change H needed for Real part)
    vd_qc.h(qr_ancilla)
    vd_qc.measure(qr_ancilla, cr)
    
    return vd_qc