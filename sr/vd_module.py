# Patched script — small excerpt from your draft for reference (truncated)\n# -------------------------------------------------------------\n# file: Calibration.py

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter

def generate_calibration_circuits(base_subcircuits, num_samples=8):
    """
    Smart Calibration: Modifies EXISTING angles instead of adding new gates.
    This preserves the circuit structure and prevents SWAP explosion.
    """
    training_circuits = []
    ideal_values = []
    
    target_qc = base_subcircuits[0] 
    
    print(f"Generating {num_samples} smart variants...")
    
    for _ in range(num_samples):
        # 1. সার্কিট কপি করি
        cal_qc = target_qc.copy()
        
        # 2. স্ট্রাকচার ঠিক রেখে প্যারামিটার চেঞ্জ করা
        # আমরা সার্কিটের সব ইনস্ট্রাকশন চেক করব
        # যদি সেটা রোটেশন গেট হয় (rx, ry, rz), আমরা একটু নয়েজ অ্যাড করব\n# -------------------------------------------------------------\n\n
# vd_module.py
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
import numpy as np

def build_vd_circuit_for_fragment(fragment_qc: QuantumCircuit, copies=2):
    """Construct a simple VD circuit that estimates purity-like quantity for the fragment.
    This is a conservative skeleton useful for prototyping; adapt to your fragment registers."""
    n = fragment_qc.num_qubits
    qregs = QuantumRegister(n * copies + 1, 'q')
    cregs = ClassicalRegister(1, 'c')  # ancilla measurement
    vd_qc = QuantumCircuit(qregs, cregs)

    anc_idx = 0
    for i in range(copies):
        base_offset = 1 + i * n
        for instr, qargs, cargs in fragment_qc.data:
            q_indices = [base_offset + q.index for q in qargs]
            try:
                vd_qc.append(instr, q_indices, [])
            except Exception:
                name = getattr(instr, 'name', '').lower()
                if name == 'cx' and len(q_indices) >= 2:
                    vd_qc.cx(q_indices[0], q_indices[1])
                elif name in ('rx','ry','rz') and len(instr.params) == 1:
                    p = float(instr.params[0])
                    if name == 'rx':
                        vd_qc.rx(p, q_indices[0])
                    elif name == 'ry':
                        vd_qc.ry(p, q_indices[0])
                    else:
                        vd_qc.rz(p, q_indices[0])
                else:
                    pass

    # Note: For full VD you'll need swap-test style constructions; this is a template for prototyping.
    return vd_qc
