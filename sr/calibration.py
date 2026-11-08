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
# calibration.py
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
import numpy as np

def _remove_measurement_ops(qc: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of qc without measure instructions (safe for statevector)."""
    new_qc = qc.copy()
    new_qc.data = [instr for instr in new_qc.data if getattr(instr[0], "name", "") != 'measure']
    return new_qc

def generate_calibration_circuits(base_subcircuits, num_samples=8):
    """
    Create small numeric variations of a fragment and compute ideal expectation
    for qubit-0 using statevector (no measurements). Returns (training_circuits, ideal_values).
    """
    training_circuits = []
    ideal_values = []
    # Choose first fragment as template if list provided
    if isinstance(base_subcircuits, (list, tuple)) and len(base_subcircuits) > 0:
        template_qc = base_subcircuits[0].copy()
    elif isinstance(base_subcircuits, QuantumCircuit):
        template_qc = base_subcircuits.copy()
    else:
        raise ValueError("base_subcircuits must be a QuantumCircuit or list of them")

    for _ in range(num_samples):
        new_qc = template_qc.copy()
        param_shifted = False

        # Safely perturb numeric single-qubit rotation gates (rx, ry, rz)
        # We reconstruct new_qc instructions by appending to a fresh circuit
        safe_qc = QuantumCircuit(new_qc.num_qubits, new_qc.num_clbits)
        for instr, qargs, cargs in new_qc.data:
            name = getattr(instr, 'name', '').lower()
            params = getattr(instr, 'params', [])
            # only perturb numeric single-qubit rotation gates
            if name in ('rx', 'ry', 'rz') and len(params) == 1 and isinstance(params[0], (int, float, np.floating, np.integer)):
                p = float(params[0])
                shift = float(np.random.uniform(-0.5, 0.5))
                if name == 'rx':
                    safe_qc.rx(p + shift, qargs[0].index)
                elif name == 'ry':
                    safe_qc.ry(p + shift, qargs[0].index)
                else:
                    safe_qc.rz(p + shift, qargs[0].index)
                param_shifted = True
            else:
                # append original instruction as-is
                try:
                    safe_qc.append(instr, [q.index for q in qargs], [c.index for c in cargs])
                except Exception:
                    # fallback: try to copy by name for common gates
                    if name == 'cx' and len(qargs) >= 2:
                        safe_qc.cx(qargs[0].index, qargs[1].index)
                    else:
                        # best-effort: skip unknown instruction
                        pass

        if not param_shifted:
            # add a tiny rotation on qubit-0 to create a variant
            safe_qc.rx(float(np.random.uniform(0.05, 0.2)), 0)

        # compute ideal expectation for qubit-0 using statevector (no measurements)
        sv_qc = _remove_measurement_ops(safe_qc)
        psi = Statevector.from_instruction(sv_qc)
        probs = psi.probabilities_dict()
        p0 = 0.0
        p1 = 0.0
        # Qiskit bitstring keys are big-endian; using rightmost char as qubit-0 mapping is common
        for bitstr, prob in probs.items():
            if len(bitstr) == 0:
                continue
            if bitstr[-1] == '0':
                p0 += prob
            else:
                p1 += prob
        ideal_values.append(p0 - p1)
        training_circuits.append(safe_qc)

    return training_circuits, ideal_values
