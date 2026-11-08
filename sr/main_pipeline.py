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
# main_pipeline.py
from calibration import generate_calibration_circuits
from executor import run_vd_experiments
from ml_model import LearnedCorrector
from vd_module import build_vd_circuit_for_fragment

def prototype_flow(fake_backend, real_backend=None):
    frag = None
    # Minimal fallback fragment if none provided
    from qiskit import QuantumCircuit
    frag = QuantumCircuit(3, 3)
    for q in range(3):
        frag.ry(0.3, q)
    frag.cx(0,1); frag.cx(1,2)
    frag.measure(range(3), range(3))

    cal_circs, ideal_vals = generate_calibration_circuits(frag, num_samples=8)

    vd_circs = [build_vd_circuit_for_fragment(c, copies=2) for c in cal_circs]
    vd_results = run_vd_experiments(fake_backend, vd_circs, shots=1024)

    noisy_features = [[v] for v in vd_results]
    gp = LearnedCorrector()
    gp.train(noisy_features, ideal_vals)

    if real_backend is not None and len(vd_results) > 0:
        test_vd = build_vd_circuit_for_fragment(frag, copies=2)
        test_res = run_vd_experiments(real_backend, [test_vd], shots=1024)
        if len(test_res) > 0:
            corrected = gp.correct([test_res[0]])
            print("Real raw:", test_res[0], "corrected:", corrected)

if __name__ == '__main__':
    print("Prototype orchestration created. Import and call prototype_flow with proper backend objects.")
