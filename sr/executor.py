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
# executor.py
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from collections import Counter

def is_aer_simulator(backend_obj):
    try:
        return backend_obj.__class__.__name__ == "AerSimulator"
    except Exception:
        return False

def run_on_aer_simulator(backend_aer, qc, shots=1024):
    t_qc = transpile(qc, backend=backend_aer)
    job = backend_aer.run([t_qc], shots=shots)
    res = job.result()
    try:
        counts = res.get_counts(0)
        total = sum(counts.values()) if counts else 0
        if total > 0:
            probs = {k: v / total for k, v in counts.items()}
            return probs
        else:
            return counts
    except Exception:
        if hasattr(res, "quasi_dists") and len(res.quasi_dists) > 0:
            return res.quasi_dists[0]
        if hasattr(res, "values") and len(res.values) > 0:
            return {"values": list(res.values[0])}
        return {"raw_result": str(res)}

def run_vd_experiments(backend, vd_circuits, shots=1024):
    """Run VD circuits robustly, return list of expectation values for ancilla measurement (±1 scale).
    Ancilla assumed to be last measured bit in each circuit's classical register; adjust if your mapping differs."""
    transpiled = [transpile(qc, backend=backend, optimization_level=2) for qc in vd_circuits]
    expvals = []
    try:
        sampler = Sampler(backend)
        run_obj = sampler.run(transpiled, shots=shots)
        res = run_obj.result()
        for i in range(len(transpiled)):
            if hasattr(res, 'quasi_dists') and len(res.quasi_dists) > i:
                qd = res.quasi_dists[i]
                p0 = sum(prob for bstr, prob in qd.items() if bstr[-1] == '0')
                p1 = 1.0 - p0
                expvals.append(p0 - p1)
                continue
            if hasattr(res, 'values') and len(res.values) > i:
                v = res.values[i]
                if isinstance(v, dict):
                    total = sum(v.values())
                    p0 = sum(prob for bstr, prob in v.items() if str(bstr)[-1] == '0') / (total if total else 1)
                    p1 = 1.0 - p0
                    expvals.append(p0 - p1)
                    continue
        return expvals
    except Exception:
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        for i in range(len(transpiled)):
            counts = result.get_counts(i)
            total = sum(counts.values()) if counts else 1
            p0 = sum(cnt for bitstr, cnt in counts.items() if bitstr[-1] == '0') / total
            p1 = 1.0 - p0
            expvals.append(p0 - p1)
        return expvals
