#!/usr/bin/env python3
"""
generate_counts_dataset.py
- Saves 1000 circuits into a single .qpy file.
- Saves CSV with circuit_id, qubit, depth, ideal_counts, noisy_counts.
"""

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

# Optional NoiseModel import
try:
    from qiskit_aer.noise import NoiseModel
except Exception:
    NoiseModel = None

# ---------------- Configuration ----------------
N_QUBITS = 3              # Number of Qubits
DEPTH = 3                 # Depth of random circuit
TOTAL_SAMPLES = 10000     # How many circuits to generate
SHOTS = 1024              # Number of shots for counts
METADATA_FN = 'test.csv'
ALL_CIRCUITS_QPY = 'test.qpy'

RNG = random.Random(42)

# ---------------- Utility functions ----------------
def json_dumps(obj):
    """Helper to save dictionary as JSON string in CSV"""
    return json.dumps(obj, separators=(',', ':'), sort_keys=True)

def generate_complex_circuit(n_qubits, depth):
    """Generates a random circuit similar to your previous structure"""
    qc = random_circuit(n_qubits, depth, measure=False, seed=RNG.randint(0, 1e6))
    
    # Add some random unitary blocks for complexity
    for _ in range(2):
        qubits = RNG.sample(range(n_qubits), 2)
        u_mat = random_unitary(4).data
        qc.append(UnitaryGate(u_mat, label="DenseU"), qubits)
        
    return qc

def get_noise_model():
    try: 
        return NoiseModel.from_backend(FakeGuadalupeV2())
    except: 
        print("Warning: FakeGuadalupeV2 not found, using clean simulation.")
        return None

# ---------------- Main Flow ----------------
def main():
    # 1. Setup Simulator and Noise
    noise_model = get_noise_model()
    # We use the same simulator for both, just toggling the noise_model arg
    backend_sim = AerSimulator()
    
    # Get basis gates from the fake backend to ensure valid transpilation
    fake_backend = FakeGuadalupeV2()
    basis_gates = fake_backend.operation_names

    saved_data = []
    saved_circuits_list = []
    
    print(f"Starting generation of {TOTAL_SAMPLES} circuits...")
    
    for i in tqdm(range(TOTAL_SAMPLES), desc="Generating Data"):
        
        # 2. Generate Circuit
        qc = generate_complex_circuit(N_QUBITS, DEPTH)
        
        # 3. Add Measurements (Crucial for getting 'counts')
        qc.measure_all()
        
        # 4. Transpile
        # We transpile ONCE so both ideal and noisy run on the exact same structure
        t_qc = transpile(qc, basis_gates=basis_gates, optimization_level=1, seed_transpiler=RNG.randint(0, 10000))
        
        # Give it a name based on ID
        t_qc.name = f"circuit_{i}"
        
        # 5. Run Ideal Simulation (No Noise)
        job_ideal = backend_sim.run(t_qc, shots=SHOTS)
        ideal_counts = job_ideal.result().get_counts()
        
        # 6. Run Noisy Simulation (With Noise Model)
        if noise_model:
            job_noisy = backend_sim.run(t_qc, shots=SHOTS, noise_model=noise_model)
            noisy_counts = job_noisy.result().get_counts()
        else:
            # Fallback if noise model fails
            noisy_counts = ideal_counts

        # 7. Prepare CSV Data Row
        row = {
            'circuit_id': i,
            'qubit': N_QUBITS,
            'depth': t_qc.depth(),
            'ideal_fidelity_counts': json_dumps(ideal_counts),
            'noisy_fidelity_counts': json_dumps(noisy_counts)
        }
        
        saved_data.append(row)
        saved_circuits_list.append(t_qc)

    # ---------------- Saving Outputs ----------------
    
    # Save CSV
    df = pd.DataFrame(saved_data)
    df.to_csv(METADATA_FN, index=False)
    print(f"\nSaved CSV data to: {METADATA_FN}")
    
    # Save QPY
    print(f"Saving {len(saved_circuits_list)} circuits to {ALL_CIRCUITS_QPY}...")
    with open(ALL_CIRCUITS_QPY, 'wb') as fd:
        qpy.dump(saved_circuits_list, fd)
    print("Done!")

if __name__ == '__main__':
    main()