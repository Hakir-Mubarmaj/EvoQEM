#!/usr/bin/env python3
"""
final_dg.py (Modified for QPY Saving)
- Saves 1000 circuits into a single .qpy file alongside CSV.
"""

import json
import random
import warnings
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# Qiskit imports (v1.x compatible)
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import UnitaryGate, XGate, YGate
from qiskit.quantum_info import Statevector, random_unitary
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import PadDynamicalDecoupling

# Optional NoiseModel
try:
    from qiskit_aer.noise import NoiseModel
except Exception:
    NoiseModel = None

# ---------------- Configuration ----------------
N_QUBITS = 3
PRIMARY_DEPTH = 3     
TARGET_SAMPLES = 1000   
IMPROVEMENT_THRESHOLD = 0.01 
GENOMES_PER_CIRCUIT = 60 
GENOME_LENGTH = 5       
BASIS_GATES = ['u', 'cx']
METADATA_FN = 'metadata_robust_final.csv'
ALL_CIRCUITS_QPY = 'dataset_circuits.qpy' # ### NEW ADDITION: QPY file name

RNG = random.Random(42)

# ---------------- Utility functions ----------------
def get_gate_counts(qc: QuantumCircuit) -> dict:
    counts = Counter(item.operation.name for item in qc.data)
    return dict(counts)

def json_dumps(obj):
    return json.dumps(obj, separators=(',', ':'), sort_keys=True)

def safe_transpile(qc: QuantumCircuit, optimization_level=1):
    try:
        return transpile(qc, basis_gates=BASIS_GATES, optimization_level=optimization_level, seed_transpiler=RNG.randint(0, 100000))
    except Exception:
        return transpile(qc, basis_gates=BASIS_GATES, optimization_level=0, seed_transpiler=RNG.randint(0, 100000))

# ---------------- Gene Functions ----------------
PAULIS = ['i', 'x', 'y', 'z']

def do_nothing_gene(qc: QuantumCircuit) -> QuantumCircuit:
    return qc

def pauli_twirl_gene(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for item in qc.data:
        instr = item.operation
        qargs = item.qubits
        cargs = item.clbits
        if instr.name == 'cx':
            paulis = [RNG.choice(PAULIS) for _ in qargs]
            for q, p in zip(qargs, paulis):
                if p=='x': new_qc.x(q)
                elif p=='y': new_qc.y(q)
                elif p=='z': new_qc.z(q)
            new_qc.append(instr, qargs, cargs)
            for q, p in zip(qargs, paulis):
                if p=='x': new_qc.x(q)
                elif p=='y': new_qc.y(q)
                elif p=='z': new_qc.z(q)
        else:
            new_qc.append(instr, qargs, cargs)
    return new_qc

def xy4_dd_gene(qc: QuantumCircuit) -> QuantumCircuit:
    try:
        new_qc = qc.copy()
        new_qc.barrier()
        for q in new_qc.qubits:
            new_qc.x(q); new_qc.y(q); new_qc.x(q); new_qc.y(q)
        return new_qc
    except: return qc

def spin_echo_gene(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    half = len(qc.data) // 2
    for i, item in enumerate(qc.data):
        new_qc.append(item.operation, item.qubits, item.clbits)
        if i == half:
            new_qc.barrier()
            for q in new_qc.qubits: new_qc.x(q)
            new_qc.barrier()
    for q in new_qc.qubits: new_qc.x(q)
    return new_qc

def composite_pulse_gene(qc: QuantumCircuit) -> QuantumCircuit:
    new_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for item in qc.data:
        instr = item.operation
        qargs = item.qubits
        cargs = item.clbits
        if instr.name in ['u', 'rx', 'ry', 'rz'] and RNG.random() < 0.2:
            q = qargs[0]
            try:
                new_qc.rz(0.1, q); new_qc.rx(1.57, q); new_qc.rz(0.1, q); new_qc.rx(-1.57, q)
            except: new_qc.append(instr, qargs, cargs)
        else:
            new_qc.append(instr, qargs, cargs)
    return new_qc

GENE_MAP = {
    'DoNothing': do_nothing_gene,
    'PauliTwirl': pauli_twirl_gene,
    'XY4': xy4_dd_gene,
    'SpinEcho': spin_echo_gene,
    'CompositePulse': composite_pulse_gene
}

# ---------------- Main Flow ----------------
def generate_complex_circuit():
    qc = random_circuit(N_QUBITS, PRIMARY_DEPTH, measure=False, seed=RNG.randint(0, 1e6))
    for _ in range(3):
        qubits = RNG.sample(range(N_QUBITS), 2)
        u_mat = random_unitary(4).data
        qc.append(UnitaryGate(u_mat, label="DenseU"), qubits)
    return qc

def get_noise_model():
    try: return NoiseModel.from_backend(FakeGuadalupeV2())
    except: return None

def compute_fidelity(qc, noise_model, simulator):
    try:
        qc_ideal = qc.copy()
        qc_ideal.remove_final_measurements()
        sv = Statevector.from_instruction(qc_ideal)
        
        qc_noisy = qc.copy()
        qc_noisy.save_density_matrix()
        job = simulator.run(qc_noisy, noise_model=noise_model)
        rho = job.result().data()['density_matrix']
        return float(np.real(sv.expectation_value(rho)))
    except: return 0.0

def main():
    noise_model = get_noise_model()
    sim = AerSimulator(method='density_matrix')
    saved_data = []
    
    # ### NEW ADDITION: List to hold all passing circuits
    saved_circuits_list = [] 
    
    pbar = tqdm(total=TARGET_SAMPLES, desc="Generating Robust Data")
    
    attempts = 0
    gene_keys = list(GENE_MAP.keys())
    
    while len(saved_data) < TARGET_SAMPLES:
        attempts += 1
        
        # 1. Generate & Transpile
        complex_qc = generate_complex_circuit()
        transpiled_qc = safe_transpile(complex_qc, optimization_level=1)
        
        # 2. Baseline Fidelity
        fid_baseline = compute_fidelity(transpiled_qc, noise_model, sim)
        
        if fid_baseline > 0.98 or fid_baseline < 0.2:
            continue
            
        # 3. Test Genomes
        best_fid = -1.0
        best_genome_seq = None
        
        for _ in range(GENOMES_PER_CIRCUIT):
            genome = RNG.choices(gene_keys, k=GENOME_LENGTH)
            
            qc_candidate = transpiled_qc.copy()
            for g in genome:
                qc_candidate = GENE_MAP[g](qc_candidate)
            
            qc_candidate_t = safe_transpile(qc_candidate, optimization_level=0)
            fid_g = compute_fidelity(qc_candidate_t, noise_model, sim)
            
            if fid_g > best_fid:
                best_fid = fid_g
                best_genome_seq = genome
        
        # 4. Strict Filter
        if best_fid > (fid_baseline + IMPROVEMENT_THRESHOLD):
            init_counts = get_gate_counts(complex_qc)
            final_counts = get_gate_counts(transpiled_qc)
            
            row = {
                'circuit_id': len(saved_data), # Added ID for easier matching
                'n_qubits': N_QUBITS,
                'initial_depth': PRIMARY_DEPTH,
                'initial_gate_counts': json_dumps(init_counts),
                'final_depth': transpiled_qc.depth(),
                'final_gate_counts': json_dumps(final_counts),
                'noisy_fidelity_before': fid_baseline,
                'best_genome': json_dumps(best_genome_seq),
                'fidelity_after_genome': best_fid,
                'improvement': best_fid - fid_baseline
            }
            saved_data.append(row)
            
            # ### NEW ADDITION: Save the successful circuit to list
            # We give it a name matching its ID in CSV
            transpiled_qc.name = f"circuit_{len(saved_data)-1}"
            saved_circuits_list.append(transpiled_qc)

            pbar.update(1)
            pbar.set_postfix({
                'RejectionRate': f"{100*(1 - len(saved_data)/attempts):.1f}%",
                'LastGain': f"+{row['improvement']:.3f}"
            })
    
    # Save CSV
    df = pd.DataFrame(saved_data)
    df.to_csv(METADATA_FN, index=False)
    print(f"\nSaved {len(df)} robust samples to {METADATA_FN}")
    
    # ### NEW ADDITION: Dump all circuits to QPY
    print(f"Saving {len(saved_circuits_list)} circuits to {ALL_CIRCUITS_QPY}...")
    with open(ALL_CIRCUITS_QPY, 'wb') as fd:
        qpy.dump(saved_circuits_list, fd)
    print("Done!")
    
    print(f"Total Circuits Simulated: {attempts}")

if __name__ == '__main__':
    main()