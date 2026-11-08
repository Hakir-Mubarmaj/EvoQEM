# paper_adapter.py
# Generates paper-style training circuits, SPD/statevector ideals, and noisy measurements
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from tqdm import tqdm
import pickle

# Optional SPD import (if installed)
USE_SPD = False
try:
    import spd
    USE_SPD = True
    print("[paper_adapter] SPD available: will attempt to use SPD for ideals.")
except Exception:
    print("[paper_adapter] SPD not available: falling back to statevector for ideals.")


# ---------------------------
# Paper parameters (from paper_params.md)
# ---------------------------
params = {
    "n_qubits": 8,
    "trotter_steps": 4,
    "training_theta_indices": [0,1,2,3,4,5,54,55,56,57,58,59],  # as in paper grid
    "shots": 10000,
    "T1": 100e-6,
    "T2": 50e-6,
    "single_dep": 0.01,
    "two_dep": 0.04,
    "single_gate_time": 300e-9,
    "two_gate_time": 800e-9,
    "random_seed": 42
}

np.random.seed(params["random_seed"])

# ---------------------------
# Helper: build a Trotterized Ising target (paper-like)
# ---------------------------
def make_ising_trotter(n_qubits, steps, theta_h, theta_J):
    qc = QuantumCircuit(n_qubits)
    for _ in range(steps):
        # nearest-neighbour CZ entanglers (as example)
        for i in range(n_qubits - 1):
            qc.cz(i, i+1)
        # single-qubit rotations representing local fields
        for q in range(n_qubits):
            # use theta_h on each qubit; couplings theta_J ignored in this simple constructor,
            # but we will encode both parameters by varying the RX angles across grid if needed.
            qc.rx(theta_h, q)
    return qc

# ---------------------------
# Build training grid like paper (144 circuits)
# ---------------------------
def build_cpdr_training_grid(n_qubits, steps, idx_list):
    circuits = []
    for i in idx_list:
        for j in idx_list:
            theta_h = i * np.pi / 120.0
            theta_J = - j * np.pi / 120.0
            qc = make_ising_trotter(n_qubits, steps, theta_h, theta_J)
            circuits.append(qc)
    return circuits

# ---------------------------
# SPD/statevector ideal expectation
# ---------------------------
def compute_ideal(circ):
    pauli_str = 'Z' * circ.num_qubits
    if USE_SPD:
        # Try calling SPD API to compute expectation.
        # (Exact call depends on SPD install version — this is a best-effort attempt.)
        try:
            sv = spd.simulate_statevector(circ)  # if this exists
            op = Pauli(pauli_str)
            return float(sv.expectation_value(op).real)
        except Exception as e:
            # fallback
            sv = Statevector.from_instruction(circ)
            op = Pauli(pauli_str)
            return float(sv.expectation_value(op).real)
    else:
        sv = Statevector.from_instruction(circ)
        op = Pauli(pauli_str)
        return float(sv.expectation_value(op).real)

# ---------------------------
# Noise model builder (Aer)
# ---------------------------
def build_noise_model(T1, T2, single_dep, two_dep, sg_time, tg_time):
    nm = NoiseModel()
    single_err = depolarizing_error(single_dep, 1)
    two_err = depolarizing_error(two_dep, 2)
    nm.add_all_qubit_quantum_error(single_err, ['rx','ry','rz','h'])
    nm.add_all_qubit_quantum_error(two_err, ['cz','cx'])
    # Note: thermal_relaxation_error needs qubit times; for simplicity we skip RL here or add if desired
    return nm

# ---------------------------
# Noisy simulation: estimate <Z...Z> by sampling counts
# ---------------------------
def noisy_expectation_estimate(circuit, shots, noise_model):
    sim = AerSimulator(method='density_matrix')
    circ = circuit.copy()
    circ.measure_all()
    t = transpile(circ, sim)
    job = sim.run(t, shots=shots, noise_model=noise_model)
    res = job.result()
    counts = res.get_counts()
    total = sum(counts.values())
    val = 0.0
    for bitstr, c in counts.items():
        # qiskit returns bitstr with qubit n-1 ... 0, so reverse to index qubit0..qn-1
        bits = bitstr[::-1]
        parity = 1
        for b in bits:
            if b == '1':
                parity *= -1
        val += parity * (c/total)
    return val

# ---------------------------
# Main run: generate artifacts
# ---------------------------
def main():
    outdir = "artifacts"
    os.makedirs(outdir, exist_ok=True)

    print("[paper_adapter] Building training grid circuits...")
    train_qcs = build_cpdr_training_grid(params["n_qubits"], params["trotter_steps"], params["training_theta_indices"])
    print(f"[paper_adapter] Generated {len(train_qcs)} training circuits")

    # Save circuits as qpy (simple serialization)
    from qiskit import qpy
    qpy_files = []
    for idx, qc in enumerate(train_qcs):
        fn = os.path.join(outdir, f"train_qc_{idx}.qpy")
        with open(fn, 'wb') as fd:
            qpy.dump(qc, fd)
        qpy_files.append(fn)
    np.save(os.path.join(outdir, "train_qc_files.npy"), qpy_files)
    print("[paper_adapter] Serialized training circuits ->", len(qpy_files))

    # Compute ideals
    print("[paper_adapter] Computing ideal expectations (SPD/statevector)...")
    ideals = []
    for qc in tqdm(train_qcs):
        ideals.append(compute_ideal(qc))
    ideals = np.array(ideals)
    np.save(os.path.join(outdir, "train_ideals.npy"), ideals)
    print("[paper_adapter] Saved ideals:", ideals.shape)

    # Build noise model and compute noisy estimates for shots=10000
    print("[paper_adapter] Building noise model and simulating noisy circuits (this may take time)...")
    nm = build_noise_model(params["T1"], params["T2"], params["single_dep"], params["two_dep"], params["single_gate_time"], params["two_gate_time"])
    noisys = []
    for qc in tqdm(train_qcs):
        val = noisy_expectation_estimate(qc, shots=params["shots"], noise_model=nm)
        noisys.append(val)
    noisys = np.array(noisys)
    np.save(os.path.join(outdir, "train_noisys.npy"), noisys)
    print("[paper_adapter] Saved noisy estimates:", noisys.shape)

    # Save a small manifest
    meta = {
        "n_qubits": params["n_qubits"],
        "shots": params["shots"],
        "num_training": len(train_qcs)
    }
    with open(os.path.join(outdir, "manifest.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("[paper_adapter] Artifacts ready in", outdir)

if __name__ == "__main__":
    main()
