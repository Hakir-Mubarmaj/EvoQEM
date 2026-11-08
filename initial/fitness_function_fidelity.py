import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo
from qiskit.compiler import transpile
from qiskit.result import Counts

def get_hellinger_fidelity(counts1: Counts, counts2: Counts) -> float:
    """Calculates the Hellinger fidelity between two Qiskit counts dictionaries."""
    
    # Get the set of all possible outcomes (keys) from both distributions
    all_keys = set(counts1.keys()) | set(counts2.keys())
    
    # Get total shots for normalization
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    
    # Create probability vectors, ensuring all keys are present in both
    p1 = np.array([counts1.get(key, 0) for key in all_keys]) / shots1
    p2 = np.array([counts2.get(key, 0) for key in all_keys]) / shots2
    
    # Calculate Hellinger distance
    hellinger_distance = np.sqrt(np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)) / np.sqrt(2)
    
    # Calculate and return Hellinger fidelity
    hellinger_fidelity = (1 - hellinger_distance ** 2)
    
    return hellinger_fidelity

def fitness_function(circuit: QuantumCircuit, ideal_counts: Counts, noisy_simulator: AerSimulator) -> float:
    """
    Executes a circuit on a noisy simulator and returns its fitness score
    based on Hellinger fidelity with the ideal results.
    """
    # Transpile the circuit for the noisy backend
    transpiled_circuit = transpile(circuit, noisy_simulator)
    
    # Run the simulation
    job = noisy_simulator.run(transpiled_circuit, shots=1024)
    result = job.result()
    noisy_counts = result.get_counts()
    
    # Calculate fidelity between the noisy result and the ideal result
    fidelity = get_hellinger_fidelity(ideal_counts, noisy_counts)
    
    return fidelity

# Main execution block
if __name__ == "__main__":
    # --- Step 1: Setup the environment (same as before) ---
    print("--- Setting up the environment ---")
    
    # Create benchmark circuit
    benchmark_circuit = QuantumCircuit(3, 3)
    benchmark_circuit.h(0)
    benchmark_circuit.cx(0, 1)
    benchmark_circuit.cx(0, 2)
    benchmark_circuit.measure([0, 1, 2], [0, 1, 2])
    
    # Get ideal counts
    ideal_simulator = AerSimulator()
    ideal_counts = ideal_simulator.run(benchmark_circuit, shots=1024).result().get_counts()
    print(f"Ideal Counts: {ideal_counts}")
    
    # Setup noisy simulator
    backend = FakeVigo()
    noisy_simulator = AerSimulator.from_backend(backend)
    print("Environment setup complete.\n")
    
    # --- Step 2: Calculate the baseline fitness of the unmitigated circuit ---
    print("--- Calculating baseline fitness ---")
    
    # Calculate the fitness of our original, unmitigated circuit
    baseline_fitness = fitness_function(benchmark_circuit, ideal_counts, noisy_simulator)
    
    print(f"\nThe Hellinger Fidelity (Fitness Score) of the original noisy circuit is: {baseline_fitness:.4f}")
    print("Our goal is to evolve a mitigation strategy that makes this score closer to 1.0")