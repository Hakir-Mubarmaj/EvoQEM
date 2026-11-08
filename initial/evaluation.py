import random
import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo
from qiskit.compiler import transpile
from qiskit.result import Counts

# DEAP library for Genetic Algorithm
from deap import base, creator, tools

# --- Previous Functions (from Step 2) ---
def get_hellinger_fidelity(counts1: Counts, counts2: Counts) -> float:
    all_keys = set(counts1.keys()) | set(counts2.keys())
    shots1 = sum(counts1.values())
    shots2 = sum(counts2.values())
    if shots1 == 0 or shots2 == 0: return 0.0
    p1 = np.array([counts1.get(key, 0) for key in all_keys]) / shots1
    p2 = np.array([counts2.get(key, 0) for key in all_keys]) / shots2
    hellinger_distance = np.sqrt(np.sum((np.sqrt(p1) - np.sqrt(p2)) ** 2)) / np.sqrt(2)
    hellinger_fidelity = (1 - hellinger_distance ** 2)
    return hellinger_fidelity

# --- Part 1: Define the "Genes" (Error Mitigation Building Blocks) ---
# Each gene is a function that returns a QuantumCircuit object.

def gene_do_nothing(num_qubits: int) -> QuantumCircuit:
    """Gene: Does nothing. Represents the absence of an operation."""
    return QuantumCircuit(num_qubits)

def gene_add_extra_x(num_qubits: int) -> QuantumCircuit:
    """Gene: Applies an X gate to all qubits. A simple form of Pauli twirling."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.x(i)
    return qc

def gene_add_decoupling_sequence(num_qubits: int) -> QuantumCircuit:
    """Gene: Applies a Y-Y decoupling sequence. Tries to cancel out certain noise types."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.y(i)
    qc.barrier()
    for i in range(num_qubits):
        qc.y(i)
    return qc

# Create a GENE_POOL from which the algorithm can choose
GENE_POOL = [gene_do_nothing, gene_add_extra_x, gene_add_decoupling_sequence]
NUM_QUBITS = 3
GENOME_LENGTH = 4 # Our mitigation strategy will be a sequence of 4 "genes".

# --- Part 2: Helper function to apply a genome to a circuit ---
def apply_genome(genome: list, base_circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Takes a genome (list of gene indices) and applies the corresponding
    operations to the end of the base circuit.
    """
    mitigated_circuit = base_circuit.copy()
    
    for gene_index in genome:
        # Get the gene function from the pool
        gene_function = GENE_POOL[gene_index]
        # Create the circuit for that gene
        gene_circuit = gene_function(NUM_QUBITS)
        # Append the gene circuit to our main circuit
        mitigated_circuit.compose(gene_circuit, inplace=True)
        
    return mitigated_circuit

# --- Part 3: Setup DEAP Genetic Algorithm framework ---
print("--- Setting up DEAP Genetic Algorithm framework ---")

# Define the fitness strategy (maximize a single objective: fidelity)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Define the structure of an "Individual" (our genome)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# How to create a single gene: pick a random index from our GENE_POOL
toolbox.register("gene_index", random.randint, 0, len(GENE_POOL) - 1)

# How to create an individual (genome): repeat gene creation GENOME_LENGTH times
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene_index, n=GENOME_LENGTH)

# How to create a population: a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Setup the environment (needed for the fitness function) ---
benchmark_circuit = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
benchmark_circuit.h(0)
benchmark_circuit.cx(0, 1)
benchmark_circuit.cx(0, 2)
# We add a barrier before measurement to clearly separate the main logic from mitigation/measurement
benchmark_circuit.barrier() 
# Note: Measurement is now part of the final mitigated circuit, not the base.

ideal_simulator = AerSimulator()
# Important: The ideal circuit for comparison needs the measurement part
ideal_circuit_for_sim = benchmark_circuit.copy()
ideal_circuit_for_sim.measure(range(NUM_QUBITS), range(NUM_QUBITS))
ideal_counts = ideal_simulator.run(ideal_circuit_for_sim, shots=1024).result().get_counts()

backend = FakeVigo()
noisy_simulator = AerSimulator.from_backend(backend)

# --- Define the Evaluation (Fitness) Function for DEAP ---
def evaluate(individual: list) -> tuple:
    """
    Fitness function for DEAP. Applies the genome, runs the circuit,
    and returns the fidelity score as a tuple.
    """
    # 1. Apply the genome to the base circuit
    mitigated_circuit = apply_genome(individual, benchmark_circuit)
    # 2. Add the final measurement
    mitigated_circuit.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    
    # 3. Transpile and run on noisy simulator
    transpiled = transpile(mitigated_circuit, noisy_simulator)
    result = noisy_simulator.run(transpiled, shots=1024).result()
    noisy_counts = result.get_counts()
    
    # 4. Calculate fidelity
    fidelity = get_hellinger_fidelity(ideal_counts, noisy_counts)
    
    return (fidelity,) # DEAP expects a tuple

# Register all the necessary tools for the evolution
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint) # Crossover
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(GENE_POOL)-1, indpb=0.1) # Mutation
toolbox.register("select", tools.selTournament, tournsize=3) # Selection

print("DEAP framework is ready.")

# --- Let's test by creating one random individual ---
print("\n--- Testing the setup ---")
random_individual = toolbox.individual()
print(f"A random individual (genome): {random_individual}")

final_circuit = apply_genome(random_individual, benchmark_circuit)
final_circuit.measure(range(NUM_QUBITS), range(NUM_QUBITS))

print("\nThe circuit after applying this random genome:")
print(final_circuit)
from deap import algorithms, tools

def main():
    """The main function to run the genetic algorithm."""
    # --- GA Parameters ---
    POPULATION_SIZE = 40
    CROSSOVER_PROB = 0.6
    MUTATION_PROB = 0.3
    N_GENERATIONS = 20

    print("\n--- Starting the Evolution Process ---")
    print(f"Population Size: {POPULATION_SIZE}, Generations: {N_GENERATIONS}")
    print(f"Crossover Prob: {CROSSOVER_PROB}, Mutation Prob: {MUTATION_PROB}")
    
    # Create the initial population
    population = toolbox.population(n=POPULATION_SIZE)

    # Setup statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    # The eaSimple algorithm is a standard evolutionary loop
    final_population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=N_GENERATIONS,
        stats=stats,
        verbose=True  # This will print progress for each generation
    )

    # --- Analysis of the Results ---
    print("\n--- Evolution Finished ---")
    
    # Find the best individual from the final population
    best_individual = tools.selBest(final_population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"\nBest Genome Found: {best_individual}")
    print(f"Best Fitness Score: {best_fitness:.4f}")

    # --- Compare with Baseline ---
    # Create the unmitigated circuit with measurement for a fair comparison
    unmitigated_circuit = benchmark_circuit.copy()
    unmitigated_circuit.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    # Evaluate the baseline fitness again
    baseline_fitness = evaluate([]) # An empty genome is the baseline
    
    print(f"\nBaseline Fitness (No Mitigation): {baseline_fitness[0]:.4f}")
    print(f"Fitness after EvoQEM Mitigation: {best_fitness:.4f}")
    
    improvement = ((best_fitness - baseline_fitness[0]) / (1.0 - baseline_fitness[0])) * 100
    if improvement > 0:
        print(f"Error reduced by approximately: {improvement:.2f}%")

    # --- Show the best circuit discovered ---
    print("\nOptimal mitigation circuit discovered by the algorithm:")
    best_circuit = apply_genome(best_individual, benchmark_circuit)
    best_circuit.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    print(best_circuit)

# Run the main function
if __name__ == "__main__":
    main()