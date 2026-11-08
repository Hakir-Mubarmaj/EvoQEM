import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeVigoV2 as FakeVigo
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile

# --- Part 1: Create our benchmark circuit ---
# We will create a 3-qubit GHZ (Greenberger–Horne–Zeilinger) state.
# It's a classic example of entanglement and is sensitive to noise.
print("🚀 1. Creating the benchmark GHZ circuit...")
benchmark_circuit = QuantumCircuit(3, 3)
benchmark_circuit.h(0)
benchmark_circuit.cx(0, 1)
benchmark_circuit.cx(0, 2)
benchmark_circuit.measure([0, 1, 2], [0, 1, 2])

print("\nOur benchmark circuit:")
print(benchmark_circuit)

# --- Part 2: Calculate the ideal (noise-free) result ---
# What would be the result of this circuit on a perfect quantum computer?
print("\n📊 2. Calculating the ideal (noise-free) result...")
ideal_simulator = AerSimulator()
job_ideal = ideal_simulator.run(benchmark_circuit, shots=1024, memory=True)
result_ideal = job_ideal.result()
ideal_counts = result_ideal.get_counts()

print("\nIdeal result:")
print(ideal_counts)
# Ideally, we should only get '000' and '111', in roughly equal numbers.


# --- Part 3: Create a simulator with a realistic noise model ---
# We will use the noise profile of an older IBM quantum computer called 'Vigo'.
# This makes our research much more realistic.
print("\n🛠️ 3. Loading realistic noise model (from FakeVigo)...")
backend = FakeVigo()
# We create the simulator from the backend properties
noisy_simulator = AerSimulator.from_backend(backend)

# --- Part 4: Run the circuit in the noisy environment ---
# Now, let's see what our results look like under the influence of realistic noise.
print("\n🔬 4. Running the noisy simulation...")

# **THIS IS THE FIX: Transpile the circuit for the noisy backend before running**
transpiled_circuit = transpile(benchmark_circuit, noisy_simulator)

# **Run the *transpiled* circuit, not the original one**
job_noisy = noisy_simulator.run(transpiled_circuit, shots=1024, memory=True)
result_noisy = job_noisy.result()
noisy_counts = result_noisy.get_counts()

print("\nNoisy result:")
print(noisy_counts)
# Here you will see many unexpected results besides '000' and '111'!

# --- Result Visualization ---
print("\n📈 Plotting histograms to compare results...")
legend = ['Ideal Counts', 'Noisy Counts']
hist = plot_histogram([ideal_counts, noisy_counts], legend=legend)
hist.show() # This command will either display or save the image.