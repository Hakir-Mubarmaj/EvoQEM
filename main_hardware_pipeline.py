import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

# Qiskit Runtime Imports for Real Hardware
# Updated to include SamplerV2 for primitives-based execution
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile

# Import existing modules (Must be in the same directory)
from src.circuit_utils_6q import get_cut_subcircuits, ideal_reconstruction, get_true_ideal_value
from src.ml_model import LearnedCorrector
from src.calibration import generate_calibration_circuits

# Filter warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class RealHardwareExecutor:
    """
    Executor specifically designed for IBM Quantum Hardware (e.g., ibm_fez).
    Updated to use Qiskit Runtime Primitives (SamplerV2).
    """
    def __init__(self, backend_name='ibm_fez'):
        print(f"   [Init] Connecting to IBM Quantum Service...")
        try:
            # Loads account from saved credentials (VS Code / .qiskit)
            self.service = QiskitRuntimeService()
            self.backend = self.service.backend(backend_name)
            print(f"   [Success] Connected to Real Backend: {self.backend.name}")
            
            try:
                print(f"   [Info] Qubits: {self.backend.num_qubits}")
            except:
                print("   [Info] Backend connected (Status details unavailable)")
                
        except Exception as e:
            print(f"   [Error] Failed to connect to backend '{backend_name}': {e}")
            raise e

    def run_mitigation_experiments(self, circuits, shots=4000):
        """
        Runs circuits on real hardware using SamplerV2.
        """
        # 1. Add Measurements explicitly
        # measure_all creates a classical register named 'meas' by default
        circuits_with_meas = []
        for qc in circuits:
            qc_meas = qc.measure_all(inplace=False)
            circuits_with_meas.append(qc_meas)

        print(f"   [Hardware] Transpiling {len(circuits)} circuits for {self.backend.name}...")
        
        # 2. Transpile with high optimization for hardware
        # Even with Sampler, passing ISA (Instruction Set Architecture) circuits is recommended
        transpiled_circuits = transpile(
            circuits_with_meas, 
            self.backend, 
            optimization_level=3, 
            layout_method='sabre'
        )
        
        print(f"   [Hardware] Submitting Job (Shots={shots}) via SamplerV2... Waiting in Queue...")
        
        # 3. Run Job using SamplerV2
        try:
            # Initialize Sampler with the specific backend
            sampler = Sampler(mode=self.backend)
            
            # Run the job
            # SamplerV2 takes a list of circuits (Primitive Unified Blocs - PUBs)
            job = sampler.run(transpiled_circuits, shots=shots)
            print(f"   [Job ID] {job.job_id()}")
            
            # Wait for the job to complete
            result = job.result()
            print("   [Hardware] Job Completed!")
            
            # 4. Process Results (SamplerV2 returns a list of PubResults)
            expvals = []
            
            # Iterate through each circuit's result
            for i, pub_result in enumerate(result):
                try:
                    # Access counts from the 'meas' register created by measure_all()
                    # SamplerV2 structure: result[i].data.<register_name>.get_counts()
                    counts = pub_result.data.meas.get_counts()
                except AttributeError:
                    print(f"   [Warning] Could not find 'meas' register in result {i}. Checking structure...")
                    # Fallback: try to grab the first available register if 'meas' fails
                    try:
                        first_creg = list(pub_result.data.__dict__.keys())[0]
                        counts = getattr(pub_result.data, first_creg).get_counts()
                    except:
                        print(f"Error getting counts for circuit {i}")
                        expvals.append(0.0)
                        continue

                val = 0
                total_counts = 0
                
                for state, count in counts.items():
                    # Clean state string (SamplerV2 usually returns bitstrings like '101')
                    cleaned_state = state.replace(" ", "")
                    
                    # Calculate parity (number of 1s)
                    ones = cleaned_state.count('1')
                    
                    if ones % 2 == 0:
                        val += count  # Even parity -> +1
                    else:
                        val -= count  # Odd parity -> -1
                    total_counts += count
                
                if total_counts > 0:
                    expvals.append(val / total_counts)
                else:
                    expvals.append(0.0)
            
            return expvals
            
        except Exception as e:
            print(f"   [Error] Job execution failed: {e}")
            # Return zeros if job fails to prevent pipeline crash
            return [0.0] * len(circuits)

def main():
    print("=== Starting Real Hardware (ibm_fez) Validation Pipeline ===\n")
    print("Note: This process involves real queue times. Please be patient.\n")
    
    try:
        # Initialize Real Hardware Executor
        executor = RealHardwareExecutor(backend_name='ibm_fez')
    except Exception:
        print("Exiting pipeline due to connection error.")
        return
    
    corrector = LearnedCorrector()
    
    # 1. Circuit Cutting
    subcircuits = get_cut_subcircuits()
    print(f"Original Circuit cut into {len(subcircuits)} fragments.")
    
    final_corrected_sub_expvals = []
    final_noisy_sub_expvals = []
    
    # 2. Process each fragment
    for i, sub_qc in enumerate(subcircuits):
        print(f"\n>>> Processing Fragment {i+1}/{len(subcircuits)} on REAL HARDWARE <<<")
        
        # A. Calibration (Training)
        print("   -> Generating calibration data (Smart Perturbation)...")
        # Generate training circuits
        train_circuits, train_ideals = generate_calibration_circuits([sub_qc], num_samples=25)
        
        print("   -> Running calibration circuits on ibm_fez...")
        train_noisy_vals = executor.run_mitigation_experiments(train_circuits, shots=4000)
        
        # Train Model
        print("   -> Training Error Correction Model...")
        corrector.train(train_noisy_vals, train_ideals)
        
        # B. Inference (Actual Experiment)
        print("   -> Running actual subcircuit on ibm_fez...")
        real_noisy_val_list = executor.run_mitigation_experiments([sub_qc], shots=8000)
        real_noisy_val = real_noisy_val_list[0]
        
        # C. Correction
        corrected_val = corrector.predict([real_noisy_val])[0]
        
        print(f"      [Fragment {i+1}] Raw Noisy: {real_noisy_val:.10f} | Corrected: {corrected_val:.10f}")
        
        final_noisy_sub_expvals.append(real_noisy_val)
        final_corrected_sub_expvals.append(corrected_val)

    # 3. Reconstruction
    print("\n>>> Reconstructing Global Expectation Value <<<")
    
    reconstructed_noisy = ideal_reconstruction(final_noisy_sub_expvals)
    reconstructed_corrected = ideal_reconstruction(final_corrected_sub_expvals)
    ideal_target = get_true_ideal_value()
    
    print("="*40)
    print(f"True Ideal Target:          {ideal_target:.10f}")
    print("-" * 30)
    print(f"Hardware (Noisy) Result:    {reconstructed_noisy:.10f}")
    print(f"Error (Noisy):              {abs(reconstructed_noisy - ideal_target):.10f}")
    print("-" * 30)
    print(f"Hardware (Mitigated) Result:{reconstructed_corrected:.10f}")
    print(f"Error (Mitigated):          {abs(reconstructed_corrected - ideal_target):.10f}")
    print("="*40)
    
    # 4. Visualization
    print("\nGenerating Result Plot...")
    
    methods = ['Ideal', 'Real Hardware (Noisy)', 'Mitigated (Yours)']
    values = [ideal_target, reconstructed_noisy, reconstructed_corrected]
    colors = ['gray', 'orange', 'green']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, values, color=colors, alpha=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02 if yval > 0 else yval - 0.05, 
                 f"{yval:.4f}", ha='center', fontweight='bold')
    
    plt.ylabel('Expectation Value <ZZZZ>')
    plt.title('Real Hardware (ibm_fez) Error Mitigation Results')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()