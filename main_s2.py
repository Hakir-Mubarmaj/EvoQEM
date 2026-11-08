import time
import matplotlib.pyplot as plt
from src.circuit_utils_6q import get_cut_subcircuits, ideal_reconstruction, get_true_ideal_value
from src.ml_model_s import LearnedCorrector
# আপনার হার্ডওয়্যার এক্সিকিউটর ইমপোর্ট করছি (আগে তৈরি করা main_hardware_pipeline থেকে)
# নোট: ক্লাসটি এখানে সরাসরি ডিফাইন করা হলো সুবিধার জন্য
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import transpile

class RealHardwareExecutor:
    def __init__(self, backend_name='ibm_fez'):
        print(f"   [Init] Connecting to IBM Quantum Service...")
        try:
            self.service = QiskitRuntimeService()
            self.backend = self.service.backend(backend_name)
            print(f"   [Success] Connected to: {self.backend.name}")
        except Exception as e:
            print(f"   [Error] Connection failed: {e}")
            raise e

    def run_inference(self, circuits, shots=8000):
        # Measure All
        circuits_with_meas = [qc.measure_all(inplace=False) for qc in circuits]
        
        # Transpile
        transpiled = transpile(circuits_with_meas, self.backend, optimization_level=3, layout_method='sabre')
        
        print(f"   [Hardware] Submitting Job to {self.backend.name}...")
        job_start = time.time()
        
        try:
            sampler = Sampler(mode=self.backend)
            job = sampler.run(transpiled, shots=shots)
            print(f"   [Job ID] {job.job_id()}")
            result = job.result()
            job_end = time.time()
            print(f"   [Hardware] Job Completed in {job_end - job_start:.2f}s (excluding queue)")
            
            # Process Results
            expvals = []
            for i, pub_result in enumerate(result):
                try:
                    # Try getting counts from 'meas' register
                    counts = pub_result.data.meas.get_counts()
                except AttributeError:
                    # Fallback
                    first_creg = list(pub_result.data.__dict__.keys())[0]
                    counts = getattr(pub_result.data, first_creg).get_counts()

                val = 0
                total = 0
                for state, count in counts.items():
                    cleaned_state = state.replace(" ", "")
                    ones = cleaned_state.count('1')
                    if ones % 2 == 0: val += count
                    else: val -= count
                    total += count
                
                expvals.append(val/total if total > 0 else 0.0)
            
            return expvals, (job_end - job_start)
            
        except Exception as e:
            print(f"   [Error] {e}")
            return [0.0] * len(circuits), 0

def main():
    print("=== STEP 2: Real Hardware Inference & Benchmarking ===\n")
    MODEL_DIR = "saved_models"
    
    # ---------------------------------------------------------
    # PART A: Classical Simulation (The "Heavy" Task)
    # ---------------------------------------------------------
    print(">>> PART A: Classical Verification (Calculating Ground Truth) <<<")
    print("Simulating full 12-qubit circuit on CPU...")
    
    classic_start = time.time()
    # এখানে আমরা সত্যিকারের আইডিয়াল ভ্যালু বের করছি
    ideal_target = get_true_ideal_value()
    classic_end = time.time()
    
    classic_time = classic_end - classic_start
    print(f" -> True Ideal Value: {ideal_target:.4f}")
    print(f" -> CPU Simulation Time: {classic_time:.4f} seconds")
    print("(Note: For 50+ qubits, this would take days/years!)\n")

    # ---------------------------------------------------------
    # PART B: Quantum Execution (The "Fast" Task)
    # ---------------------------------------------------------
    print(">>> PART B: Quantum Execution on ibm_fez <<<")
    
    try:
        executor = RealHardwareExecutor(backend_name='ibm_fez')
    except:
        return

    subcircuits = get_cut_subcircuits()
    
    # রান করা হচ্ছে
    print(f"Running {len(subcircuits)} fragments on Quantum Hardware...")
    real_noisy_vals, qpu_time = executor.run_inference(subcircuits, shots=8000)

    # ---------------------------------------------------------
    # PART C: ML Correction (Using Saved Models)
    # ---------------------------------------------------------
    print("\n>>> PART C: Applying Learned Error Mitigation <<<")
    
    corrected_vals = []
    
    for i, val in enumerate(real_noisy_vals):
        # লোড মডেল
        corrector = LearnedCorrector()
        filename = f"{MODEL_DIR}/model_fragment_{i}.pkl"
        
        if corrector.load_model(filename):
            c_val = corrector.predict([val])[0]
            corrected_vals.append(c_val)
            print(f"   Fragment {i+1}: Raw={val:.4f} -> Corrected={c_val:.4f}")
        else:
            corrected_vals.append(val)

    # ---------------------------------------------------------
    # PART D: Final Results & Comparison
    # ---------------------------------------------------------
    print("\n>>> Final Reconstruction & Comparison <<<")
    
    rec_noisy = ideal_reconstruction(real_noisy_vals)
    rec_mitigated = ideal_reconstruction(corrected_vals)
    
    print("="*50)
    print(f"{'Metric':<20} | {'Value':<15}")
    print("-" * 40)
    print(f"{'True Ideal':<20} | {ideal_target:.10f}")
    print(f"{'QPU Raw':<20} | {rec_noisy:.10f} (Error: {abs(rec_noisy-ideal_target):.10f})")
    print(f"{'QPU + ML (Yours)':<20} | {rec_mitigated:.10f} (Error: {abs(rec_mitigated-ideal_target):.10f})")
    print("-" * 40)
    print(f"{'CPU Time':<20} | {classic_time:.10f}s")
    print(f"{'QPU Execution Time':<20} | {qpu_time:.10f}s (Approx)")
    print("="*50)

    # Plot
    methods = ['Classical CPU', 'QPU Raw', 'QPU + ML']
    accuracies = [ideal_target, rec_noisy, rec_mitigated]
    
    plt.figure(figsize=(8, 6))
    plt.bar(methods, accuracies, color=['blue', 'red', 'green'], alpha=0.7)
    plt.title(f"Quantum Advantage Demo: Accuracy Comparison\nTarget: {ideal_target:.4f}")
    plt.ylabel("Expectation Value")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()