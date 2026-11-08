import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

# Make imports robust: use local module filenames (no 'src.' prefix)
from src.circuit_utils_6q import get_cut_subcircuits, ideal_reconstruction, get_true_ideal_value
from src.executor import VirtualExecutor
from src.ml_model import LearnedCorrector
from src.calibration import generate_calibration_circuits

np.random.seed(42)


def main():
    print("=== Starting Learned-Error-Mitigation Full Reconstruction Pipeline ===\n")

    executor = VirtualExecutor(backend_name='fez')
    corrector = LearnedCorrector()

    # 1. Circuit Cutting
    subcircuits = get_cut_subcircuits()
    print(f"Original Circuit cut into {len(subcircuits)} fragments.")

    final_corrected_sub_expvals = []
    final_noisy_sub_expvals = []

    # 2. Process each fragment
    for i, sub_qc in enumerate(subcircuits):
        print(f"\n>>> Processing Fragment {i+1}/{len(subcircuits)} <<<")

        # A. Calibration (Training)
        print("   -> Generating calibration data...")
        train_circuits, train_ideals = generate_calibration_circuits([sub_qc], num_samples=50)

        # Run training circuits on Noisy Backend
        print("   -> Running calibration circuits on backend...")
        train_noisy_vals = executor.run_mitigation_experiments(train_circuits, shots=4000)

        # Train the ML Model
        print("   -> Training Error Correction Model...")
        corrector.train(train_noisy_vals, train_ideals)

        # B. Inference (Actual Experiment)
        print("   -> Running actual subcircuit...")
        real_noisy_val = executor.run_mitigation_experiments([sub_qc], shots=8000)[0]

        # C. Correction
        corrected_val = corrector.predict([real_noisy_val])[0]

        print(f"      Raw Noisy: {real_noisy_val:.4f} | Corrected: {corrected_val:.4f}")

        final_noisy_sub_expvals.append(real_noisy_val)
        final_corrected_sub_expvals.append(corrected_val)

    # 3. Reconstruction
    print("\n>>> Reconstructing Global Expectation Value <<<")

    reconstructed_noisy = ideal_reconstruction(final_noisy_sub_expvals)
    reconstructed_corrected = ideal_reconstruction(final_corrected_sub_expvals)

    # Get True Theoretical Value
    ideal_target = get_true_ideal_value()

    print("="*40)
    print(f"True Ideal Target:          {ideal_target:.10f}")
    print("-" * 30)
    print(f"Standard Noisy Result:      {reconstructed_noisy:.10f}")
    print(f"Error (Noisy):              {abs(reconstructed_noisy - ideal_target):.10f}")
    print("-" * 30)
    print(f"Learned-Mitigated Result:   {reconstructed_corrected:.10f}")
    print(f"Error (Mitigated):          {abs(reconstructed_corrected - ideal_target):.10f}")
    print("="*40)
    
    # suppose for each fragment we have: noisy_val_i, and corrector.predict_with_uncertainty gives samples_i (n_samples x 1)
    # Collect samples for each fragment:
    samples_list = []
    n_samples = 2000
    for noisy_val in final_noisy_sub_expvals:
        _, _, samples = corrector.predict_with_uncertainty([noisy_val], n_samples=n_samples)
        # samples shape (n_samples, n_fragments?) -> when passing one noisy_val expect (n_samples, 1)
        samples_list.append(samples.flatten())

    # combine into (n_samples, n_fragments)
    import numpy as np
    M = np.vstack(samples_list).T  # shape (n_samples, n_fragments)
    global_samples = M.prod(axis=1)   # product across fragments
    mean = global_samples.mean()
    std = global_samples.std()
    print(f"Reconstructed (mitigated) = {mean:.6f} ± {1.96*std:.6f} (95% CI)")


    # 4. Visualization (try/except so script can run in headless environments)
    try:
        print("\nGenerating Result Plot...")
        methods = ['Ideal', 'Unmitigated (Noisy)', 'Mitigated (Yours)']
        values = [ideal_target, reconstructed_noisy, reconstructed_corrected]
        colors = ['gray', 'red', 'green']

        plt.figure(figsize=(8, 6))
        bars = plt.bar(methods, values, color=colors, alpha=0.8)

        for bar in bars:
            yval = bar.get_height()
            offset = 0.02 if yval > 0 else -0.05
            plt.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.4f}", ha='center', fontweight='bold')

        plt.ylabel('Expectation Value <ZZZZZZ>')
        plt.title('Quantum Error Mitigation Results')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
    except Exception as e:
        print(f"Plotting skipped (headless environment?): {e}")


if __name__ == "__main__":
    main()
