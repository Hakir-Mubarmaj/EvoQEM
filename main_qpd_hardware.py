import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

# Import your specific modules
from src.calibration import generate_calibration_circuits
from src.executor import VirtualExecutor, RealHardwareExecutor
from src.qpd_lib import decompose_cz_gate, reconstruct_qpd_result

def get_entangled_circuit():
    """
    Creates a simple entangled circuit (RX-RX-CZ).
    This will be our target for QPD.
    """
    qc = QuantumCircuit(2)
    qc.rx(0.6, 0)
    qc.rx(0.8, 1)
    # CZ gate is implicit, we will cut it using QPD
    return qc

def get_true_ideal_value_with_cz():
    """
    Calculates the True Ideal value (mathematically) for validation.
    """
    qc = QuantumCircuit(2)
    qc.rx(0.6, 0)
    qc.rx(0.8, 1)
    qc.cz(0, 1)
    psi = Statevector.from_instruction(qc)
    op = Pauli('ZZ')
    return psi.expectation_value(op).real

def train_linear_model(noisy_data, ideal_data):
    """
    Implements the 'Perturbation Data Regression' (PDR) logic.
    Instead of complex GPR, we fit a robust Linear Model (y = mx + c).
    
    Why: Locally around the target parameters, the noise map is often linear.
    This prevents overfitting that complex models might suffer from.
    """
    # Fit a 1st degree polynomial (Linear Regression)
    # slope (m), intercept (c) = y = mx + c
    slope, intercept = np.polyfit(noisy_data, ideal_data, 1)
    return slope, intercept

def main():
    print("=== Starting PDR-QPD (Perturbation Data Regression) Pipeline ===\n")
    print("Methodology: QPD Cutting + Local Linear Regression on Perturbed Data\n")
    
    # Initialize Executor
    #executor = VirtualExecutor(backend_name='fez')
    executor = RealHardwareExecutor(backend_name='ibm_fez')
    
    # 1. Prepare Target
    base_qc = get_entangled_circuit()
    print("-> Circuit Prepared. Applying QPD decomposition...")
    
    # 2. QPD Decomposition (from qpd_lib.py)
    qpd_data = decompose_cz_gate(base_qc, qubit_pair=(0, 1))
    variants = [item[0] for item in qpd_data]
    weights = [item[1] for item in qpd_data]
    print(f"-> Generated {len(variants)} QPD variants.")
    
    raw_results = []
    mitigated_results = []
    
    # 3. Process Each Variant
    for i, variant_qc in enumerate(variants):
        print(f"\n--- Processing Variant {i+1} (Weight: {weights[i]}) ---")
        
        # A. Generate Data (from calibration.py)
        # We use the perturbed circuits as our training set
        print("   -> Generating perturbed calibration circuits...")
        train_circuits, train_ideals = generate_calibration_circuits([variant_qc], num_samples=30)
        
        # B. Execute Training Circuits (from executor.py)
        print("   -> Running calibration data on Backend...")
        # Note: run_mitigation_experiments handles measurement addition internally
        noisy_train_values = executor.run_mitigation_experiments(train_circuits, shots=8000)
        
        # C. Train Linear Model (PDR)
        print("   -> Fitting Local Linear Model (PDR)...")
        slope, intercept = train_linear_model(noisy_train_values, train_ideals)
        print(f"      [Model] Slope: {slope:.4f}, Intercept: {intercept:.4f}")
        
        # D. Execute Target Variant
        print("   -> Running actual variant...")
        raw_val = executor.run_mitigation_experiments([variant_qc], shots=8000)[0]
        
        # E. Mitigate using Linear Model
        # Ideal_est = m * Noisy + c
        mitigated_val = (slope * raw_val) + intercept
        
        print(f"      [Result] Raw: {raw_val:.4f} | Mitigated: {mitigated_val:.4f}")
        
        raw_results.append(raw_val)
        mitigated_results.append(mitigated_val)
        
    # 4. Reconstruction (from qpd_lib.py)
    print("\n>>> Reconstructing Global Results <<<")
    final_raw = reconstruct_qpd_result(raw_results, weights)
    final_mitigated = reconstruct_qpd_result(mitigated_results, weights)
    true_ideal = get_true_ideal_value_with_cz()
    
    # 5. Final Report
    print("="*50)
    print(f"True Ideal Target:              {true_ideal:.5f}")
    print("-" * 40)
    print(f"Standard QPD (Noisy):           {final_raw:.5f}")
    print(f"Error (Noisy):                  {abs(final_raw - true_ideal):.5f}")
    print("-" * 40)
    print(f"PDR-QPD (Mitigated):            {final_mitigated:.5f}")
    print(f"Error (Mitigated):              {abs(final_mitigated - true_ideal):.5f}")
    print("="*50)
    
    # Visualization of the Regression (for the last variant)
    print("\nGenerating Regression Plot for Variant 4...")
    plt.figure(figsize=(8, 5))
    plt.scatter(noisy_train_values, train_ideals, color='blue', alpha=0.6, label='Perturbed Data')
    
    # Plot the fit line
    x_range = np.linspace(min(noisy_train_values), max(noisy_train_values), 100)
    y_fit = slope * x_range + intercept
    plt.plot(x_range, y_fit, color='red', linestyle='--', label=f'Linear Fit (y={slope:.2f}x+{intercept:.2f})')
    
    # Plot the actual point
    plt.scatter([raw_val], [mitigated_val], color='green', marker='*', s=200, label='Actual Experiment')
    
    plt.xlabel('Noisy Expectation Value')
    plt.ylabel('Ideal Expectation Value')
    plt.title('Perturbation Data Regression (PDR) Fit')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()