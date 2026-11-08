import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
import csv
import pandas as pd

# Import existing modules
from src.calibration import generate_calibration_circuits
from src.executor import VirtualExecutor
from src.qpd_lib import decompose_cz_gate, reconstruct_qpd_result

from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import r2_score


class RobustPDRPipeline:
    def __init__(self, backend_name='fezi'):
        self.executor = VirtualExecutor(backend_name=backend_name)
        self.target_circuit = self._get_entangled_circuit()
        self.ideal_value = self._get_true_ideal_value()
        
    """def _get_entangled_circuit(self):
        #Creates target circuit (RX-RX-CZ).
        qc = QuantumCircuit(2)
        qc.rx(0.6, 0)
        qc.rx(0.8, 1)
        return qc

    def _get_true_ideal_value(self):
        #Calculates theoretical ideal value.
        qc = QuantumCircuit(2)
        qc.rx(0.6, 0)
        qc.rx(0.8, 1)
        qc.cz(0, 1)
        psi = Statevector.from_instruction(qc)
        op = Pauli('ZZ')
        return psi.expectation_value(op).real"""
    
    def _get_entangled_circuit(self):
        """
        Creates target circuit based on CPDR Paper (Ising Model Time Evolution).
        Ref: Page 21, Section IV.C & Eq. IV2.
        """
        # Parameters strictly from the paper
        N_steps = 5              # Paper uses N=5 Trotter steps
        theta_J = -np.pi / 2     # Fixed parameter mentioned in paper
        theta_h = 1.0            # Paper varies this in [0, pi/2]. We picked 1.0 as a test case.
        
        qc = QuantumCircuit(2)
        
        # Trotterized Evolution
        for _ in range(N_steps):
            # Layer 1: RX gates (Transverse Field)
            qc.rx(theta_h, 0)
            qc.rx(theta_h, 1)
            
            # Layer 2: RZZ gates (Interaction)
            # Note: RZZ is not a native gate on many backends, so Qiskit/Executor 
            # might decompose it to CNOT-RZ-CNOT automatically.
            qc.rzz(theta_J, 0, 1)
            
        # Paper uses heavy-weight observables, but for 2-qubit demo 
        # we stick to the interaction measurement setup you likely have.
        return qc

    def _get_true_ideal_value(self):
        """Calculates theoretical ideal value for the CPDR Ising circuit."""
        # Re-create the exact same circuit to calculate expectation
        N_steps = 5
        theta_J = -np.pi / 2
        theta_h = 1.0
        
        qc = QuantumCircuit(2)
        for _ in range(N_steps):
            qc.rx(theta_h, 0)
            qc.rx(theta_h, 1)
            qc.rzz(theta_J, 0, 1)
            
        # Standard calculation using Statevector
        psi = Statevector.from_instruction(qc)
        op = Pauli('ZZ') # Measuring ZZ correlation (Ising energy term)
        return psi.expectation_value(op).real

    def _fit_pdr_model(self, X_train, y_train):
        # try ordinary least squares first
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_train, y_train)
        r2 = r_value**2
        if r2 < 0.82:
            # fallback to robust Theil-Sen
            model = TheilSenRegressor(random_state=0)
            X = np.array(X_train).reshape(-1,1)
            y = np.array(y_train)
            model.fit(X, y)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            r2 = r2_score(y, model.predict(X))
            print(f"    [Robust fit used] new R2={r2:.3f}")
        return slope, intercept, r2, p_value, std_err

    """def _fit_pdr_model(self, X_train, y_train):
        
        Non-linear local fit using Gaussian Process Regression (sklearn).
        Returns: slope_like (we'll return mean derivative approx as slope), intercept, r2, p_value (nan), std_err (nan)
        Note: for small sample sizes GPR works well. For large sample sizes consider subsampling.
        
        import numpy as _np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        # prepare arrays
        X = _np.asarray(X_train).ravel().astype(float)
        y = _np.asarray(y_train).ravel().astype(float)

        # defensive checks
        if X.size < 2 or y.size < 2:
            intercept = float(y.mean()) if y.size>0 else 0.0
            return 0.0, intercept, 0.0, float('nan'), float('nan')

        # limit sample size (optional): if too big, subsample randomly but reproducibly
        MAX_N = 200
        if X.size > MAX_N:
            rng = _np.random.default_rng(42)
            idx = rng.choice(len(X), size=MAX_N, replace=False)
            X = X[idx]
            y = y[idx]

        # reshape for sklearn
        X_col = X.reshape(-1,1)

        # scale inputs and outputs (important for kernel lengthscale)
        xsc = StandardScaler()
        ysc = StandardScaler()
        Xs = xsc.fit_transform(X_col)
        ys = ysc.fit_transform(y.reshape(-1,1)).ravel()

        # kernel: Constant * (Matern or RBF) + WhiteKernel (noise)
        # Try Matern(nu=1.5) or RBF. Good default:
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=1.5) \
                + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1))

        # instantiate GPR
        gpr = GaussianProcessRegressor(kernel=kernel,
                                    alpha=0.0,            # we use WhiteKernel for noise
                                    normalize_y=False,
                                    n_restarts_optimizer=5,
                                    random_state=42)

        # fit
        try:
            gpr.fit(Xs, ys)
        except Exception as e:
            # fallback to mean prediction if GPR fails
            intercept = float(y.mean())
            return 0.0, intercept, 0.0, float('nan'), float('nan')

        # predictions on training (to compute r2)
        y_pred_scaled = gpr.predict(Xs)
        y_pred = ysc.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

        # compute r2 on original scale
        try:
            r2 = float(r2_score(y, y_pred))
        except Exception:
            r2 = 0.0

        # For interpretability, estimate a slope-like value by finite difference of the GP mean around X mean
        x_mean_scaled = _np.array([[0.0]])  # after StandardScaler, mean is 0
        delta = 1e-3
        m_plus = gpr.predict(x_mean_scaled + delta)
        m_minus = gpr.predict(x_mean_scaled - delta)
        # derivative in scaled-output space
        deriv_scaled = (m_plus - m_minus) / (2 * delta)
        # convert derivative back to original scale: dy/dx = (ys.std_/xsc.std_)*deriv_scaled
        scale_factor = (ysc.scale_[0] / xsc.scale_[0]) if hasattr(ysc, 'scale_') and hasattr(xsc, 'scale_') else 1.0
        slope_like = float(deriv_scaled.ravel()[0] * scale_factor)

        # intercept approximation: use mean(y) - slope * mean(x)
        intercept = float(y.mean() - slope_like * X.mean())

        # optionally, you can also return predictive uncertainty on new raw point via gpr.predict(..., return_std=True)

        return slope_like, intercept, r2, float('nan'), float('nan')"""


    def run_single_experiment(self, n_repeat):
        """
        Executes ONE complete iteration of the QPD + PDR pipeline.
        Returns: (noisy_val, mitigated_val)
        """
        # 1. Decompose
        qpd_data = decompose_cz_gate(self.target_circuit, qubit_pair=(0, 1))
        variants = [item[0] for item in qpd_data]
        weights = [item[1] for item in qpd_data]
        
        variant_raw_results = []
        variant_mitigated_results = []
        
        all_variant_stats = []
        
        # 2. Process Variants
        for i, variant_qc in enumerate(variants):
            # A. Generate Calibration Data (Structure-Aware)
            # Using 25 samples per variant for speed vs accuracy balance
            train_circuits, train_ideals = generate_calibration_circuits([variant_qc], num_samples=50)
            
            # B. Run Calibration
            # Executor handles measurement addition internally
            noisy_train_X = self.executor.run_mitigation_experiments(train_circuits, shots=8000)
            
            # C. Train PDR Model (Local Linear Regression)
            slope, intercept, r2, p_value, std_err = self._fit_pdr_model(noisy_train_X, train_ideals)
            
            # Validation check: If R^2 is too low, the noise model might be non-linear or random
            if r2 < 0.8:
                print(f"  [Warning] low R2 for variant {i+1}: R2={r2:.3f}")
                
            # D. Run Target Variant
            raw_val = self.executor.run_mitigation_experiments([variant_qc], shots=12000)[0]
            
            # E. Mitigate
            mitigated_val = (slope * raw_val) + intercept
            
            variant_raw_results.append(raw_val)
            variant_mitigated_results.append(mitigated_val)
            all_variant_stats.append({
                "variant": i+1,
                "slope": slope, "intercept": intercept, "r2": r2,
                "p_value": p_value, "std_err": std_err,
                "raw_val": raw_val, "mit_val": mitigated_val
            })
            
        # 3. Reconstruct Global Result
        global_raw = reconstruct_qpd_result(variant_raw_results, weights)
        global_mitigated = reconstruct_qpd_result(variant_mitigated_results, weights)
        filename = f'csv/variant_stats{n_repeat}.csv'
        pd.DataFrame(all_variant_stats).to_csv(filename, index=False)
        
        return global_raw, global_mitigated

    def run_statistical_validation(self, n_repeats=10):
        """
        Runs the experiment N times to gather statistics (Mean, Std, CI).
        This addresses the 'Single-run risk' and 'QPD Variance' issues.
        """
        print(f"\n=== Starting Robust Statistical Validation (N={n_repeats}) ===")
        print(f"Target Ideal Value: {self.ideal_value:.5f}\n")
        
        noisy_distribution = []
        mitigated_distribution = []
        
        for i in range(n_repeats):
            print(f" -> Run {i+1}/{n_repeats}...", end="\r")
            try:
                raw, mit = self.run_single_experiment(i+1)
                noisy_distribution.append(raw)
                mitigated_distribution.append(mit)
            except Exception as e:
                print(f"\n[Error in Run {i+1}]: {e}")
        
        print("\n\n" + "="*60)
        print("STATISTICAL ANALYSIS REPORT")
        print("="*60)
        
        # Helper to calc stats
        def print_stats(name, data, ideal):
            mean = np.mean(data)
            std = np.std(data, ddof=1) # Sample standard deviation
            sem = std / np.sqrt(len(data)) # Standard Error of Mean
            
            # 95% Confidence Interval (t-distribution)
            confidence = 0.95
            h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
            
            error = abs(mean - ideal)
            
            print(f"[{name}]")
            print(f"  Mean:       {mean:.5f}")
            print(f"  Std Dev:    {std:.5f}")
            print(f"  95% CI:     [{mean-h:.5f}, {mean+h:.5f}]")
            print(f"  Abs Error:  {error:.5f}")
            print("-" * 40)
            return mean, h

        # Analyze
        noisy_mean, noisy_ci = print_stats("Standard QPD (Noisy)", noisy_distribution, self.ideal_value)
        mit_mean, mit_ci = print_stats("PDR-QPD (Mitigated)", mitigated_distribution, self.ideal_value)
        
        # Improvement Logic
        reduction = (abs(noisy_mean - self.ideal_value) - abs(mit_mean - self.ideal_value))
        print(f"Net Error Reduction: {reduction:.5f}")
        
        noisy = np.array(noisy_distribution)
        mit = np.array(mitigated_distribution)
        ideal = self.ideal_value  # or self.ideal_value if inside class

        # per-run absolute errors
        err_raws = np.abs(noisy - ideal)
        err_mits = np.abs(mit - ideal)
        diffs = err_raws - err_mits   # positive => improvement

        # Basic stats
        print("N runs:", len(diffs))
        print("Mean raw err:", err_raws.mean(), "±", err_raws.std(ddof=1))
        print("Mean mit err:", err_mits.mean(), "±", err_mits.std(ddof=1))
        print("Mean improvement (raw_err - mit_err):", diffs.mean())

        # Fraction improved
        frac_improved = np.mean(diffs > 0)
        print("Fraction of runs improved:", frac_improved)

        # Paired t-test on errors (mit vs raw)
        t, p = stats.ttest_rel(err_mits, err_raws)
        print("Paired t-test on errors: t = %.4f, p = %.4g" % (t, p))

        # Bootstrap 95% CI for mean improvement
        rng = np.random.default_rng(12345)
        B = 5000
        boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(B)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print("Bootstrap 95% CI for mean improvement:", (lo, hi))

        # Visualization
        self._plot_results(noisy_distribution, mitigated_distribution, noisy_mean, mit_mean, noisy_ci, mit_ci)

    def _plot_results(self, noisy_data, mit_data, noisy_mean, mit_mean, noisy_ci, mit_ci):
        """Generates a Research-Grade Box Plot with Error Bars."""
        plt.figure(figsize=(10, 6))
        
        # 1. Box Plots for Distribution
        plt.boxplot([noisy_data, mit_data], positions=[1, 2], widths=0.4, 
                    patch_artist=True, boxprops=dict(facecolor="lightgray"))
        
        # 2. Individual Run Points (Jitter) to show distribution density
        x_noisy = np.random.normal(1, 0.04, len(noisy_data))
        x_mit = np.random.normal(2, 0.04, len(mit_data))
        plt.scatter(x_noisy, noisy_data, alpha=0.6, color='red', label='Individual Runs')
        plt.scatter(x_mit, mit_data, alpha=0.6, color='green')
        
        # 3. Ideal Line
        plt.axhline(self.ideal_value, color='blue', linestyle='--', linewidth=2, label=f'Ideal ({self.ideal_value:.4f})')
        
        # 4. Error Bars (95% CI)
        plt.errorbar(1, noisy_mean, yerr=noisy_ci, fmt='o', color='black', capsize=5, label='Mean ± 95% CI')
        plt.errorbar(2, mit_mean, yerr=mit_ci, fmt='o', color='black', capsize=5)
        
        plt.xticks([1, 2], ['Standard QPD\n(Unmitigated)', 'PDR-QPD\n(Mitigated)'], fontsize=12)
        plt.ylabel('Expectation Value <ZZ>', fontsize=12)
        plt.title(f'Statistical Validation of PDR-QPD (N={len(noisy_data)})', fontsize=14)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run the robust pipeline with 10 repeats
    pipeline = RobustPDRPipeline(backend_name='fez')
    pipeline.run_statistical_validation(n_repeats=5)