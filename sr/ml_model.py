# Patched script — small excerpt from your draft for reference (truncated)\n# -------------------------------------------------------------\n# file: Calibration.py

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter

def generate_calibration_circuits(base_subcircuits, num_samples=8):
    """
    Smart Calibration: Modifies EXISTING angles instead of adding new gates.
    This preserves the circuit structure and prevents SWAP explosion.
    """
    training_circuits = []
    ideal_values = []
    
    target_qc = base_subcircuits[0] 
    
    print(f"Generating {num_samples} smart variants...")
    
    for _ in range(num_samples):
        # 1. সার্কিট কপি করি
        cal_qc = target_qc.copy()
        
        # 2. স্ট্রাকচার ঠিক রেখে প্যারামিটার চেঞ্জ করা
        # আমরা সার্কিটের সব ইনস্ট্রাকশন চেক করব
        # যদি সেটা রোটেশন গেট হয় (rx, ry, rz), আমরা একটু নয়েজ অ্যাড করব\n# -------------------------------------------------------------\n\n
# ml_model.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

class LearnedCorrector:
    def __init__(self):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0, normalize_y=True)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, noisy_features, ideal_values):
        X = np.array(noisy_features)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(ideal_values)
        errors = y - X[:, 0]
        Xs = self.scaler.fit_transform(X)
        try:
            self.gp.fit(Xs, errors)
            self.is_trained = True
        except Exception as e:
            print("GP training failed:", e)
            self.is_trained = False

    def correct(self, noisy_feature):
        if not self.is_trained:
            return noisy_feature if np.isscalar(noisy_feature) else noisy_feature[0]
        X = np.array(noisy_feature)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = self.scaler.transform(X)
        pred, std = self.gp.predict(Xs, return_std=True)
        corrected = (X[0, 0] + pred[0])
        return corrected
