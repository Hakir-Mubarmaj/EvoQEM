import numpy as np
import pickle
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

class LearnedCorrector:
    def __init__(self):
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) \
                 + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-15, 1e+1))
                 
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10,
            alpha=1e-10,
            normalize_y=True
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, noisy_input, ideal_values):
        if len(noisy_input) == 0:
            print("Warning: No training data provided.")
            return

        X = np.array(noisy_input).reshape(-1, 1)
        y = np.array(ideal_values)
        
        # Scale inputs
        X_scaled = self.scaler.fit_transform(X)
        error = y - X.flatten()
        
        try:
            self.gp.fit(X_scaled, error)
            self.is_trained = True
        except Exception as e:
            print(f"GP Training failed: {e}")

    def predict(self, noisy_input):
        if not self.is_trained:
            return noisy_input
            
        X = np.array(noisy_input).reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        predicted_error = self.gp.predict(X_scaled)
        corrected_values = X.flatten() + predicted_error
        return np.clip(corrected_values, -1.0, 1.0)

    # --- NEW: Save and Load Functions ---
    def save_model(self, filename):
        """Saves the trained model and scaler to a file."""
        with open(filename, 'wb') as f:
            pickle.dump({'gp': self.gp, 'scaler': self.scaler, 'is_trained': self.is_trained}, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Loads a trained model from a file."""
        if not os.path.exists(filename):
            print(f"Error: Model file {filename} not found.")
            return False
            
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.gp = data['gp']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']
        print(f"Model loaded from {filename}")
        return True