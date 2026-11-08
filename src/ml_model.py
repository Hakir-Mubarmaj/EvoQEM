import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


class LearnedCorrector:
    def __init__(self, random_state=42):
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e3)) \
         + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-15, 1e+2))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-10,
            normalize_y=True,
            random_state=random_state,
        )

        self.scaler = StandardScaler()
        self.is_trained = False
        np.random.seed(random_state)

    def train(self, noisy_input, ideal_values):
        """
        Trains the GPR model. Accepts either 1D noisy_input (list of scalars) or 2D array-like (n_samples, n_features).

        noisy_input: list or numpy array of shape (n_samples,) or (n_samples, n_features)
        ideal_values: list or array of shape (n_samples,) (target expectation values computed from statevector)
        """
        if len(noisy_input) == 0:
            print("Warning: No training data provided to GP.")
            return

        X = np.array(noisy_input)
        y = np.array(ideal_values)

        # If X is 1D (only noisy scalar), reshape to (n_samples, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # The target we model is the *error* = ideal - noisy_scalar
        # If noisy scalar is present as the first column, use it; otherwise assume the user provided noisy scalar in X[:,0]
        noisy_scalar = X[:, 0]
        error = y - noisy_scalar

        # Scale inputs
        X_scaled = self.scaler.fit_transform(X)

        try:
            self.gp.fit(X_scaled, error)
            self.is_trained = True
        except Exception as e:
            print(f"GP Training failed: {e}")

    def predict(self, noisy_input):
        """
        Returns corrected value(s) (mean only). If model not trained returns raw noisy_input.
        """
        if not self.is_trained:
            print("Model not trained! Returning raw values.")
            return noisy_input

        X = np.array(noisy_input)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)
        predicted_error = self.gp.predict(X_scaled)
        corrected = X[:, 0] + predicted_error
        return np.clip(corrected, -1.0, 1.0)

    def predict_with_uncertainty(self, noisy_input, return_std=True, n_samples=1000):
        """
        Returns (mean, std) for the corrected values and an optional set of Monte-Carlo samples.
        Useful for propagating uncertainty when reconstructing global expectation.
        """
        if not self.is_trained:
            print("Model not trained! Returning raw values and infinite uncertainty.")
            X = np.array(noisy_input)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return X[:, 0], np.full(X.shape[0], np.nan)

        X = np.array(noisy_input)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)
        mean_err, std_err = self.gp.predict(X_scaled, return_std=True)
        mean_corrected = X[:, 0] + mean_err
        std_corrected = std_err

        # Monte Carlo sampling (optional)
        samples = None
        if n_samples is not None and n_samples > 0:
            rng = np.random.default_rng()
            samples_err = rng.normal(loc=mean_err, scale=std_err, size=(n_samples, len(mean_err)))
            samples = (X[:, 0][None, :] + samples_err)
            samples = np.clip(samples, -1.0, 1.0)

        return np.clip(mean_corrected, -1.0, 1.0), std_corrected, samples
