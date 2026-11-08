"""
pdr_regression.py — Unified PDR Regression Models
══════════════════════════════════════════════════
Three modes for correcting noisy QPD variant expectation values:

Mode 'linear':
    Regression f: E_noisy → E_ideal using TheilSen + OLS fallback.
    Formally justified by Kraus representation: for small ε,
    the depolarising channel acts linearly → E_ideal ≈ slope·E_noisy + intercept.

Mode 'gpr':
    Gaussian Process Regression with RBF + noise kernel.
    Handles non-linear noise without assuming a specific channel model.
    Bayesian posterior gives uncertainty estimates.

Mode 'bloch':
    Per-qubit Bloch-matrix inversion (Choi-inspired).
    Fits affine map: r_noisy = T_q @ r_ideal + t_q for each qubit q.
    Inverts to get corrected Bloch vector → corrected ZZ = Z_q0 × Z_q1.
    More expressive than scalar regression: captures cross-coupling
    between X, Y, Z noise components.

All modes report:
    fit_slope, fit_intercept, r2_score
    epsilon (depolarising parameter ε = 1 − 1/fit_slope)
    damping_m (paper-convention "m" = 1/fit_slope ≈ 1−ε)
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from channel_model import (interpret_regression_params, fit_bloch_matrix,
                            invert_channel_from_bloch)


class PDRRegressor:
    """
    Unified PDR regression model.

    Usage
    -----
    reg = PDRRegressor(mode='linear')
    reg.fit(cal_data)           # cal_data from calibration.generate_calibration_data()
    zz_corrected = reg.predict(noisy_zz_target)

    For 'bloch' mode, use:
    reg.fit(cal_data)           # cal_data in 'bloch' mode
    zz_corrected = reg.predict_bloch(noisy_bv_q0, noisy_bv_q1)
    """

    def __init__(self, mode: str = 'linear', seed: int = 42):
        assert mode in ('linear', 'gpr', 'bloch'), f"Unknown mode: {mode!r}"
        self.mode        = mode
        self.seed        = seed

        # Reported stats (set after fit)
        self.fit_slope   = np.nan
        self.fit_intercept = np.nan
        self.r2          = np.nan
        self.channel_info = {}

        # Internal model objects
        self._gpr_model   = None
        self._bloch_T_q0  = None
        self._bloch_t_q0  = None
        self._bloch_T_q1  = None
        self._bloch_t_q1  = None
        self._fitted      = False

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, cal_data: dict) -> 'PDRRegressor':
        """
        Fit the regression model from calibration data.

        Parameters
        ----------
        cal_data : dict
            Output of calibration.generate_calibration_data().
            Required keys:
              Scalar/GPR: 'noisy_zz', 'ideal_zz'
              Bloch:      + 'noisy_bloch_q0/q1', 'ideal_bloch_q0/q1'
        """
        if self.mode == 'linear':
            self._fit_linear(cal_data['noisy_zz'], cal_data['ideal_zz'])
        elif self.mode == 'gpr':
            self._fit_gpr(cal_data['noisy_zz'], cal_data['ideal_zz'])
        elif self.mode == 'bloch':
            self._fit_bloch(cal_data)

        self.channel_info = interpret_regression_params(
            self.fit_slope, self.fit_intercept)
        self._fitted = True
        return self

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict(self, noisy_zz: float) -> float:
        """
        Correct a noisy ZZ expectation value (scalar modes: linear, gpr).
        Returns ideal ZZ estimate.
        """
        assert self._fitted, "Call fit() before predict()."
        assert self.mode != 'bloch', "Use predict_bloch() for 'bloch' mode."

        if self.mode == 'linear':
            return float(self.fit_slope * noisy_zz + self.fit_intercept)
        elif self.mode == 'gpr':
            return float(self._gpr_model.predict([[noisy_zz]])[0])

    def predict_bloch(self, noisy_bv_q0: np.ndarray,
                       noisy_bv_q1: np.ndarray) -> float:
        """
        Correct via Bloch-matrix inversion (Choi-inspired mode).

        Parameters
        ----------
        noisy_bv_q0 : np.ndarray (3,)  — noisy [X, Y, Z] for qubit 0
        noisy_bv_q1 : np.ndarray (3,)  — noisy [X, Y, Z] for qubit 1

        Returns
        -------
        float : Corrected ⟨ZZ⟩ = corrected_Z_q0 × corrected_Z_q1
        """
        assert self._fitted and self.mode == 'bloch'

        corr_bv_q0 = invert_channel_from_bloch(
            noisy_bv_q0, self._bloch_T_q0, self._bloch_t_q0)
        corr_bv_q1 = invert_channel_from_bloch(
            noisy_bv_q1, self._bloch_T_q1, self._bloch_t_q1)

        # ZZ = Z_q0 × Z_q1 for product state (QPD variants are separable)
        corr_z0 = float(np.clip(corr_bv_q0[2], -1.0, 1.0))
        corr_z1 = float(np.clip(corr_bv_q1[2], -1.0, 1.0))
        return corr_z0 * corr_z1

    # ── Private fit methods ───────────────────────────────────────────────────

    def _fit_linear(self, noisy_zz: np.ndarray, ideal_zz: np.ndarray):
        """
        TheilSen regression with OLS fallback.

        Regression direction: ideal = slope · noisy + intercept
        → slope > 1 for depolarising (noisy values are smaller than ideal)

        TheilSen is robust to outlier hardware glitches.
        Falls back to OLS when TheilSen R² < OLS R² − 0.05.
        """
        # OLS via scipy (fast)
        slope_ols, intercept_ols, r_ols, _, _ = stats.linregress(
            noisy_zz, ideal_zz)
        r2_ols = r_ols ** 2

        # TheilSen (robust)
        ts = TheilSenRegressor(random_state=self.seed,
                               max_subpopulation=1000)
        ts.fit(noisy_zz.reshape(-1, 1), ideal_zz)
        pred_ts = ts.predict(noisy_zz.reshape(-1, 1))
        ss_res  = np.sum((ideal_zz - pred_ts) ** 2)
        ss_tot  = np.sum((ideal_zz - ideal_zz.mean()) ** 2)
        r2_ts   = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        # Choose the better fit
        if r2_ts >= r2_ols - 0.05:
            self.fit_slope     = float(ts.coef_[0])
            self.fit_intercept = float(ts.intercept_)
            self.r2            = r2_ts
        else:
            self.fit_slope     = float(slope_ols)
            self.fit_intercept = float(intercept_ols)
            self.r2            = r2_ols

    def _fit_gpr(self, noisy_zz: np.ndarray, ideal_zz: np.ndarray):
        """
        Gaussian Process Regression with RBF + WhiteKernel.

        Handles non-linear noise profiles.
        Fit slope/intercept reported as linear approximation for reporting.
        """
        kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e1)) \
               + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))
        self._gpr_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=self.seed,
            normalize_y=True,
        )
        self._gpr_model.fit(noisy_zz.reshape(-1, 1), ideal_zz)

        pred    = self._gpr_model.predict(noisy_zz.reshape(-1, 1))
        ss_res  = np.sum((ideal_zz - pred) ** 2)
        ss_tot  = np.sum((ideal_zz - ideal_zz.mean()) ** 2)
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        # Linear approximation for slope reporting
        slope_lin, intercept_lin, _, _, _ = stats.linregress(noisy_zz, pred)
        self.fit_slope     = float(slope_lin)
        self.fit_intercept = float(intercept_lin)

    def _fit_bloch(self, cal_data: dict):
        """
        Fit per-qubit Bloch-matrix noise models.

        For each qubit q:
            r_noisy_q = T_q @ r_ideal_q + t_q

        For depolarising: T_q = (1−ε)·I₃, t_q = 0.
        This is the Choi-inspired full-channel characterisation.
        """
        self._bloch_T_q0, self._bloch_t_q0 = fit_bloch_matrix(
            cal_data['ideal_bloch_q0'], cal_data['noisy_bloch_q0'])
        self._bloch_T_q1, self._bloch_t_q1 = fit_bloch_matrix(
            cal_data['ideal_bloch_q1'], cal_data['noisy_bloch_q1'])

        # Depolarising parameter from Z-diagonal of T (most relevant component)
        eps_q0 = 1.0 - self._bloch_T_q0[2, 2]   # From Z-Z component
        eps_q1 = 1.0 - self._bloch_T_q1[2, 2]
        epsilon_mean = float((eps_q0 + eps_q1) / 2)

        # Report slope as equivalent scalar fit for consistency
        damping = 1.0 - epsilon_mean
        self.fit_slope     = 1.0 / damping if damping > 1e-6 else np.nan
        self.fit_intercept = 0.0

        # R² on Z component of qubit 0 (for comparison with scalar methods)
        z_pred = (self._bloch_T_q0[2, :] @ cal_data['ideal_bloch_q0'].T
                  + self._bloch_t_q0[2])
        z_true = cal_data['noisy_bloch_q0'][:, 2]
        ss_res = np.sum((z_true - z_pred) ** 2)
        ss_tot = np.sum((z_true - z_true.mean()) ** 2)
        self.r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
