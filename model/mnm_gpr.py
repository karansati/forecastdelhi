"""Sequential MNM-GPR hybrid: the best model from the study.

Stage 1 - Fundamental Multiplicative Neuron Model (MNM):
    A single multiplicative neuron aggregates the inputs through a product
    of weighted, biased inputs and applies a logistic activation:
        u = prod_i (w_i * x_i + b_i)        (clipped for stability)
        y = sigmoid(u)
    Trained by gradient descent on the mean square error.

Stage 2 - Gaussian Process Regression on the MNM residuals:
    A GPR with an RBF kernel is fitted to the residuals (actual - MNM) of the
    training set. It both corrects the systematic error and supplies a
    predictive standard deviation, giving a confidence band on the forecast.

Final forecast (normalised space):  y_hat = y_mnm + residual_gpr_mean
Confidence band: +/- 1.96 * residual_gpr_std  (95%)
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

CLIP = 50.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP, CLIP)))


class MNM:
    """Single fundamental multiplicative neuron, trained with Adam.

    The inputs are shifted internally by `shift` so that each factor
    (w_i * (x_i + shift) + b_i) stays well away from zero; without this the
    product of many inputs in [0, 1] collapses and the gradient vanishes.
    """

    def __init__(self, n_features, eta=0.02, epochs=80, batch=256, stall=15,
                 tol=1e-7, shift=0.5, seed=0):
        self.eta, self.epochs, self.batch = eta, epochs, batch
        self.stall, self.tol, self.shift = stall, tol, shift
        rng = np.random.default_rng(seed)
        self.w = rng.normal(1.0, 0.3, n_features)
        self.b = rng.normal(0.0, 0.3, n_features)
        self.seed = seed

    def _forward(self, X):
        Xs = X + self.shift
        k = np.clip(Xs * self.w + self.b, -1e3, 1e3)     # (n, d)
        u = np.clip(np.prod(k, axis=1), -CLIP, CLIP)
        return Xs, k, u, _sigmoid(u)

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        mw = vw = np.zeros(d); mb = vb = np.zeros(d)
        b1, b2, eps = 0.9, 0.999, 1e-8
        best, best_wb, stalled, tstep = np.inf, (self.w.copy(), self.b.copy()), 0, 0
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):       # mini-batches escape the constant trap
                bi = idx[s:s + self.batch]
                Xs, k, u, yhat = self._forward(X[bi])
                g = (yhat - y[bi]) * yhat * (1.0 - yhat)
                loo = _leave_one_out_prod(k)
                gw = (g[:, None] * (Xs * loo)).mean(0)
                gb = (g[:, None] * loo).mean(0)
                tstep += 1
                mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
                mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
                self.w -= self.eta * (mw / (1 - b1 ** tstep)) / (np.sqrt(vw / (1 - b2 ** tstep)) + eps)
                self.b -= self.eta * (mb / (1 - b1 ** tstep)) / (np.sqrt(vb / (1 - b2 ** tstep)) + eps)
            mse = float(np.mean((self.predict(X) - y) ** 2))
            if best - mse > self.tol:
                best, best_wb, stalled = mse, (self.w.copy(), self.b.copy()), 0
            else:
                stalled += 1
                if stalled >= self.stall:
                    break
        self.w, self.b = best_wb
        return self

    def predict(self, X):
        return self._forward(X)[3]


def _leave_one_out_prod(k):
    """For each element, product of all OTHER elements in its row (prefix*suffix)."""
    n, d = k.shape
    pre = np.ones((n, d))
    suf = np.ones((n, d))
    for i in range(1, d):
        pre[:, i] = pre[:, i - 1] * k[:, i - 1]
    for i in range(d - 2, -1, -1):
        suf[:, i] = suf[:, i + 1] * k[:, i + 1]
    return np.clip(pre * suf, -CLIP, CLIP)


class SequentialMNMGPR:
    """The full best model: MNM base + residual GPR (with uncertainty)."""

    def __init__(self, gpr_subset=3000, eta=0.02, epochs=80, batch=256, seed=0):
        self.gpr_subset = gpr_subset
        self.eta, self.epochs, self.batch, self.seed = eta, epochs, batch, seed
        self.mnm = None
        self.gpr = None
        self.sigma_cal = 0.0      # calibration noise floor (scaled units)

    def _fit_core(self, Xs, ys):
        d = Xs.shape[1]
        mnm = MNM(d, eta=self.eta, epochs=self.epochs, batch=self.batch,
                  seed=self.seed).fit(Xs, ys)
        resid = ys - mnm.predict(Xs)
        rng = np.random.default_rng(self.seed)
        m = min(self.gpr_subset, len(Xs))
        idx = rng.choice(len(Xs), size=m, replace=False)
        kernel = (ConstantKernel(1.0)
                  * RBF(length_scale=0.4)
                  + WhiteKernel(noise_level=1e-3))
        gpr = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, normalize_y=True, random_state=self.seed)
        gpr.fit(Xs[idx], resid[idx])
        return mnm, gpr

    def fit(self, Xs, ys):
        """Xs, ys already scaled (X in [0,1], y in [TGT_LO, TGT_HI])."""
        # hold out the last 15% to calibrate the prediction interval
        c = int(0.85 * len(Xs))
        mnm_c, gpr_c = self._fit_core(Xs[:c], ys[:c])
        cal_pred = mnm_c.predict(Xs[c:]) + gpr_c.predict(Xs[c:])
        self.sigma_cal = float(np.std(ys[c:] - cal_pred))
        # refit on ALL training data for the deployed model
        self.mnm, self.gpr = self._fit_core(Xs, ys)
        return self

    def predict(self, Xs, return_std=False):
        base = self.mnm.predict(Xs)
        rmean, rstd = self.gpr.predict(Xs, return_std=True)
        yhat = base + rmean
        if return_std:
            total = np.sqrt(rstd ** 2 + self.sigma_cal ** 2)   # calibrated band
            return yhat, total
        return yhat
