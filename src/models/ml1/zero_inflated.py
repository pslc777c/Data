from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """Two-stage regressor: P(y>0) * E[y|y>0], with optional hard-zero threshold."""

    def __init__(
        self,
        classifier=None,
        regressor=None,
        zero_threshold: float = 0.35,
        min_positive: float = 1e-6,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.zero_threshold = float(zero_threshold)
        self.min_positive = float(min_positive)

    def fit(self, x, y, sample_weight=None):
        y_arr = np.asarray(y, dtype=np.float64)
        pos = y_arr > self.min_positive

        if self.classifier is None:
            self.classifier_ = HistGradientBoostingClassifier(
                random_state=42,
                max_depth=5,
                learning_rate=0.05,
                max_iter=250,
                min_samples_leaf=10,
            )
        else:
            self.classifier_ = clone(self.classifier)

        if self.regressor is None:
            self.regressor_ = HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=42,
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=10,
            )
        else:
            self.regressor_ = clone(self.regressor)

        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if sw is not None and len(sw) != len(y_arr):
            sw = None

        y_bin = pos.astype(np.float64)
        try:
            if sw is None:
                self.classifier_.fit(x, y_bin)
            else:
                self.classifier_.fit(x, y_bin, sample_weight=sw)
        except TypeError:
            self.classifier_.fit(x, y_bin)

        if bool(pos.any()):
            x_pos = x[pos]
            y_pos = y_arr[pos]
            sw_pos = sw[pos] if sw is not None else None
            try:
                if sw_pos is None:
                    self.regressor_.fit(x_pos, y_pos)
                else:
                    self.regressor_.fit(x_pos, y_pos, sample_weight=sw_pos)
            except TypeError:
                self.regressor_.fit(x_pos, y_pos)
            self.has_positive_ = True
            self.pos_mean_ = float(np.average(y_pos, weights=sw_pos)) if sw_pos is not None and sw_pos.sum() > 0 else float(np.mean(y_pos))
        else:
            self.has_positive_ = False
            self.pos_mean_ = 0.0

        return self

    def _predict_pos_proba(self, x) -> np.ndarray:
        if hasattr(self.classifier_, "predict_proba"):
            p = np.asarray(self.classifier_.predict_proba(x), dtype=np.float64)[:, 1]
        else:
            p = np.asarray(self.classifier_.predict(x), dtype=np.float64)
        return np.clip(p, 0.0, 1.0)

    def predict(self, x):
        p_pos = self._predict_pos_proba(x)
        if self.has_positive_:
            y_pos = np.asarray(self.regressor_.predict(x), dtype=np.float64)
            y_pos = np.where(np.isfinite(y_pos), y_pos, self.pos_mean_)
            y_pos = np.clip(y_pos, 0.0, None)
        else:
            y_pos = np.full(len(p_pos), self.pos_mean_, dtype=np.float64)

        y_hat = p_pos * y_pos
        y_hat = np.where(p_pos < self.zero_threshold, 0.0, y_hat)
        return y_hat.astype(np.float64)

