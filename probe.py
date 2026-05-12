"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()

        self._scaler = StandardScaler()
        self._models = []
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()
        self._positive_rate: float = 0.5

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        return None

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return torch.from_numpy(logits).float()

    def _choose_threshold_by_accuracy(
        self,
        y_true: np.ndarray,
        probs: np.ndarray,
    ) -> float:
        candidates = np.unique(
            np.concatenate([
                probs,
                np.linspace(0.0, 1.0, 301),
            ])
        )

        best_threshold = 0.5
        best_accuracy = -1.0
        best_rate_diff = 10.0

        true_positive_rate = y_true.mean()

        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            accuracy = (y_pred == y_true).mean()
            rate_diff = abs(y_pred.mean() - true_positive_rate)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = float(threshold)
                best_rate_diff = rate_diff

            elif accuracy == best_accuracy and rate_diff < best_rate_diff:
                best_threshold = float(threshold)
                best_rate_diff = rate_diff

        return best_threshold

    def _choose_threshold_by_prior(
        self,
        probs: np.ndarray,
    ) -> float:
        return float(np.quantile(probs, 1.0 - self._positive_rate))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        np.random.seed(42)
        torch.manual_seed(42)

        self._positive_rate = float(y.mean())

        X_scaled = self._scaler.fit_transform(X)

        self._models = []

        pca_components = [16, 32, 64, 96, 128]
        c_values = [0.03, 0.1, 0.3, 1.0]
        class_weights = [None, "balanced"]

        for n_components in pca_components:
            n_components = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])

            pca = PCA(
                n_components=n_components,
                random_state=42,
            )
            X_pca = pca.fit_transform(X_scaled)

            for c in c_values:
                for class_weight in class_weights:
                    clf = LogisticRegression(
                        C=c,
                        class_weight=class_weight,
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=42,
                    )
                    clf.fit(X_pca, y)

                    self._models.append((pca, clf))

        train_probs = self.predict_proba(X)[:, 1]
        self._threshold = self._choose_threshold_by_prior(train_probs)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        self._threshold = self._choose_threshold_by_accuracy(y_val, probs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)

        all_probs = []

        for pca, clf in self._models:
            X_pca = pca.transform(X_scaled)
            probs = clf.predict_proba(X_pca)[:, 1]
            all_probs.append(probs)

        prob_pos = np.mean(all_probs, axis=0)

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

