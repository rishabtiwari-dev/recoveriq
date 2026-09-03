"""Action-specific regularized logistic regression model and STOP invariant handler.

MATHEMATICAL FORMULATION (SPEC.md Section 7):
    P(recovery = 1 | x, a) = sigma(w_a^T x + b_a)
    where sigma(z) = 1 / (1 + exp(-z)).

STOP INVARIANT (SPEC.md Section 9 & Sprint 4 Requirements):
    Action.STOP is a terminal non-recovery action.
    P(recovery | x, STOP) = 0.0 strictly by definition.
    It is never fitted as a statistical model.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from recoveriq.domain.actions import Action


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


class ActionLogisticRegression:
    """Interpretable, regularized logistic regression for a single recovery action."""

    def __init__(
        self,
        action: Action,
        c_regularization: float = 1.0,
        random_state: int = 42,
    ):
        self.action = action
        self.c_regularization = c_regularization
        self.random_state = random_state

        self.weights: Optional[np.ndarray] = None  # (d,) float array
        self.bias: float = 0.0
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

        # STOP action is hardcoded domain invariant
        if self.action == Action.STOP:
            self.is_fitted = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> "ActionLogisticRegression":
        """Fit regularized logistic regression on feature matrix X and binary labels y."""
        self.feature_names = list(feature_names)

        # Domain invariant: STOP is never trained on data
        if self.action == Action.STOP:
            self.weights = np.zeros(len(feature_names), dtype=np.float64)
            self.bias = 0.0
            self.is_fitted = True
            return self

        if len(y) == 0:
            raise ValueError(f"Cannot fit model for action {self.action.value} on empty dataset.")

        # Check if labels have at least two classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Degenerate case: all 0 or all 1
            self.weights = np.zeros(len(feature_names), dtype=np.float64)
            self.bias = 10.0 if unique_classes[0] == 1 else -10.0
            self.is_fitted = True
            return self

        clf = LogisticRegression(
            C=self.c_regularization,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        clf.fit(X, y)

        self.weights = clf.coef_[0].astype(np.float64)
        self.bias = float(clf.intercept_[0])
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute recovery probabilities P(Y=1 | x, a) via sigmoid(w_a^T x + b_a)."""
        if not self.is_fitted:
            raise RuntimeError(f"ActionLogisticRegression for {self.action.value} must be fitted before prediction.")

        # STOP invariant: always returns 0.0
        if self.action == Action.STOP:
            return np.zeros(X.shape[0], dtype=np.float64)

        z = np.dot(X, self.weights) + self.bias
        probs = sigmoid(z)
        return np.clip(probs, 0.0, 1.0)

    @property
    def coefficients_dict(self) -> Dict[str, float]:
        """Return inspectable dictionary of feature name -> weight w_{a,j}."""
        if self.weights is None or not self.feature_names:
            return {}
        return {name: float(w) for name, w in zip(self.feature_names, self.weights)}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model parameters to an inspectable dictionary."""
        return {
            "action": self.action.value,
            "c_regularization": self.c_regularization,
            "random_state": self.random_state,
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionLogisticRegression":
        """Deserialize model from dictionary."""
        action = Action(data["action"])
        m = cls(
            action=action,
            c_regularization=float(data["c_regularization"]),
            random_state=int(data["random_state"]),
        )
        m.weights = np.array(data["weights"], dtype=np.float64) if data["weights"] else None
        m.bias = float(data["bias"])
        m.feature_names = list(data["feature_names"])
        m.is_fitted = bool(data["is_fitted"])
        return m
