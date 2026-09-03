"""Deterministic feature engineering and preprocessing for probability models.

DESIGN INVARIANTS:
1. Fitted ONLY on training data. Never fitted on test records.
2. Uses standardized domain enum vocabularies for stable categorical columns.
3. Completely serializable into model artifacts.
4. Provides inspectable feature names for weight interpretation.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    PaymentContext,
    PaymentMethod,
)


class FeaturePreprocessor:
    """Encodes and standardizes observable features for logistic regression."""

    def __init__(self):
        # Known categorical vocabularies from domain models
        self.categories = [c.value for c in FailureCategory]
        self.severities = [s.value for s in FailureSeverity]
        self.tiers = [t.value for t in CustomerTier]
        self.methods = [m.value for m in PaymentMethod]

        self.num_features = ["amount", "attempt_count", "failure_hour", "failure_day_of_week"]

        # Statistics fitted on training data
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted: bool = False

        self._feature_names: List[str] = []
        self._build_feature_names()

    def _build_feature_names(self) -> None:
        names = []
        for c in self.categories:
            names.append(f"cat_{c}")
        for s in self.severities:
            names.append(f"sev_{s}")
        for t in self.tiers:
            names.append(f"tier_{t}")
        for m in self.methods:
            names.append(f"method_{m}")
        for nf in self.num_features:
            names.append(f"num_{nf}")
        self._feature_names = names

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def num_dimensions(self) -> int:
        return len(self._feature_names)

    def fit(self, examples: List[Dict[str, Any]]) -> "FeaturePreprocessor":
        """Compute numerical feature scaling parameters strictly from training data."""
        if not examples:
            raise ValueError("Cannot fit FeaturePreprocessor on empty dataset.")

        for feat in self.num_features:
            vals = np.array([float(ex.get(feat, 0.0)) for ex in examples], dtype=np.float64)
            self.means[feat] = float(np.mean(vals))
            std_val = float(np.std(vals))
            self.stds[feat] = std_val if std_val > 1e-6 else 1.0

        self.is_fitted = True
        return self

    def _encode_single(self, ex: Dict[str, Any]) -> np.ndarray:
        """Encode a single feature dictionary into a 1D float vector."""
        vec = np.zeros(self.num_dimensions, dtype=np.float64)
        idx = 0

        # One-hot failure category
        cat_val = str(ex.get("failure_category", ""))
        for c in self.categories:
            if cat_val == c:
                vec[idx] = 1.0
            idx += 1

        # One-hot failure severity
        sev_val = str(ex.get("failure_severity", ""))
        for s in self.severities:
            if sev_val == s:
                vec[idx] = 1.0
            idx += 1

        # One-hot customer tier
        tier_val = str(ex.get("customer_tier", ""))
        for t in self.tiers:
            if tier_val == t:
                vec[idx] = 1.0
            idx += 1

        # One-hot payment method
        meth_val = str(ex.get("payment_method", ""))
        for m in self.methods:
            if meth_val == m:
                vec[idx] = 1.0
            idx += 1

        # Standardized numericals
        for nf in self.num_features:
            raw_val = float(ex.get(nf, 0.0))
            mean = self.means.get(nf, 0.0)
            std = self.stds.get(nf, 1.0)
            vec[idx] = (raw_val - mean) / (std + 1e-8)
            idx += 1

        return vec

    def transform(self, examples: List[Dict[str, Any]]) -> np.ndarray:
        """Transform a batch of feature dictionaries into a 2D numpy array (N, d)."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before calling transform.")
        if not examples:
            return np.empty((0, self.num_dimensions), dtype=np.float64)

        rows = [self._encode_single(ex) for ex in examples]
        return np.vstack(rows)

    def transform_context(self, context: PaymentContext, amount: float = 0.0) -> np.ndarray:
        """Transform a runtime PaymentContext object into a 2D array (1, d)."""
        ex = {
            "failure_category": context.failure_category.value,
            "failure_severity": context.failure_severity.value,
            "customer_tier": context.customer_tier.value,
            "payment_method": context.payment_method.value,
            "amount": amount,
            "attempt_count": context.attempt_count,
            "failure_hour": context.last_attempt_timestamp.hour if context.last_attempt_timestamp else 0,
            "failure_day_of_week": context.last_attempt_timestamp.weekday() if context.last_attempt_timestamp else 0,
        }
        return self._encode_single(ex).reshape(1, -1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessor state to a JSON-compatible dictionary."""
        return {
            "categories": self.categories,
            "severities": self.severities,
            "tiers": self.tiers,
            "methods": self.methods,
            "num_features": self.num_features,
            "means": self.means,
            "stds": self.stds,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturePreprocessor":
        """Deserialize preprocessor state from dictionary."""
        p = cls()
        p.categories = data["categories"]
        p.severities = data["severities"]
        p.tiers = data["tiers"]
        p.methods = data["methods"]
        p.num_features = data["num_features"]
        p.means = {k: float(v) for k, v in data["means"].items()}
        p.stds = {k: float(v) for k, v in data["stds"].items()}
        p.is_fitted = bool(data["is_fitted"])
        p._build_feature_names()
        return p
