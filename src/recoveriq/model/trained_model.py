"""Trained statistical recovery probability model implementing the RecoveryProbabilityModel protocol.

ARCHITECTURAL CONTRACT:
- Implements RecoveryProbabilityModel:
      def estimate_probabilities(self, context: PaymentContext) -> Dict[Action, ProbabilityEstimate]:
- Evaluates candidate actions using action-specific regularized logistic regressions:
      P(Y=1 | x, a) = sigma(w_a^T x + b_a)
- Enforces STOP domain invariant: P(recovery | STOP) = Decimal("0.00") strictly.
- Fails fast on missing/corrupted artifacts. NEVER silently falls back to StubProbabilityModel.
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Union

from recoveriq.domain.actions import Action
from recoveriq.domain.models import PaymentContext
from recoveriq.model.logistic_regression import ActionLogisticRegression
from recoveriq.model.preprocessing import FeaturePreprocessor
from recoveriq.model.probability import ProbabilityEstimate, RecoveryProbabilityModel


class TrainedRecoveryProbabilityModel:
    """Interpretable statistical probability model consisting of 6 action-specific submodels."""

    def __init__(
        self,
        preprocessor: FeaturePreprocessor,
        action_models: Dict[Action, ActionLogisticRegression],
        model_version: str = "logistic-regression-v1",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.preprocessor = preprocessor
        self.action_models = action_models
        self.model_version = model_version
        self.metadata = metadata or {}

        # Ensure all 6 actions are represented in action_models
        for action in Action:
            if action not in self.action_models:
                raise ValueError(f"Missing action model for {action.value}")

    def estimate_probabilities(
        self,
        context: PaymentContext,
    ) -> Dict[Action, ProbabilityEstimate]:
        """Estimate recovery probability P(recovery | context, a) for all candidate actions.

        Conforms strictly to the RecoveryProbabilityModel protocol.
        """
        estimates: Dict[Action, ProbabilityEstimate] = {}

        # Transform context into numerical feature vector (1, d)
        # Decision-time amount is 0.0 if not in context, or read from extra_metadata if passed
        amount = float(context.extra_metadata.get("amount", 0.0))
        X = self.preprocessor.transform_context(context, amount=amount)

        for action in Action:
            if action == Action.STOP:
                # Deterministic domain invariant
                prob_dec = Decimal("0.00")
            else:
                model = self.action_models[action]
                p_raw = float(model.predict_proba(X)[0])
                # Format to 4 decimal places for stable precision
                prob_dec = Decimal(f"{p_raw:.4f}")
                # Clamp within [0.00, 1.00]
                prob_dec = max(Decimal("0.00"), min(Decimal("1.00"), prob_dec))

            estimates[action] = ProbabilityEstimate(
                action=action,
                probability=prob_dec,
                model_version=self.model_version,
            )

        return estimates

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize complete model artifact (preprocessor + all 6 action models + metadata)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model_version": self.model_version,
            "metadata": self.metadata,
            "preprocessor": self.preprocessor.to_dict(),
            "action_models": {
                action.value: model.to_dict()
                for action, model in self.action_models.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainedRecoveryProbabilityModel":
        """Load model artifact from file. Fails fast if file is missing or invalid."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{path}'. "
                "Failing fast: trained probability model requires an explicit artifact."
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
        except Exception as e:
            raise ValueError(f"Corrupt or unparseable model artifact at '{path}': {e}") from e

        if "preprocessor" not in artifact or "action_models" not in artifact:
            raise ValueError(f"Invalid model artifact schema at '{path}': missing required sections.")

        preprocessor = FeaturePreprocessor.from_dict(artifact["preprocessor"])

        action_models: Dict[Action, ActionLogisticRegression] = {}
        for action_val, model_data in artifact["action_models"].items():
            action = Action(action_val)
            action_models[action] = ActionLogisticRegression.from_dict(model_data)

        return cls(
            preprocessor=preprocessor,
            action_models=action_models,
            model_version=artifact.get("model_version", "unknown"),
            metadata=artifact.get("metadata", {}),
        )
