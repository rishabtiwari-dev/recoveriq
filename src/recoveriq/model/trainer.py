"""Model trainer coordinating counterfactual dataset building, feature fitting, and model estimation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np

from recoveriq.domain.actions import Action
from recoveriq.model.dataset import CounterfactualDatasetBuilder
from recoveriq.model.logistic_regression import ActionLogisticRegression
from recoveriq.model.preprocessing import FeaturePreprocessor
from recoveriq.model.trained_model import TrainedRecoveryProbabilityModel
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.schema import SyntheticPaymentRecord


class ModelTrainer:
    """Trains 6 independent action models on counterfactually evaluated training records."""

    def __init__(
        self,
        c_regularization: float = 1.0,
        random_state: int = 42,
        model_version: str = "logistic-regression-v1",
    ):
        self.c_regularization = c_regularization
        self.random_state = random_state
        self.model_version = model_version

    def train(
        self,
        train_records: List[SyntheticPaymentRecord],
        env: SimulationEnvironment,
    ) -> TrainedRecoveryProbabilityModel:
        """Execute end-to-end model training strictly on the training partition.

        CRITICAL ANTI-LEAKAGE INVARIANT:
        - train_records must come exclusively from the training partition.
        - env must be initialized strictly with training ground truth records.
        - Neither true_probability nor latent_recoverability_profile is ever seen by the model.
        """
        # 1. Build full factorial counterfactual dataset (N payments x 6 actions)
        dataset_builder = CounterfactualDatasetBuilder()
        dataset = dataset_builder.build_dataset(
            observable_records=train_records,
            env=env,
            seed=self.random_state,
        )

        # 2. Fit feature preprocessor strictly on training features
        all_feature_dicts, _ = dataset.get_feature_matrix_and_labels()
        preprocessor = FeaturePreprocessor()
        preprocessor.fit(all_feature_dicts)

        # 3. Train independent logistic regression for each action
        action_models: Dict[Action, ActionLogisticRegression] = {}

        for action in Action:
            action_feat_dicts, action_labels = dataset.get_feature_matrix_and_labels(action)
            X_action = preprocessor.transform(action_feat_dicts)
            y_action = np.array(action_labels, dtype=np.int64)

            model = ActionLogisticRegression(
                action=action,
                c_regularization=self.c_regularization,
                random_state=self.random_state,
            )
            model.fit(X_action, y_action, feature_names=preprocessor.feature_names)
            action_models[action] = model

        metadata: Dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_training_payments": len(train_records),
            "n_training_rows": len(dataset),
            "c_regularization": self.c_regularization,
            "random_state": self.random_state,
            "feature_names": preprocessor.feature_names,
            "num_features": preprocessor.num_dimensions,
        }

        return TrainedRecoveryProbabilityModel(
            preprocessor=preprocessor,
            action_models=action_models,
            model_version=self.model_version,
            metadata=metadata,
        )
