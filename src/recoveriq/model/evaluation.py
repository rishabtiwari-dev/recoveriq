"""Model evaluation and research diagnostic metrics for probability models.

RESEARCH VALIDITY NOTE:
true_probability is queried strictly within this evaluation harness to measure
ground-truth fidelity. It is NEVER accessed by ModelTrainer, FeaturePreprocessor,
ActionLogisticRegression, or the trained inference model.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from recoveriq.domain.actions import Action
from recoveriq.model.dataset import extract_observable_features
from recoveriq.model.trained_model import TrainedRecoveryProbabilityModel
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.schema import SyntheticPaymentRecord


@dataclass
class ActionEvaluationMetrics:
    """Statistical evaluation metrics for a single recovery action."""

    action: Action
    sample_count: int
    empirical_recovery_rate: float
    mean_predicted_probability: float
    brier_score: float
    log_loss_val: float
    roc_auc: Optional[float]
    mae_vs_ground_truth: float  # Diagnostic comparison against true_probability


@dataclass
class ModelEvaluationReport:
    """Overall evaluation report across all 6 candidate actions."""

    metrics_per_action: Dict[Action, ActionEvaluationMetrics] = field(default_factory=dict)
    overall_mean_brier_score: float = 0.0
    overall_mean_log_loss: float = 0.0
    test_sample_count: int = 0

    def summary_table(self) -> str:
        """Produce a formatted ASCII table of evaluation metrics."""
        lines = [
            f"{'Action':<15} | {'N':<6} | {'Empirical':<10} | {'Mean Pred':<10} | {'Brier':<8} | {'LogLoss':<8} | {'ROC-AUC':<8} | {'MAE vs GT':<10}",
            "-" * 95,
        ]
        for action in Action:
            m = self.metrics_per_action.get(action)
            if not m:
                continue
            auc_str = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "N/A (STOP)"
            lines.append(
                f"{action.value:<15} | {m.sample_count:<6} | {m.empirical_recovery_rate:<10.4f} | "
                f"{m.mean_predicted_probability:<10.4f} | {m.brier_score:<8.4f} | {m.log_loss_val:<8.4f} | "
                f"{auc_str:<8} | {m.mae_vs_ground_truth:<10.4f}"
            )
        return "\n".join(lines)


class ModelEvaluator:
    """Evaluates a TrainedRecoveryProbabilityModel on held-out test data."""

    def evaluate(
        self,
        model: TrainedRecoveryProbabilityModel,
        test_records: List[SyntheticPaymentRecord],
        test_env: SimulationEnvironment,
    ) -> ModelEvaluationReport:
        """Evaluate model performance across all 6 candidate actions on test records."""
        if not test_records:
            raise ValueError("Cannot evaluate model on empty test records.")

        # Extract features for all test records
        feat_dicts = [extract_observable_features(r) for r in test_records]
        X_test = model.preprocessor.transform(feat_dicts)

        metrics_map: Dict[Action, ActionEvaluationMetrics] = {}
        brier_scores: List[float] = []
        log_losses: List[float] = []

        for action in Action:
            # 1. Obtain test outcomes and diagnostic true probabilities from test environment
            outcomes = test_env.batch_apply_action([r.payment_id for r in test_records], action)
            y_true = np.array([int(o.recovered) for o in outcomes], dtype=np.int64)
            y_gt_prob = np.array([o.true_probability for o in outcomes], dtype=np.float64)

            # 2. Compute model predictions
            action_model = model.action_models[action]
            y_pred = action_model.predict_proba(X_test)

            # Clamp predictions slightly for numerical stability in log loss
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

            # 3. Calculate metrics
            n = len(y_true)
            emp_rate = float(np.mean(y_true))
            mean_pred = float(np.mean(y_pred))
            brier = float(brier_score_loss(y_true, y_pred))

            if action == Action.STOP:
                # STOP has all-zero labels and all-zero predictions
                ll = 0.0
                auc = None
            else:
                ll = float(log_loss(y_true, y_pred_clipped, labels=[0, 1]))
                try:
                    auc = float(roc_auc_score(y_true, y_pred))
                except ValueError:
                    auc = None

            # Diagnostic comparison: MAE between predicted probability and true world probability
            mae_gt = float(np.mean(np.abs(y_pred - y_gt_prob)))

            metrics_map[action] = ActionEvaluationMetrics(
                action=action,
                sample_count=n,
                empirical_recovery_rate=emp_rate,
                mean_predicted_probability=mean_pred,
                brier_score=brier,
                log_loss_val=ll,
                roc_auc=auc,
                mae_vs_ground_truth=mae_gt,
            )

            brier_scores.append(brier)
            log_losses.append(ll)

        return ModelEvaluationReport(
            metrics_per_action=metrics_map,
            overall_mean_brier_score=float(np.mean(brier_scores)),
            overall_mean_log_loss=float(np.mean(log_losses)),
            test_sample_count=len(test_records),
        )
