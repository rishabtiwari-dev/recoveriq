"""Counterfactual dataset generation for statistical recovery probability modeling.

CRITICAL RESEARCH VALIDITY & ANTI-LEAKAGE INVARIANTS:
1. Observable features only: Training records are built strictly from observable
   decision-time attributes (SyntheticPaymentRecord or PaymentContext).
2. Binary outcome target: The target label y is strictly int(action_outcome.recovered) in {0, 1}.
3. Hidden ground-truth fields (true_probability, latent_recoverability_profile,
   action_base_probabilities) must NEVER be included as features, targets, or derived values.
4. Full factorial counterfactual structure: For each payment, all 6 candidate actions
   are evaluated against the SimulationEnvironment, sharing identical underlying observable features.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from recoveriq.domain.actions import Action
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.schema import SyntheticPaymentRecord


@dataclass(frozen=True)
class TrainingExample:
    """A single labeled training observation (x, a, y)."""

    payment_id: str
    action: Action
    features: Dict[str, Any]
    label: int  # 0 or 1 (strictly binary recovered outcome)


@dataclass
class CounterfactualDataset:
    """Container for an action-conditioned counterfactual dataset."""

    examples: List[TrainingExample] = field(default_factory=list)
    seed: int = 42

    def __len__(self) -> int:
        return len(self.examples)

    def get_action_examples(self, action: Action) -> List[TrainingExample]:
        """Return all training examples for a specific action."""
        return [ex for ex in self.examples if ex.action == action]

    def get_feature_matrix_and_labels(
        self, action: Optional[Action] = None
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Extract feature dictionaries and binary labels, optionally filtered by action."""
        subset = self.get_action_examples(action) if action is not None else self.examples
        x_list = [ex.features for ex in subset]
        y_list = [ex.label for ex in subset]
        return x_list, y_list


def extract_observable_features(record: SyntheticPaymentRecord) -> Dict[str, Any]:
    """Extract strictly observable decision-time features from a payment record.

    NOTE: No hidden simulation variables (latent_recoverability_profile,
    true_probability, action_base_probabilities) are accessed or returned.
    """
    return {
        "failure_category": record.failure_category.value,
        "failure_severity": record.failure_severity.value,
        "customer_tier": record.customer_tier.value,
        "payment_method": record.payment_method.value,
        "amount": float(record.amount),
        "attempt_count": int(record.attempt_count),
        "failure_hour": record.failure_timestamp.hour if record.failure_timestamp else 0,
        "failure_day_of_week": record.failure_timestamp.weekday() if record.failure_timestamp else 0,
    }


class CounterfactualDatasetBuilder:
    """Builds a full factorial counterfactual dataset from observable records and environment."""

    def __init__(self, actions: Optional[List[Action]] = None):
        self.actions = actions or list(Action)

    def build_dataset(
        self,
        observable_records: List[SyntheticPaymentRecord],
        env: SimulationEnvironment,
        seed: int = 42,
    ) -> CounterfactualDataset:
        """Construct an action-conditioned dataset for all observable payments.

        For each payment i:
          Evaluate all candidate actions a in A against env.
          Record (x_i, a, y_{i,a}) where y_{i,a} = int(outcome.recovered).
        """
        examples: List[TrainingExample] = []

        for record in observable_records:
            # Extract observable features once per payment
            x = extract_observable_features(record)

            for action in self.actions:
                # Resolve outcome via simulation environment
                outcome = env.apply_action(record.payment_id, action)

                # Strictly target binary outcome (0 or 1)
                # CRITICAL: outcome.true_probability is explicitly ignored
                y = int(outcome.recovered)

                examples.append(
                    TrainingExample(
                        payment_id=record.payment_id,
                        action=action,
                        features=dict(x),
                        label=y,
                    )
                )

        return CounterfactualDataset(examples=examples, seed=seed)
