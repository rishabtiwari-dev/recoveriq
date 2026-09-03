"""Simulation environment — action-conditioned outcome resolver.

The SimulationEnvironment is the only component that holds a reference to
GroundTruthRecord data. It acts as the synthetic "world" that responds
to recovery actions by resolving stochastic outcomes.

DESIGN INVARIANT:
- The environment takes a pre-built GroundTruthRecord index at construction.
- The environment NEVER exposes latent_recoverability_profile or
  action_base_probabilities to callers; it only returns outcome booleans.
- The environment uses its own seeded RNG, separate from the generation RNG,
  so that outcome resolution does not interfere with generation reproducibility.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from recoveriq.domain.actions import Action
from recoveriq.simulation.ground_truth import resolve_outcome
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


@dataclass(frozen=True)
class ActionOutcome:
    """Result of applying a recovery action in the simulation environment."""

    payment_id: str
    action: Action
    recovered: bool
    # Ground-truth probability used (for analysis / evaluation — NOT for training)
    true_probability: float


@dataclass
class SimulationEnvironment:
    """Stochastic simulation world for evaluating recovery strategies.

    Usage:
        env = SimulationEnvironment(ground_truth_records, seed=42)
        outcome = env.apply_action(payment_id="pay_42_000001", action=Action.RETRY_NOW)
    """

    _ground_truth_index: Dict[str, GroundTruthRecord] = field(
        default_factory=dict, repr=False
    )
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    seed: int = 42

    def __init__(
        self,
        ground_truth_records: List[GroundTruthRecord],
        seed: int = 42,
    ) -> None:
        self._ground_truth_index = {gt.payment_id: gt for gt in ground_truth_records}
        self._rng = random.Random(seed)
        self.seed = seed

    def apply_action(
        self,
        payment_id: str,
        action: Action,
    ) -> ActionOutcome:
        """Apply a recovery action and resolve stochastic outcome.

        The caller passes observable payment_id and action.
        The environment resolves the outcome using hidden ground truth.

        Args:
            payment_id: Identifier of the payment being recovered.
            action: The recovery action applied.

        Returns:
            ActionOutcome with recovered=True/False.

        Raises:
            KeyError: If payment_id is not in the environment's ground-truth index.
        """
        gt = self._ground_truth_index.get(payment_id)
        if gt is None:
            raise KeyError(
                f"Payment '{payment_id}' not found in simulation environment. "
                "Ensure ground truth was registered at environment construction time."
            )

        recovered = resolve_outcome(gt, action, self._rng)
        true_p = gt.action_base_probabilities.get(action, 0.0)

        return ActionOutcome(
            payment_id=payment_id,
            action=action,
            recovered=recovered,
            true_probability=true_p,
        )

    def batch_apply_action(
        self,
        payment_ids: List[str],
        action: Action,
    ) -> List[ActionOutcome]:
        """Apply the same action to a batch of payments."""
        return [self.apply_action(pid, action) for pid in payment_ids]

    @property
    def n_payments(self) -> int:
        """Number of payments registered in this environment."""
        return len(self._ground_truth_index)
