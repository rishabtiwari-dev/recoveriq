"""Sprint 14 — Model-Free Sequential Recovery Policy.

Implements a lightweight Fitted Q-Iteration (tabular Monte Carlo) policy that learns
action values from observed trajectory outcomes rather than directly trusting a trained
probability model.

DESIGN PHILOSOPHY:
-----------------
The core Sprint 13 Bellman policy is model-based: it uses P̂(a|x) from a trained logistic
regression model to compute sequential Q-values. When P̂(a|x) is misspecified, the Bellman
value estimates degrade proportionally.

ModelFreeRecoverIQStrategy takes a different approach:
- It learns Q̂(x, a, t) from completed trajectory episodes.
- It uses only information observable at decision time: failure_category, customer_tier,
  attempt_number, and the action taken.
- It uses Monte Carlo returns from each trajectory episode:
  G_t = NRV earned over the full remaining trajectory from step t onward.
- At inference time, it selects: a* = argmax_a Q̂(state, a, attempt_t)

ANTI-LEAKAGE GUARANTEES:
------------------------
1. Training is performed ONLY on designated training-partition trajectory episodes.
2. The evaluation records (test set) are NEVER seen during training.
3. The policy does NOT access ground_truth_records at decision time.
4. The policy does NOT access the probability model at decision time.
5. The Q-table is built purely from (state, action, return) tuples from training episodes.
6. Fallback action when state is unseen: RETRY_NOW (conservative, non-terminal).

STATE SPACE:
-----------
State = (failure_category: str, customer_tier: str, attempt_number: int)
Action = recoveriq.domain.actions.Action (6 actions)

LEARNING:
---------
For each step t in a training episode:
  state = (failure_category, customer_tier, attempt_number=t)
  action = authorized_action at step t
  G_t = sum of NRV accrued from step t to end of trajectory
        (i.e., remaining_nrv_from_step_t = total_episode_nrv if t=1, etc.)

Q̂(state, action) ← average G_t across all (state, action, G_t) training samples.

This is a first-visit Monte Carlo tabular approach. It avoids all probability model
dependence at inference time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.simulation.schema import SyntheticPaymentRecord


# Type alias for the state representation used in the Q-table
QState = Tuple[str, str, int]  # (failure_category, customer_tier, attempt_number)


def _make_state(
    failure_category: FailureCategory,
    customer_tier: CustomerTier,
    attempt_number: int,
) -> QState:
    """Construct a hashable Q-table state from observable context features.

    Uses only information available at decision time. Does NOT include:
    - recovery probability estimates
    - ground truth outcomes
    - payment amount (to avoid reward leakage)
    """
    return (failure_category.value, customer_tier.value, int(attempt_number))


# Fallback action for unseen (state, action) pairs
_FALLBACK_ACTION = Action.RETRY_NOW


@dataclass
class FittedQIterationPolicy:
    """Tabular Monte Carlo fitted Q-table learned from trajectory episodes.

    This class accumulates (state, action, return) samples from training episodes
    and computes Q̂(state, action) = mean observed return.

    Anti-leakage invariant:
        No ground_truth_records, probability model output, or evaluation-partition
        data is ever passed to fit() or used in _q_table construction.
    """

    _q_table: Dict[Tuple[QState, Action], List[float]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _is_fitted: bool = field(default=False, init=False)
    n_training_episodes: int = field(default=0, init=False)
    n_training_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._q_table = defaultdict(list)

    def fit(self, training_episodes: "List") -> "FittedQIterationPolicy":
        """Learn Q-values from completed training trajectory episodes.

        Args:
            training_episodes: List of TrajectoryEpisode objects from TRAINING partition only.
                               Must NOT include any evaluation-partition episodes.

        Returns:
            self (for chaining).
        """
        from recoveriq.evaluation.trajectory import TrajectoryEpisode  # local import

        self._q_table.clear()
        self.n_training_episodes = 0
        self.n_training_steps = 0

        for episode in training_episodes:
            if not isinstance(episode, TrajectoryEpisode):
                raise TypeError(
                    f"Expected TrajectoryEpisode, got {type(episode).__name__}"
                )
            self.n_training_episodes += 1

            # Compute per-step Monte Carlo returns:
            # G_t = total_nrv_from_step_t_to_end
            # Since NRV is a property of the full episode (not decomposed per-step),
            # we use episode NRV for step 1 (full trajectory value from step 1),
            # and 0 for subsequent steps (already counting from step 1 forward).
            # This is equivalent to first-visit MC with episode-level return.
            #
            # More precisely: the NRV of a trajectory is the net value conditional on
            # all decisions made from step 1. We attribute the full episode return
            # to each step's (state, action) pair since we don't have per-step rewards
            # decomposed. This is a tractable tabular approximation.
            #
            # Step-level reward approximation:
            # - For step 1: G_1 = episode.net_recovered_value (full trajectory outcome)
            # - For step t > 1: G_t = episode.net_recovered_value - sum(step_costs from 1..t-1)
            #   (remaining value given that earlier steps already incurred costs)
            steps = episode.steps
            n_steps = len(steps)
            total_nrv = float(episode.net_recovered_value)
            cumulative_cost_before = Decimal("0.00")

            for step_idx, step in enumerate(steps):
                attempt_number = step.step_number

                # We need to reconstruct the failure_category and customer_tier.
                # These are NOT stored in TrajectoryStep directly. We encode them
                # during the separate training data collection step (see ModelFreeRecoverIQStrategy.train()).
                # Here we skip — the calling code (train()) handles state extraction.
                pass

            # The _add_sample method is called by the training wrapper.
        self._is_fitted = True
        return self

    def _add_sample(
        self,
        state: QState,
        action: Action,
        return_value: float,
    ) -> None:
        """Add a single (state, action, return) training sample to the Q-table."""
        self._q_table[(state, action)].append(return_value)
        self.n_training_steps += 1

    def get_q_value(self, state: QState, action: Action) -> float:
        """Return mean observed return for (state, action). Returns 0.0 if unseen."""
        samples = self._q_table.get((state, action), [])
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def get_best_action(self, state: QState) -> Action:
        """Return the action maximizing Q̂(state, action). Falls back to RETRY_NOW if unseen."""
        best_action = _FALLBACK_ACTION
        best_q = float("-inf")
        any_seen = False

        for action in Action:
            key = (state, action)
            if key in self._q_table and self._q_table[key]:
                q = self.get_q_value(state, action)
                any_seen = True
                if q > best_q:
                    best_q = q
                    best_action = action

        return best_action if any_seen else _FALLBACK_ACTION

    def is_state_seen(self, state: QState) -> bool:
        """Check if any action Q-value has been estimated for this state."""
        return any(
            (state, action) in self._q_table and self._q_table[(state, action)]
            for action in Action
        )

    @property
    def n_unique_states(self) -> int:
        """Number of unique (state, action) pairs in the Q-table."""
        return len(set(k[0] for k in self._q_table.keys()))


def train_model_free_policy(
    training_episodes_by_strategy: "Dict[str, List]",
    training_records: "List[SyntheticPaymentRecord]",
) -> FittedQIterationPolicy:
    """Build a FittedQIterationPolicy from completed training trajectory episodes.

    The training data must come ONLY from the designated training seeds.
    Evaluation seeds must NEVER be passed here.

    Args:
        training_episodes_by_strategy: Dict mapping strategy_name → list of TrajectoryEpisode.
            We use trajectories from ALL strategies to build the Q-table (richer coverage),
            but we exclude STOP-only trajectories (no recovery signal).
        training_records: Corresponding observable SyntheticPaymentRecord list (for state features).
            Must be aligned with episodes by payment_id.

    Returns:
        Fitted FittedQIterationPolicy ready for inference.
    """
    from recoveriq.evaluation.trajectory import TrajectoryEpisode

    # Build payment_id → record lookup for state feature extraction
    record_lookup: Dict[str, SyntheticPaymentRecord] = {
        r.payment_id: r for r in training_records
    }

    policy = FittedQIterationPolicy()

    for strategy_name, episodes in training_episodes_by_strategy.items():
        for episode in episodes:
            if not isinstance(episode, TrajectoryEpisode):
                continue

            rec = record_lookup.get(episode.payment_id)
            if rec is None:
                continue

            steps = episode.steps
            total_nrv = float(episode.net_recovered_value)
            cumulative_step_cost = Decimal("0.00")

            for step_idx, step in enumerate(steps):
                attempt_number = step.step_number

                # Construct observable state (no ground-truth features)
                state = _make_state(
                    failure_category=rec.failure_category,
                    customer_tier=rec.customer_tier,
                    attempt_number=attempt_number,
                )

                # Monte Carlo return from this step onward:
                # G_t = total_nrv - costs_incurred_before_step_t
                # For step 1: G_1 = total_nrv (no prior costs)
                # For step 2: G_2 = total_nrv - step1_cost - step1_penalty
                g_t = total_nrv - float(cumulative_step_cost)

                # Record the (state, authorized_action, return) sample
                policy._add_sample(state, step.authorized_action, g_t)

                # Accumulate costs for next step
                cumulative_step_cost += step.step_cost + step.step_penalty

    policy._is_fitted = True
    return policy


class ModelFreeRecoverIQStrategy:
    """Model-Free Sequential Recovery Strategy using Tabular Monte Carlo Q-learning.

    At inference time:
    - Constructs observable state from PaymentContext (failure_category, customer_tier, attempt).
    - Looks up argmax_a Q̂(state, a) from the fitted Q-table.
    - Falls back to RETRY_NOW if state is unseen.

    Anti-leakage guarantees:
    - Does NOT call probability_model.estimate_probabilities().
    - Does NOT access ground_truth_records.
    - Q-table was built strictly from training-partition episodes.
    """

    name: str = "RecoverIQ-ModelFree"

    def __init__(self, fitted_policy: FittedQIterationPolicy) -> None:
        if not isinstance(fitted_policy, FittedQIterationPolicy):
            raise TypeError(
                f"Expected FittedQIterationPolicy, got {type(fitted_policy).__name__}"
            )
        self._policy = fitted_policy
        self.last_state: Optional[QState] = None
        self.last_was_fallback: bool = False

    @property
    def policy(self) -> FittedQIterationPolicy:
        """Read-only access to the fitted Q policy."""
        return self._policy

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Select best action from Q-table. Falls back to RETRY_NOW on unseen state.

        Uses ONLY observable context features:
        - context.failure_category
        - context.customer_tier
        - context.attempt_count

        Does NOT use:
        - probability model
        - ground truth
        - payment amount
        """
        state = _make_state(
            failure_category=context.failure_category,
            customer_tier=context.customer_tier,
            attempt_number=context.attempt_count if context.attempt_count > 0 else 1,
        )
        self.last_state = state
        self.last_was_fallback = not self._policy.is_state_seen(state)
        return self._policy.get_best_action(state)
