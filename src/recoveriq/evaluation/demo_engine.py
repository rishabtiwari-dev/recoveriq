"""Sprint 16 — Demo Adapter and Engine.

Provides clean, anti-leakage helper functions for app/demo.py:
- Single-payment generation and observable context creation
- Strategy suite instantiation
- Step-by-step and full-trajectory execution under CRN
- Action comparison across all paradigms on the exact same payment
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.bellman_policy import BellmanRecoverIQStrategy
from recoveriq.evaluation.hybrid_policy import (
    HybridActionEvaluation,
    HybridRecoverIQStrategy,
    HybridRegime,
)
from recoveriq.evaluation.model_free_policy import (
    FittedQIterationPolicy,
    ModelFreeRecoverIQStrategy,
    train_model_free_policy,
)
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import (
    FixedRetryStrategy,
    RecoverIQStrategy,
    RuleBasedStrategy,
)
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    TrajectoryEpisode,
    TrajectoryEvaluationRunner,
    TrajectoryStep,
)
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


class DemoEngine:
    """Manages trained models, policies, and interactive simulations for the demo."""

    def __init__(self, seed: int = 42, max_attempts: int = 3) -> None:
        self.seed = seed
        self.max_attempts = max_attempts
        self.runner = TrajectoryEvaluationRunner(
            max_attempts=max_attempts, scheduled_cooldown_seconds=900
        )
        self._is_initialized = False

        self.trained_model: Optional[Any] = None
        self.shared_mf_policy: Optional[FittedQIterationPolicy] = None
        self.strategies: Dict[str, Any] = {}
        self.demo_records: List[SyntheticPaymentRecord] = []
        self.demo_ground_truth: List[GroundTruthRecord] = []

    def initialize(self) -> None:
        """Initialize simulation environment, train base model, and fit ModelFree Q-table."""
        if self._is_initialized:
            return

        cfg = SimulationConfig(n_payments=600, n_customers=150, train_fraction=0.75)
        gen = SyntheticPaymentGenerator(cfg)
        dataset = gen.generate(seed=self.seed)
        part = partition_dataset(dataset, train_fraction=cfg.train_fraction)

        # 1. Train base probability model on observable training set
        train_env = SimulationEnvironment(part.train_ground_truth, seed=self.seed)
        trainer = ModelTrainer(c_regularization=1.0, random_state=self.seed)
        self.trained_model = trainer.train(part.train_observable, train_env)

        # 2. Train shared ModelFree Q policy on training episodes
        train_episodes = []
        for strat in [
            FixedRetryStrategy(),
            RuleBasedStrategy(),
            RecoverIQStrategy(probability_model=self.trained_model),
        ]:
            env = SimulationEnvironment(part.train_ground_truth, seed=self.seed)
            eps = [self.runner.evaluate_episode(r, strat, env) for r in part.train_observable]
            train_episodes.extend(eps)

        self.shared_mf_policy = train_model_free_policy(
            training_episodes_by_strategy={"train": train_episodes},
            training_records=part.train_observable,
        )

        # 3. Instantiate strategies
        mf_strat = ModelFreeRecoverIQStrategy(fitted_policy=self.shared_mf_policy)
        mf_strat.name = "RecoverIQ-ModelFree"

        bellman_strat = BellmanRecoverIQStrategy(
            probability_model=self.trained_model,
            max_attempts=self.max_attempts,
            planning_horizon=self.max_attempts,
        )
        bellman_strat.name = "RecoverIQ-Bellman"

        tiered_strat = TieredRecoverIQStrategy(
            probability_model=self.trained_model, max_attempts=self.max_attempts
        )
        tiered_strat.name = "RecoverIQ-Tiered"

        hybrid_unc = HybridRecoverIQStrategy(
            bellman_strategy=bellman_strat,
            modelfree_strategy=mf_strat,
            regime=HybridRegime.UNCERTAINTY_AWARE,
        )
        hybrid_unc.name = "RecoverIQ-Hybrid-Uncertainty"

        hybrid_eq = HybridRecoverIQStrategy(
            bellman_strategy=bellman_strat,
            modelfree_strategy=mf_strat,
            regime=HybridRegime.EQUAL_WEIGHT,
        )
        hybrid_eq.name = "RecoverIQ-Hybrid-Equal"

        hybrid_fx = HybridRecoverIQStrategy(
            bellman_strategy=bellman_strat,
            modelfree_strategy=mf_strat,
            regime=HybridRegime.FIXED_WEIGHT,
            fixed_bellman_weight=0.70,
        )
        hybrid_fx.name = "RecoverIQ-Hybrid-Fixed"

        self.strategies = {
            "Rule-Based": RuleBasedStrategy(),
            "Fixed-Retry": FixedRetryStrategy(),
            "Always-Stop": AlwaysStopStrategy(),
            "RecoverIQ-Tiered": tiered_strat,
            "RecoverIQ-Bellman": bellman_strat,
            "RecoverIQ-ModelFree": mf_strat,
            "RecoverIQ-Hybrid-Uncertainty": hybrid_unc,
            "RecoverIQ-Hybrid-Equal": hybrid_eq,
            "RecoverIQ-Hybrid-Fixed": hybrid_fx,
        }

        self.demo_records = part.test_observable
        self.demo_ground_truth = part.test_ground_truth
        self._is_initialized = True

    def get_sample_payment(self, index: int = 0) -> SyntheticPaymentRecord:
        """Return an observable payment record from the held-out test set."""
        self.initialize()
        idx = max(0, min(index, len(self.demo_records) - 1))
        return self.demo_records[idx]

    def create_custom_payment(
        self,
        payment_id: str,
        amount: float,
        failure_category: FailureCategory,
        customer_tier: CustomerTier,
    ) -> SyntheticPaymentRecord:
        """Create a purely synthetic observable payment record with user-chosen context."""
        self.initialize()
        base = self.demo_records[0]
        import dataclasses
        return dataclasses.replace(
            base,
            payment_id=payment_id,
            amount=Decimal(str(round(amount, 2))),
            failure_category=failure_category,
            customer_tier=customer_tier,
        )

    def evaluate_action_at_step(
        self,
        strategy_name: str,
        record: SyntheticPaymentRecord,
        attempt: int = 1,
    ) -> Action:
        """Evaluate which action a strategy proposes for an observable payment at attempt t."""
        self.initialize()
        strategy = self.strategies[strategy_name]
        ctx = PaymentContext(
            payment_id=record.payment_id,
            customer_id=record.customer_id,
            customer_tier=record.customer_tier,
            payment_method=record.payment_method,
            raw_error_code=record.raw_error_code,
            raw_error_message=record.raw_error_message,
            failure_category=record.failure_category,
            failure_severity=record.failure_severity,
            attempt_count=attempt,
            extra_metadata={"amount": float(record.amount)},
        )
        return strategy.propose_action(record, ctx)

    def run_full_trajectory(
        self,
        strategy_name: str,
        record: SyntheticPaymentRecord,
        seed: Optional[int] = None,
    ) -> TrajectoryEpisode:
        """Execute a full multi-step sequential recovery trajectory under CRN."""
        self.initialize()
        strategy = self.strategies[strategy_name]
        sim_seed = seed if seed is not None else self.seed
        env = SimulationEnvironment(self.demo_ground_truth, seed=sim_seed)
        return self.runner.evaluate_episode(record, strategy, env)

    def compare_decisions_for_payment(
        self,
        record: SyntheticPaymentRecord,
        attempt: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare decision outputs across all main paradigms on the same payment context."""
        self.initialize()
        results: Dict[str, Dict[str, Any]] = {}

        ctx = PaymentContext(
            payment_id=record.payment_id,
            customer_id=record.customer_id,
            customer_tier=record.customer_tier,
            payment_method=record.payment_method,
            raw_error_code=record.raw_error_code,
            raw_error_message=record.raw_error_message,
            failure_category=record.failure_category,
            failure_severity=record.failure_severity,
            attempt_count=attempt,
            extra_metadata={"amount": float(record.amount)},
        )

        for name, strat in self.strategies.items():
            action = strat.propose_action(record, ctx)
            details: Dict[str, Any] = {"action": action.value}

            # If Bellman or Hybrid, attach extra evaluation metadata
            if name == "RecoverIQ-Bellman" and hasattr(strat, "last_decision") and strat.last_decision:
                ev = strat.last_decision.selected_evaluation
                details["q_value"] = float(ev.total_q_value)
                details["immediate_ev"] = float(ev.immediate_ev)
                details["future_option_value"] = float(ev.future_option_value)
                details["probability"] = float(ev.probability)

            elif "Hybrid" in name and hasattr(strat, "last_decision") and strat.last_decision:
                ev = strat.last_decision.selected_evaluation
                details["q_hybrid"] = float(ev.q_hybrid)
                details["weight_bellman"] = float(ev.weight_bellman)
                details["weight_modelfree"] = float(ev.weight_modelfree)

            results[name] = details

        return results
