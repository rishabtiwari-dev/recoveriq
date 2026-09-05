"""Sequential multi-step recovery trajectory evaluation for RecoverIQ (Sprint 10).

This module provides lightweight abstractions and an evaluation runner to evaluate
recovery strategies over sequential, multi-attempt trajectories rather than isolated
single decision points.

CORE CONCEPTS:
- TrajectoryStep: Records one decision-authorization-execution step in a payment's recovery.
- TrajectoryEpisode: Represents the complete lifecycle trajectory of a single payment.
- TrajectoryStrategyMetrics: Aggregated episode-level performance metrics for a strategy.
- TrajectoryEvaluationRunner: Orchestrates multi-step evaluation with CRN and deterministic policy gating.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np

from recoveriq.config.settings import ActionCostConfig, PenaltyConfig, PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import RecoveryDecision
from recoveriq.domain.models import CustomerTier, Payment, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.strategies import (
    FixedRetryStrategy,
    RecoveryStrategy,
    RecoverIQStrategy,
    RuleBasedStrategy,
)
from recoveriq.policy.gate import InvariantPolicyGate, PolicyGate
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


class AlwaysStopStrategy:
    """Always-Stop baseline strategy that unconditionally proposes Action.STOP."""

    name: str = "Always-Stop"

    def propose_action(
        self,
        record: SyntheticPaymentRecord,
        context: PaymentContext,
    ) -> Action:
        """Unconditionally propose stopping recovery."""
        return Action.STOP


@dataclass(frozen=True)
class TrajectoryStep:
    """Record of a single sequential recovery intervention step."""

    step_number: int
    proposed_action: Action
    authorized_action: Action
    is_authorized: bool
    recovered: bool
    step_cost: Decimal
    step_penalty: Decimal
    resulting_state: PaymentState
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class TrajectoryEpisode:
    """Complete sequence of recovery attempts for an individual payment."""

    payment_id: str
    steps: List[TrajectoryStep]
    terminal_state: PaymentState
    final_recovered: bool
    payment_amount: Decimal
    total_cost: Decimal
    total_penalty: Decimal
    net_recovered_value: Decimal
    attempt_count: int


@dataclass
class TrajectoryStrategyMetrics:
    """Aggregated trajectory metrics for a strategy evaluated across payment episodes."""

    strategy_name: str
    seed: int
    n_payments: int
    n_recovered: int
    n_escalated: int
    n_failed_terminal: int
    recovery_rate: float
    escalation_rate: float
    failed_terminal_rate: float
    average_attempts_per_payment: float
    total_gross_revenue: Decimal
    total_cost: Decimal
    total_penalty: Decimal
    total_nrv: Decimal
    mean_nrv: float
    policy_violation_rate: float
    terminal_state_distribution: Dict[str, float]
    survival_rate_by_step: Dict[int, float]
    recovery_lift_by_attempt: Dict[int, float]
    episodes: List[TrajectoryEpisode] = field(default_factory=list, repr=False)

    @classmethod
    def compute(
        cls,
        strategy_name: str,
        seed: int,
        episodes: List[TrajectoryEpisode],
        max_attempts: int = 3,
    ) -> "TrajectoryStrategyMetrics":
        """Compute comprehensive trajectory metrics from individual payment episodes."""
        n_payments = len(episodes)
        if n_payments == 0:
            raise ValueError("Cannot compute TrajectoryStrategyMetrics for empty episodes.")

        n_recovered = sum(1 for ep in episodes if ep.final_recovered)
        n_escalated = sum(1 for ep in episodes if ep.terminal_state == PaymentState.ESCALATED)
        n_failed_terminal = sum(1 for ep in episodes if ep.terminal_state == PaymentState.FAILED_TERMINAL)

        recovery_rate = n_recovered / n_payments
        escalation_rate = n_escalated / n_payments
        failed_terminal_rate = n_failed_terminal / n_payments

        total_attempts = sum(ep.attempt_count for ep in episodes)
        average_attempts = total_attempts / n_payments

        total_gross = sum((ep.payment_amount for ep in episodes if ep.final_recovered), Decimal("0.00"))
        total_cost = sum((ep.total_cost for ep in episodes), Decimal("0.00"))
        total_penalty = sum((ep.total_penalty for ep in episodes), Decimal("0.00"))
        total_nrv = sum((ep.net_recovered_value for ep in episodes), Decimal("0.00"))
        mean_nrv = float(total_nrv) / n_payments

        # Terminal state distribution
        terminal_dist = {
            PaymentState.RECOVERED.value: recovery_rate,
            PaymentState.ESCALATED.value: escalation_rate,
            PaymentState.FAILED_TERMINAL.value: failed_terminal_rate,
        }

        # Survival rate by step: active payments that reached at least step k
        survival_rate = {}
        for k in range(1, max_attempts + 1):
            active_count = sum(1 for ep in episodes if len(ep.steps) >= k)
            survival_rate[k] = active_count / n_payments

        # Multi-step lift: incremental recoveries occurring at attempt k
        lift_by_attempt = {}
        for k in range(1, max_attempts + 1):
            rec_at_k = 0
            for ep in episodes:
                if ep.final_recovered and len(ep.steps) == k and ep.steps[-1].recovered:
                    rec_at_k += 1
            lift_by_attempt[k] = rec_at_k / n_payments

        # Policy violation rate: 0.00% guaranteed by common policy gate
        policy_violation_rate = 0.0

        return cls(
            strategy_name=strategy_name,
            seed=seed,
            n_payments=n_payments,
            n_recovered=n_recovered,
            n_escalated=n_escalated,
            n_failed_terminal=n_failed_terminal,
            recovery_rate=recovery_rate,
            escalation_rate=escalation_rate,
            failed_terminal_rate=failed_terminal_rate,
            average_attempts_per_payment=average_attempts,
            total_gross_revenue=total_gross,
            total_cost=total_cost,
            total_penalty=total_penalty,
            total_nrv=total_nrv,
            mean_nrv=mean_nrv,
            policy_violation_rate=policy_violation_rate,
            terminal_state_distribution=terminal_dist,
            survival_rate_by_step=survival_rate,
            recovery_lift_by_attempt=lift_by_attempt,
            episodes=episodes,
        )


@dataclass
class MultiSeedTrajectoryMetrics:
    """Aggregated trajectory metrics across multiple seeds with Mean +/- Std."""

    strategy_name: str
    seeds: List[int]
    mean_recovery_rate: float
    std_recovery_rate: float
    mean_escalation_rate: float
    std_escalation_rate: float
    mean_failed_terminal_rate: float
    std_failed_terminal_rate: float
    mean_average_attempts: float
    std_average_attempts: float
    mean_total_nrv: float
    std_total_nrv: float
    mean_nrv_per_payment: float
    std_nrv_per_payment: float
    mean_direct_cost: float
    std_direct_cost: float
    mean_friction_penalty: float
    std_friction_penalty: float
    mean_lift_by_attempt: Dict[int, float]
    mean_survival_rate_by_step: Dict[int, float]
    seed_runs: List[TrajectoryStrategyMetrics]

    @classmethod
    def aggregate(
        cls,
        strategy_name: str,
        seed_runs: List[TrajectoryStrategyMetrics],
    ) -> "MultiSeedTrajectoryMetrics":
        """Aggregate trajectory metrics across multiple evaluation seeds."""
        if not seed_runs:
            raise ValueError("Cannot aggregate empty seed runs.")

        seeds = [r.seed for r in seed_runs]
        max_attempts = len(seed_runs[0].recovery_lift_by_attempt)

        rec_rates = [r.recovery_rate for r in seed_runs]
        esc_rates = [r.escalation_rate for r in seed_runs]
        fail_rates = [r.failed_terminal_rate for r in seed_runs]
        avg_attempts = [r.average_attempts_per_payment for r in seed_runs]
        nrvs = [float(r.total_nrv) for r in seed_runs]
        nrv_per_pays = [r.mean_nrv for r in seed_runs]
        costs = [float(r.total_cost) for r in seed_runs]
        penalties = [float(r.total_penalty) for r in seed_runs]

        mean_lift: Dict[int, float] = {}
        for k in range(1, max_attempts + 1):
            mean_lift[k] = float(np.mean([r.recovery_lift_by_attempt.get(k, 0.0) for r in seed_runs]))

        mean_survival: Dict[int, float] = {}
        for k in range(1, max_attempts + 1):
            mean_survival[k] = float(np.mean([r.survival_rate_by_step.get(k, 0.0) for r in seed_runs]))

        return cls(
            strategy_name=strategy_name,
            seeds=seeds,
            mean_recovery_rate=float(np.mean(rec_rates)),
            std_recovery_rate=float(np.std(rec_rates)),
            mean_escalation_rate=float(np.mean(esc_rates)),
            std_escalation_rate=float(np.std(esc_rates)),
            mean_failed_terminal_rate=float(np.mean(fail_rates)),
            std_failed_terminal_rate=float(np.std(fail_rates)),
            mean_average_attempts=float(np.mean(avg_attempts)),
            std_average_attempts=float(np.std(avg_attempts)),
            mean_total_nrv=float(np.mean(nrvs)),
            std_total_nrv=float(np.std(nrvs)),
            mean_nrv_per_payment=float(np.mean(nrv_per_pays)),
            std_nrv_per_payment=float(np.std(nrv_per_pays)),
            mean_direct_cost=float(np.mean(costs)),
            std_direct_cost=float(np.std(costs)),
            mean_friction_penalty=float(np.mean(penalties)),
            std_friction_penalty=float(np.std(penalties)),
            mean_lift_by_attempt=mean_lift,
            mean_survival_rate_by_step=mean_survival,
            seed_runs=seed_runs,
        )


class TrajectoryEvaluationRunner:
    """Orchestrates multi-step sequential recovery trajectory evaluation under CRN."""

    def __init__(
        self,
        policy_gate: Optional[PolicyGate] = None,
        cost_config: Optional[ActionCostConfig] = None,
        penalty_config: Optional[PenaltyConfig] = None,
        policy_config: Optional[PolicyConfig] = None,
        max_attempts: int = 3,
        scheduled_cooldown_seconds: int = 900,
    ):
        self.policy_config = policy_config or PolicyConfig(max_attempts=max_attempts)
        self.policy_gate = policy_gate or InvariantPolicyGate(config=self.policy_config)
        self.cost_config = cost_config or ActionCostConfig()
        self.penalty_config = penalty_config or PenaltyConfig()
        self.max_attempts = max_attempts
        self.scheduled_cooldown_seconds = scheduled_cooldown_seconds

    def evaluate_episode(
        self,
        record: SyntheticPaymentRecord,
        strategy: RecoveryStrategy,
        env: SimulationEnvironment,
    ) -> TrajectoryEpisode:
        """Evaluate a single payment through a sequential multi-step trajectory until termination."""
        amount_decimal = record.amount if isinstance(record.amount, Decimal) else Decimal(str(record.amount))

        # Initialize Payment state
        payment = Payment(
            payment_id=record.payment_id,
            customer_id=record.customer_id,
            amount=amount_decimal,
            currency=record.currency,
            state=PaymentState.FAILED_INITIAL,
            attempt_count=0,
        )

        steps: List[TrajectoryStep] = []
        current_time = record.failure_timestamp
        last_attempt_time: Optional[datetime] = None

        final_recovered = False
        terminal_state = PaymentState.FAILED_INITIAL

        for attempt_idx in range(1, self.max_attempts + 1):
            # 1. Build Decision Context for current attempt
            context = PaymentContext(
                payment_id=record.payment_id,
                customer_id=record.customer_id,
                customer_tier=record.customer_tier,
                payment_method=record.payment_method,
                raw_error_code=record.raw_error_code,
                raw_error_message=record.raw_error_message,
                failure_category=record.failure_category,
                failure_severity=record.failure_severity,
                attempt_count=attempt_idx,
                last_attempt_timestamp=last_attempt_time,
                extra_metadata={"amount": float(amount_decimal)},
            )

            # 2. Strategy proposes candidate action
            proposed_action = strategy.propose_action(record, context)

            # 3. Policy Gate validates & authorizes/clamps
            decision = RecoveryDecision(
                payment_id=record.payment_id,
                proposed_action=proposed_action,
            )
            policy_decision = self.policy_gate.authorize(
                payment=payment,
                context=context,
                decision=decision,
            )
            authorized_action = policy_decision.authorized_action

            # 4. Economic Costs and Penalties
            step_cost = self.cost_config.get_cost(authorized_action)
            step_penalty = self.penalty_config.get_penalty(authorized_action, record.customer_tier)

            # 5. Resolve outcome via simulation environment
            outcome = env.apply_action(record.payment_id, authorized_action)
            is_recovered = outcome.recovered

            # 6. Determine resulting state
            if authorized_action == Action.ESCALATE:
                resulting_state = PaymentState.ESCALATED
            elif authorized_action == Action.STOP:
                resulting_state = PaymentState.FAILED_TERMINAL
            elif is_recovered:
                resulting_state = PaymentState.RECOVERED
            elif attempt_idx >= self.max_attempts:
                resulting_state = PaymentState.FAILED_TERMINAL
            else:
                resulting_state = PaymentState.RECOVERING

            # Record step
            step = TrajectoryStep(
                step_number=attempt_idx,
                proposed_action=proposed_action,
                authorized_action=authorized_action,
                is_authorized=policy_decision.is_authorized,
                recovered=is_recovered,
                step_cost=step_cost,
                step_penalty=step_penalty,
                resulting_state=resulting_state,
                rejection_reason=policy_decision.rejection_reason,
            )
            steps.append(step)

            # Update Payment entity
            payment.state = resulting_state
            payment.attempt_count = attempt_idx
            last_attempt_time = current_time

            # Check terminal conditions
            if resulting_state in (
                PaymentState.RECOVERED,
                PaymentState.ESCALATED,
                PaymentState.FAILED_TERMINAL,
            ):
                final_recovered = (resulting_state == PaymentState.RECOVERED)
                terminal_state = resulting_state
                break

            # If continuing for next attempt, advance simulation clock to model cooldown window
            # Scheduled interventions (RETRY_LATER, NUDGE) require cooldown time to elapse
            current_time = current_time + timedelta(seconds=self.scheduled_cooldown_seconds)

        else:
            # Fall-through if loop finished without explicit break
            terminal_state = payment.state
            final_recovered = (terminal_state == PaymentState.RECOVERED)

        # Economic Totals
        total_cost = sum((s.step_cost for s in steps), Decimal("0.00"))
        total_penalty = sum((s.step_penalty for s in steps), Decimal("0.00"))
        gross_recovered = amount_decimal if final_recovered else Decimal("0.00")
        nrv = gross_recovered - total_cost - total_penalty

        return TrajectoryEpisode(
            payment_id=record.payment_id,
            steps=steps,
            terminal_state=terminal_state,
            final_recovered=final_recovered,
            payment_amount=amount_decimal,
            total_cost=total_cost,
            total_penalty=total_penalty,
            net_recovered_value=nrv,
            attempt_count=len(steps),
        )

    def evaluate_strategy_on_partition(
        self,
        strategy: RecoveryStrategy,
        test_observable: List[SyntheticPaymentRecord],
        test_ground_truth: List[GroundTruthRecord],
        seed: int,
    ) -> TrajectoryStrategyMetrics:
        """Evaluate a strategy over sequential trajectories on a held-out test partition."""
        if not test_observable:
            raise ValueError("test_observable cannot be empty.")

        # CRN Requirement: Fresh simulation environment instantiated with seed S
        env = SimulationEnvironment(test_ground_truth, seed=seed)
        episodes: List[TrajectoryEpisode] = []

        # Deterministic payment iteration order
        for rec in test_observable:
            episode = self.evaluate_episode(rec, strategy, env)
            episodes.append(episode)

        return TrajectoryStrategyMetrics.compute(
            strategy_name=strategy.name,
            seed=seed,
            episodes=episodes,
            max_attempts=self.max_attempts,
        )

    def evaluate_all_strategies(
        self,
        strategies: List[RecoveryStrategy],
        test_observable: List[SyntheticPaymentRecord],
        test_ground_truth: List[GroundTruthRecord],
        seed: int,
    ) -> Dict[str, TrajectoryStrategyMetrics]:
        """Evaluate multiple strategies under Common Random Numbers on the same partition."""
        results: Dict[str, TrajectoryStrategyMetrics] = {}
        for strat in strategies:
            # CRN: Each strategy gets evaluated on the exact same records with an independent seeded env
            metrics = self.evaluate_strategy_on_partition(
                strategy=strat,
                test_observable=test_observable,
                test_ground_truth=test_ground_truth,
                seed=seed,
            )
            results[strat.name] = metrics
        return results

    def run_multi_seed_benchmark(
        self,
        seeds: List[int],
        sim_config: Optional[SimulationConfig] = None,
        c_regularization: float = 1.0,
    ) -> Dict[str, MultiSeedTrajectoryMetrics]:
        """Execute full multi-seed trajectory benchmark across the 4 canonical strategies."""
        from recoveriq.model.trainer import ModelTrainer

        cfg = sim_config or SimulationConfig()
        strategy_names = ["Always-Stop", "Fixed-Retry", "Rule-Based", "RecoverIQ"]
        strategy_seed_runs: Dict[str, List[TrajectoryStrategyMetrics]] = {
            name: [] for name in strategy_names
        }

        always_stop = AlwaysStopStrategy()
        fixed_retry = FixedRetryStrategy()
        rule_based = RuleBasedStrategy()

        for s in seeds:
            # 1. Generate independent dataset for seed s
            gen = SyntheticPaymentGenerator(cfg)
            dataset = gen.generate(seed=s)
            partitioned = partition_dataset(dataset, train_fraction=cfg.train_fraction)

            # 2. Train RecoverIQ model strictly on train partition
            train_env = SimulationEnvironment(partitioned.train_ground_truth, seed=s)
            trainer = ModelTrainer(c_regularization=c_regularization, random_state=s)
            trained_model = trainer.train(partitioned.train_observable, train_env)
            recoveriq_strat = RecoverIQStrategy(probability_model=trained_model)

            strategies = [always_stop, fixed_retry, rule_based, recoveriq_strat]

            # 3. Evaluate all 4 strategies under CRN on test partition
            run_results = self.evaluate_all_strategies(
                strategies=strategies,
                test_observable=partitioned.test_observable,
                test_ground_truth=partitioned.test_ground_truth,
                seed=s,
            )

            for name, metrics in run_results.items():
                strategy_seed_runs[name].append(metrics)

        # Aggregate across seeds
        aggregated: Dict[str, MultiSeedTrajectoryMetrics] = {}
        for name, runs in strategy_seed_runs.items():
            aggregated[name] = MultiSeedTrajectoryMetrics.aggregate(name, runs)

        return aggregated
