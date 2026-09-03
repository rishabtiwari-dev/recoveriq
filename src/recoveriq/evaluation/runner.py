"""Evaluation runner executing multi-strategy benchmarks under Common Random Numbers (CRN).

CRITICAL EXPERIMENTAL INVARIANTS:
1. Held-out test partition: All evaluations occur strictly on unseen test records.
2. Common Policy Gate: All strategies (RecoverIQ, Fixed-Retry, Rule-Based) pass their
   proposed action through the exact same Policy Gate.
3. Common Random Numbers (CRN): For each strategy, a fresh SimulationEnvironment is
   instantiated with the exact same seed S, and payments are evaluated in the exact
   same order. This guarantees identical stochastic outcomes when two strategies choose
   the same action.
4. Single-Decision-Point Scope: Evaluates exactly one intervention decision per payment.
5. Standardized Economics: NRV, costs, and friction penalties are computed identically.
"""

from decimal import Decimal
from typing import Callable, Dict, List, Optional

from recoveriq.config.settings import ActionCostConfig, PenaltyConfig, PolicyConfig
from recoveriq.domain.actions import Action
from recoveriq.domain.decisions import RecoveryDecision
from recoveriq.domain.models import Payment, PaymentContext
from recoveriq.domain.state import PaymentState
from recoveriq.evaluation.metrics import (
    MultiSeedBenchmarkReport,
    MultiSeedStrategyMetrics,
    PaymentEvaluationRecord,
    StrategyMetrics,
)
from recoveriq.evaluation.ablation_strategies import (
    RecoverIQCtxAblationStrategy,
    RecoverIQNoEconStrategy,
)
from recoveriq.evaluation.strategies import (
    FixedRetryStrategy,
    RecoveryStrategy,
    RecoverIQStrategy,
    RuleBasedStrategy,
)
from recoveriq.model.trained_model import TrainedRecoveryProbabilityModel
from recoveriq.model.trainer import ModelTrainer
from recoveriq.policy.gate import InvariantPolicyGate, PolicyGate
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


class EvaluationRunner:
    """Orchestrates end-to-end strategy evaluation with CRN and common policy gating."""

    def __init__(
        self,
        policy_gate: Optional[PolicyGate] = None,
        cost_config: Optional[ActionCostConfig] = None,
        penalty_config: Optional[PenaltyConfig] = None,
    ):
        self.policy_gate = policy_gate or InvariantPolicyGate()
        self.cost_config = cost_config or ActionCostConfig()
        self.penalty_config = penalty_config or PenaltyConfig()

    def evaluate_strategy_on_partition(
        self,
        strategy: RecoveryStrategy,
        test_observable: List[SyntheticPaymentRecord],
        test_ground_truth: List[GroundTruthRecord],
        seed: int,
    ) -> StrategyMetrics:
        """Evaluate a single strategy on the test partition using an independent CRN environment."""
        if not test_observable:
            raise ValueError("test_observable cannot be empty.")

        # CRN Requirement: Strategy gets its OWN SimulationEnvironment with seed S
        env = SimulationEnvironment(test_ground_truth, seed=seed)
        records: List[PaymentEvaluationRecord] = []

        # Iterate in strict, deterministic payment order
        for rec in test_observable:
            # Build standard decision-time context
            context = PaymentContext(
                payment_id=rec.payment_id,
                customer_id=rec.customer_id,
                customer_tier=rec.customer_tier,
                payment_method=rec.payment_method,
                raw_error_code=rec.raw_error_code,
                raw_error_message=rec.raw_error_message,
                failure_category=rec.failure_category,
                failure_severity=rec.failure_severity,
                attempt_count=rec.attempt_count,
                last_attempt_timestamp=rec.failure_timestamp,
                extra_metadata={"amount": float(rec.amount)},
            )
            payment = Payment(
                payment_id=rec.payment_id,
                customer_id=rec.customer_id,
                amount=rec.amount if isinstance(rec.amount, Decimal) else Decimal(str(rec.amount)),
                currency=rec.currency,
                state=PaymentState.FAILED_INITIAL,
                attempt_count=rec.attempt_count - 1,
            )

            # 1. Strategy proposes candidate action
            proposed_action = strategy.propose_action(rec, context)

            # 2. Hard Invariant: Submit to the COMMON Policy Gate
            policy_decision = self.policy_gate.authorize(
                payment=payment,
                context=context,
                decision=RecoveryDecision(
                    payment_id=rec.payment_id,
                    proposed_action=proposed_action,
                ),
            )
            authorized_action = policy_decision.authorized_action

            # 3. Resolve outcome via CRN simulation environment
            outcome = env.apply_action(rec.payment_id, authorized_action)

            # 4. Compute economic payoff using standardized parameters
            cost = self.cost_config.get_cost(authorized_action)
            penalty = self.penalty_config.get_penalty(authorized_action, rec.customer_tier)
            gross_recovered = rec.amount if outcome.recovered else Decimal("0.00")
            nrv = gross_recovered - cost - penalty

            records.append(
                PaymentEvaluationRecord(
                    payment_id=rec.payment_id,
                    proposed_action=proposed_action,
                    authorized_action=authorized_action,
                    is_authorized=policy_decision.is_authorized,
                    rejection_reason=policy_decision.rejection_reason,
                    recovered=outcome.recovered,
                    payment_amount=rec.amount,
                    gross_recovered=gross_recovered,
                    intervention_cost=cost,
                    friction_penalty=penalty,
                    net_recovered_value=nrv,
                )
            )

        return StrategyMetrics.compute(
            strategy_name=strategy.name,
            seed=seed,
            records=records,
        )

    def evaluate_all_strategies(
        self,
        strategies: List[RecoveryStrategy],
        test_observable: List[SyntheticPaymentRecord],
        test_ground_truth: List[GroundTruthRecord],
        seed: int,
    ) -> Dict[str, StrategyMetrics]:
        """Evaluate multiple strategies on the same test partition using Common Random Numbers."""
        results: Dict[str, StrategyMetrics] = {}
        for strat in strategies:
            results[strat.name] = self.evaluate_strategy_on_partition(
                strategy=strat,
                test_observable=test_observable,
                test_ground_truth=test_ground_truth,
                seed=seed,
            )
        return results

    def run_multi_seed_benchmark(
        self,
        seeds: List[int],
        sim_config: Optional[SimulationConfig] = None,
        c_regularization: float = 1.0,
    ) -> MultiSeedBenchmarkReport:
        """Execute full multi-seed benchmark across all 3 strategies with complete train/test isolation."""
        cfg = sim_config or SimulationConfig()
        strategy_seed_runs: Dict[str, List[StrategyMetrics]] = {
            "Fixed-Retry": [],
            "Rule-Based": [],
            "RecoverIQ": [],
        }

        fixed_retry = FixedRetryStrategy()
        rule_based = RuleBasedStrategy()

        for s in seeds:
            # 1. Generate independent dataset for seed s
            gen = SyntheticPaymentGenerator(cfg)
            dataset = gen.generate(seed=s)
            partitioned = partition_dataset(dataset, train_fraction=cfg.train_fraction)

            # 2. Train RecoverIQ model strictly on train partition of seed s
            train_env = SimulationEnvironment(partitioned.train_ground_truth, seed=s)
            trainer = ModelTrainer(c_regularization=c_regularization, random_state=s)
            trained_model = trainer.train(partitioned.train_observable, train_env)
            recoveriq_strat = RecoverIQStrategy(probability_model=trained_model)

            # 3. Evaluate all 3 strategies on held-out test partition under CRN
            strats = [fixed_retry, rule_based, recoveriq_strat]
            run_results = self.evaluate_all_strategies(
                strategies=strats,
                test_observable=partitioned.test_observable,
                test_ground_truth=partitioned.test_ground_truth,
                seed=s,
            )

            for name, metrics in run_results.items():
                strategy_seed_runs[name].append(metrics)

        # Aggregate across seeds (Mean ± Std)
        aggregated: Dict[str, MultiSeedStrategyMetrics] = {}
        for name, runs in strategy_seed_runs.items():
            aggregated[name] = MultiSeedStrategyMetrics.aggregate(name, runs)

        return MultiSeedBenchmarkReport(strategies=aggregated, seeds=seeds)

    def run_ablation_benchmark(
        self,
        seeds: List[int],
        sim_config: Optional[SimulationConfig] = None,
        c_regularization: float = 1.0,
    ) -> MultiSeedBenchmarkReport:
        """Execute full multi-seed ablation benchmark across 5 strategies under CRN.

        Evaluates:
        1. Fixed-Retry (Baseline)
        2. Rule-Based (Baseline)
        3. RecoverIQ (Full System with generator-oracle context)
        4. RecoverIQ-CtxAblation (A1: Rule-based context extraction vs generator-oracle context)
        5. RecoverIQ-NoEcon (A2: Greedy probability maximization without economic engine)
        """
        cfg = sim_config or SimulationConfig()
        strategy_seed_runs: Dict[str, List[StrategyMetrics]] = {
            "Fixed-Retry": [],
            "Rule-Based": [],
            "RecoverIQ": [],
            "RecoverIQ-CtxAblation": [],
            "RecoverIQ-NoEcon": [],
        }

        fixed_retry = FixedRetryStrategy()
        rule_based = RuleBasedStrategy()

        for s in seeds:
            # 1. Generate independent dataset for seed s
            gen = SyntheticPaymentGenerator(cfg)
            dataset = gen.generate(seed=s)
            partitioned = partition_dataset(dataset, train_fraction=cfg.train_fraction)

            # 2. Train RecoverIQ model strictly on train partition of seed s
            train_env = SimulationEnvironment(partitioned.train_ground_truth, seed=s)
            trainer = ModelTrainer(c_regularization=c_regularization, random_state=s)
            trained_model = trainer.train(partitioned.train_observable, train_env)

            recoveriq_strat = RecoverIQStrategy(probability_model=trained_model)
            ctx_ablation_strat = RecoverIQCtxAblationStrategy(probability_model=trained_model)
            no_econ_strat = RecoverIQNoEconStrategy(probability_model=trained_model)

            # 3. Evaluate all 5 strategies on held-out test partition under CRN
            strats = [
                fixed_retry,
                rule_based,
                recoveriq_strat,
                ctx_ablation_strat,
                no_econ_strat,
            ]
            run_results = self.evaluate_all_strategies(
                strategies=strats,
                test_observable=partitioned.test_observable,
                test_ground_truth=partitioned.test_ground_truth,
                seed=s,
            )

            for name, metrics in run_results.items():
                strategy_seed_runs[name].append(metrics)

        # Aggregate across seeds (Mean ± Std)
        aggregated: Dict[str, MultiSeedStrategyMetrics] = {}
        for name, runs in strategy_seed_runs.items():
            aggregated[name] = MultiSeedStrategyMetrics.aggregate(name, runs)

        return MultiSeedBenchmarkReport(strategies=aggregated, seeds=seeds)

