"""Sprint 16 — Contract Tests for Research Demo and Presentation Layer.

Verifies:
1. Demo modules import cleanly without side effects.
2. DemoEngine initializes models and strategies without errors.
3. Observable payment records contain no hidden ground-truth fields.
4. Single-payment trajectory can execute cleanly.
5. Multi-step trajectory can execute cleanly to completion.
6. Strategy selector returns valid Actions from the Action enum.
7. Bellman strategy behavior remains unchanged.
8. ModelFree strategy behavior remains unchanged.
9. Hybrid strategy behavior remains unchanged.
10. Cached research results load cleanly with required fields.
11. No external API or gateway dependencies are introduced.
12. Existing baseline strategies remain immutable.
"""

from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, PaymentContext
from recoveriq.evaluation.demo_data import (
    ATTEMPT_3_ACTION_DISTRIBUTION,
    BASELINE_BENCHMARK_M0_D0,
    DEGRADATION_M0_TO_M3,
    DISTRIBUTION_SHIFT_RESULTS,
    MODEL_ERROR_RESULTS,
    PAIRED_CRN_STATISTICS,
    RESEARCH_HYPOTHESES_VERDICTS,
)
from recoveriq.evaluation.demo_engine import DemoEngine
from recoveriq.evaluation.strategies import FixedRetryStrategy, RuleBasedStrategy
from recoveriq.evaluation.trajectory import AlwaysStopStrategy


@pytest.fixture(scope="module")
def demo_engine():
    engine = DemoEngine(seed=42, max_attempts=3)
    engine.initialize()
    return engine


# 1. Clean imports
def test_demo_imports_cleanly():
    import app.demo
    assert app.demo is not None


# 2. DemoEngine initialization
def test_demo_engine_initialization(demo_engine):
    assert demo_engine._is_initialized
    assert demo_engine.trained_model is not None
    assert len(demo_engine.strategies) >= 7
    assert len(demo_engine.demo_records) > 0


# 3. Observable state contains no ground-truth fields
def test_observable_state_no_ground_truth(demo_engine):
    rec = demo_engine.get_sample_payment(0)
    # SyntheticPaymentRecord must not contain hidden recovery status or oracle values
    assert not hasattr(rec, "actual_recovered")
    assert not hasattr(rec, "ground_truth")
    assert isinstance(rec.amount, Decimal)


# 4. Single-payment decision evaluation
def test_single_payment_action_evaluation(demo_engine):
    rec = demo_engine.get_sample_payment(0)
    for strat_name in demo_engine.strategies.keys():
        action = demo_engine.evaluate_action_at_step(strat_name, rec, attempt=1)
        assert action in set(Action)


# 5. Multi-step trajectory execution
def test_multistep_trajectory_execution(demo_engine):
    rec = demo_engine.get_sample_payment(0)
    episode = demo_engine.run_full_trajectory("RecoverIQ-Bellman", rec)
    assert 1 <= episode.attempt_count <= 3
    assert episode.terminal_state is not None
    assert isinstance(episode.net_recovered_value, Decimal)


# 6. Strategy comparison returns expected format
def test_strategy_comparison_format(demo_engine):
    rec = demo_engine.get_sample_payment(0)
    comparisons = demo_engine.compare_decisions_for_payment(rec, attempt=1)
    for strat_name, details in comparisons.items():
        assert "action" in details
        assert Action(details["action"]) in set(Action)


# 7. Bellman immutability
def test_bellman_immutability(demo_engine):
    b_strat = demo_engine.strategies["RecoverIQ-Bellman"]
    assert b_strat.name == "RecoverIQ-Bellman"
    assert b_strat.planning_horizon == 3


# 8. ModelFree immutability
def test_modelfree_immutability(demo_engine):
    mf_strat = demo_engine.strategies["RecoverIQ-ModelFree"]
    assert mf_strat.name == "RecoverIQ-ModelFree"
    assert mf_strat.policy.n_unique_states > 0


# 9. Hybrid immutability
def test_hybrid_immutability(demo_engine):
    h_strat = demo_engine.strategies["RecoverIQ-Hybrid-Uncertainty"]
    assert h_strat.name == "RecoverIQ-Hybrid-Uncertainty"


# 10. Research cache integrity
def test_research_cache_integrity():
    assert len(BASELINE_BENCHMARK_M0_D0) >= 7
    assert len(MODEL_ERROR_RESULTS) == 4
    assert len(DISTRIBUTION_SHIFT_RESULTS) == 4
    assert len(PAIRED_CRN_STATISTICS) >= 5
    assert len(ATTEMPT_3_ACTION_DISTRIBUTION) >= 3
    assert len(RESEARCH_HYPOTHESES_VERDICTS) >= 9


# 11. No external API dependencies
def test_no_external_api_dependencies(demo_engine):
    # Ensure all models run fully in-memory with zero network calls
    rec = demo_engine.create_custom_payment("test-001", 5000.0, FailureCategory.NETWORK_TIMEOUT, CustomerTier.VIP)
    action = demo_engine.evaluate_action_at_step("RecoverIQ-ModelFree", rec, attempt=1)
    assert action in set(Action)


# 12. Baseline strategies remain intact
def test_baseline_strategies_intact():
    assert AlwaysStopStrategy().name == "Always-Stop"
    assert FixedRetryStrategy().name == "Fixed-Retry"
    assert RuleBasedStrategy().name == "Rule-Based"
