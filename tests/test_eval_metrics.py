"""Sprint 5 Tests — Evaluation metrics, NRV economics, and multi-seed aggregation."""

from decimal import Decimal
import pytest

from recoveriq.domain.actions import Action
from recoveriq.evaluation.metrics import (
    MultiSeedBenchmarkReport,
    MultiSeedStrategyMetrics,
    PaymentEvaluationRecord,
    StrategyMetrics,
)


def test_payment_evaluation_record_nrv():
    """PaymentEvaluationRecord must satisfy NRV = gross - cost - penalty."""
    rec = PaymentEvaluationRecord(
        payment_id="pay_001",
        proposed_action=Action.RETRY_NOW,
        authorized_action=Action.RETRY_NOW,
        is_authorized=True,
        rejection_reason=None,
        recovered=True,
        payment_amount=Decimal("1000.00"),
        gross_recovered=Decimal("1000.00"),
        intervention_cost=Decimal("0.15"),
        friction_penalty=Decimal("0.05"),
        net_recovered_value=Decimal("999.80"),
    )

    assert rec.net_recovered_value == rec.gross_recovered - rec.intervention_cost - rec.friction_penalty


def test_strategy_metrics_computation():
    """StrategyMetrics must compute correct rates, totals, and action distributions."""
    records = [
        # Payment 1: recovered with RETRY_NOW
        PaymentEvaluationRecord(
            payment_id="pay_1",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.RETRY_NOW,
            is_authorized=True,
            rejection_reason=None,
            recovered=True,
            payment_amount=Decimal("500.00"),
            gross_recovered=Decimal("500.00"),
            intervention_cost=Decimal("0.15"),
            friction_penalty=Decimal("0.05"),
            net_recovered_value=Decimal("499.80"),
        ),
        # Payment 2: not recovered with RETRY_NOW
        PaymentEvaluationRecord(
            payment_id="pay_2",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.RETRY_NOW,
            is_authorized=True,
            rejection_reason=None,
            recovered=False,
            payment_amount=Decimal("200.00"),
            gross_recovered=Decimal("0.00"),
            intervention_cost=Decimal("0.15"),
            friction_penalty=Decimal("0.05"),
            net_recovered_value=Decimal("-0.20"),
        ),
        # Payment 3: policy blocked and clamped to STOP
        PaymentEvaluationRecord(
            payment_id="pay_3",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.STOP,
            is_authorized=False,
            rejection_reason="Hard decline",
            recovered=False,
            payment_amount=Decimal("300.00"),
            gross_recovered=Decimal("0.00"),
            intervention_cost=Decimal("0.00"),
            friction_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("0.00"),
        ),
    ]

    metrics = StrategyMetrics.compute("Fixed-Retry", seed=42, records=records)

    assert metrics.n_payments == 3
    assert metrics.n_recovered == 1
    assert metrics.recovery_rate == pytest.approx(1.0 / 3.0)
    assert metrics.total_gross_revenue == Decimal("500.00")
    assert metrics.total_cost == Decimal("0.30")
    assert metrics.total_penalty == Decimal("0.10")
    assert metrics.total_nrv == Decimal("499.60")
    assert metrics.mean_nrv == pytest.approx(float(Decimal("499.60")) / 3.0)
    assert metrics.policy_blocked_count == 1
    assert metrics.policy_block_rate == pytest.approx(1.0 / 3.0)
    assert metrics.policy_violation_rate == 0.0
    assert metrics.action_counts[Action.RETRY_NOW] == 2
    assert metrics.action_counts[Action.STOP] == 1


def test_multi_seed_metrics_aggregation():
    """MultiSeedStrategyMetrics must aggregate mean and standard deviation across seed runs."""
    r1 = [
        PaymentEvaluationRecord(
            payment_id="p1",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.RETRY_NOW,
            is_authorized=True,
            rejection_reason=None,
            recovered=True,
            payment_amount=Decimal("100.00"),
            gross_recovered=Decimal("100.00"),
            intervention_cost=Decimal("1.00"),
            friction_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("99.00"),
        )
    ]
    r2 = [
        PaymentEvaluationRecord(
            payment_id="p2",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.RETRY_NOW,
            is_authorized=True,
            rejection_reason=None,
            recovered=False,
            payment_amount=Decimal("100.00"),
            gross_recovered=Decimal("0.00"),
            intervention_cost=Decimal("1.00"),
            friction_penalty=Decimal("0.00"),
            net_recovered_value=Decimal("-1.00"),
        )
    ]

    run1 = StrategyMetrics.compute("RecoverIQ", seed=42, records=r1)
    run2 = StrategyMetrics.compute("RecoverIQ", seed=100, records=r2)

    agg = MultiSeedStrategyMetrics.aggregate("RecoverIQ", [run1, run2])

    assert agg.n_seeds == 2
    assert agg.mean_total_nrv == pytest.approx(49.0)  # (99 + (-1)) / 2
    assert agg.std_total_nrv > 0.0
    assert agg.mean_recovery_rate == pytest.approx(0.5)


def test_benchmark_report_summary_table():
    """MultiSeedBenchmarkReport summary_table must render readable comparative table."""
    r = [
        PaymentEvaluationRecord(
            payment_id="p",
            proposed_action=Action.RETRY_NOW,
            authorized_action=Action.RETRY_NOW,
            is_authorized=True,
            rejection_reason=None,
            recovered=True,
            payment_amount=Decimal("100.00"),
            gross_recovered=Decimal("100.00"),
            intervention_cost=Decimal("0.15"),
            friction_penalty=Decimal("0.05"),
            net_recovered_value=Decimal("99.80"),
        )
    ]
    runs = [StrategyMetrics.compute("Fixed-Retry", seed=42, records=r)]
    agg_fr = MultiSeedStrategyMetrics.aggregate("Fixed-Retry", runs)
    agg_rb = MultiSeedStrategyMetrics.aggregate("Rule-Based", runs)
    agg_riq = MultiSeedStrategyMetrics.aggregate("RecoverIQ", runs)

    report = MultiSeedBenchmarkReport(
        strategies={"Fixed-Retry": agg_fr, "Rule-Based": agg_rb, "RecoverIQ": agg_riq},
        seeds=[42],
    )

    table = report.summary_table()
    assert "Fixed-Retry" in table
    assert "Rule-Based" in table
    assert "RecoverIQ" in table
    assert "0.00% (PASSED)" in table
