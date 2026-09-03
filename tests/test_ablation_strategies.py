"""Sprint 6 Tests — Ablation strategy units, contracts, and decision logic (SPEC §17)."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from recoveriq.context.extractor import RuleBasedContextExtractor
from recoveriq.domain.actions import Action
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.evaluation.ablation_strategies import (
    GreedyProbabilitySelector,
    RecoverIQCtxAblationStrategy,
    RecoverIQNoEconStrategy,
)
from recoveriq.evaluation.strategies import RecoverIQStrategy
from recoveriq.model.probability import ProbabilityEstimate, StubProbabilityModel
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.schema import SyntheticPaymentRecord


@pytest.fixture
def sim_partition():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


def test_greedy_selector_returns_argmax_probability():
    """GreedyProbabilitySelector must select the non-STOP action with the highest probability."""
    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(action=Action.RETRY_NOW, probability=Decimal("0.35")),
        Action.RETRY_LATER: ProbabilityEstimate(action=Action.RETRY_LATER, probability=Decimal("0.45")),
        Action.SEND_LINK: ProbabilityEstimate(action=Action.SEND_LINK, probability=Decimal("0.40")),
        Action.NUDGE: ProbabilityEstimate(action=Action.NUDGE, probability=Decimal("0.25")),
        Action.ESCALATE: ProbabilityEstimate(action=Action.ESCALATE, probability=Decimal("0.65")),
        Action.STOP: ProbabilityEstimate(action=Action.STOP, probability=Decimal("0.00")),
    }

    selected = GreedyProbabilitySelector.select(probabilities)
    assert selected == Action.ESCALATE


def test_greedy_selector_returns_stop_when_all_zero():
    """If all non-STOP action probabilities are 0.00, selector must return STOP."""
    probabilities = {
        action: ProbabilityEstimate(action=action, probability=Decimal("0.00"))
        for action in Action
    }

    selected = GreedyProbabilitySelector.select(probabilities)
    assert selected == Action.STOP


def test_no_econ_selects_highest_prob_not_highest_ev():
    """Verify that RecoverIQNoEconStrategy picks argmax P̂ even when EV favors a cheaper action."""
    # Scenario:
    # ESCALATE has highest prob (0.70) but high cost (3.50).
    # RETRY_LATER has lower prob (0.65) but negligible cost (0.15).
    # On a very small payment (e.g. 2.00 INR):
    # EV(ESCALATE) = 0.70 * 2.00 - 3.50 - 0.00 = 1.40 - 3.50 = -2.10 (Negative EV -> STOP)
    # EV(RETRY_LATER) = 0.65 * 2.00 - 0.15 - 0.02 = 1.30 - 0.17 = +1.13
    # Default economic engine would choose RETRY_LATER or STOP.
    # But GreedyProbabilitySelector will blindly pick ESCALATE.

    class MockHighEscalateModel:
        def estimate_probabilities(self, context):
            return {
                Action.RETRY_NOW: ProbabilityEstimate(action=Action.RETRY_NOW, probability=Decimal("0.30")),
                Action.RETRY_LATER: ProbabilityEstimate(action=Action.RETRY_LATER, probability=Decimal("0.65")),
                Action.SEND_LINK: ProbabilityEstimate(action=Action.SEND_LINK, probability=Decimal("0.40")),
                Action.NUDGE: ProbabilityEstimate(action=Action.NUDGE, probability=Decimal("0.20")),
                Action.ESCALATE: ProbabilityEstimate(action=Action.ESCALATE, probability=Decimal("0.70")),
                Action.STOP: ProbabilityEstimate(action=Action.STOP, probability=Decimal("0.00")),
            }

    mock_model = MockHighEscalateModel()
    strat_full = RecoverIQStrategy(probability_model=mock_model)
    strat_no_econ = RecoverIQNoEconStrategy(probability_model=mock_model)

    rec = SyntheticPaymentRecord(
        payment_id="pay_test_001",
        customer_id="cust_001",
        amount=Decimal("2.00"),
        currency="USD",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="INSUFFICIENT_FUNDS",
        raw_error_message="Insufficient funds",
        failure_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        attempt_count=1,
    )
    ctx = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=rec.attempt_count,
        extra_metadata={"amount": float(rec.amount)},
    )

    action_full = strat_full.propose_action(rec, ctx)
    action_no_econ = strat_no_econ.propose_action(rec, ctx)

    # Full EV engine chooses RETRY_LATER because ESCALATE is net negative
    assert action_full == Action.RETRY_LATER
    # Ablated NoEcon engine blindly chooses ESCALATE because 0.70 > 0.65
    assert action_no_econ == Action.ESCALATE


def test_ctx_ablation_uses_rule_based_extractor():
    """RecoverIQCtxAblationStrategy must extract context via RuleBasedContextExtractor."""
    rec = SyntheticPaymentRecord(
        payment_id="pay_test_002",
        customer_id="cust_002",
        amount=Decimal("500.00"),
        currency="USD",
        failure_category=FailureCategory.UNKNOWN,  # Oracle says UNKNOWN
        failure_severity=FailureSeverity.RECOVERABLE,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="504_GATEWAY_TIMEOUT",     # Raw string is network timeout
        raw_error_message="Gateway timeout",
        failure_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        attempt_count=1,
    )
    # The runner passes the oracle context
    oracle_ctx = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=FailureCategory.UNKNOWN,
        failure_severity=FailureSeverity.RECOVERABLE,
        attempt_count=rec.attempt_count,
        extra_metadata={"amount": float(rec.amount)},
    )

    seen_categories = []

    class InspectingModel:
        def estimate_probabilities(self, context):
            seen_categories.append(context.failure_category)
            return StubProbabilityModel().estimate_probabilities(context)

    model = InspectingModel()
    strat = RecoverIQCtxAblationStrategy(probability_model=model)

    strat.propose_action(rec, oracle_ctx)

    # Extractor should have parsed 504_GATEWAY_TIMEOUT into NETWORK_TIMEOUT!
    assert len(seen_categories) == 1
    assert seen_categories[0] == FailureCategory.NETWORK_TIMEOUT


def test_ctx_ablation_differs_on_unrecognized_auth_rejected():
    """On AUTH_REJECTED, generator says AUTHENTICATION_REJECTED, but RuleBasedContextExtractor produces UNKNOWN."""
    rec = SyntheticPaymentRecord(
        payment_id="pay_test_003",
        customer_id="cust_003",
        amount=Decimal("1000.00"),
        currency="USD",
        failure_category=FailureCategory.AUTHENTICATION_REJECTED,
        failure_severity=FailureSeverity.STRUCTURAL,
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="AUTH_REJECTED",
        raw_error_message="Issuer rejected authentication attempt",
        failure_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        attempt_count=1,
    )
    oracle_ctx = PaymentContext(
        payment_id=rec.payment_id,
        customer_id=rec.customer_id,
        customer_tier=rec.customer_tier,
        payment_method=rec.payment_method,
        raw_error_code=rec.raw_error_code,
        raw_error_message=rec.raw_error_message,
        failure_category=rec.failure_category,
        failure_severity=rec.failure_severity,
        attempt_count=rec.attempt_count,
        extra_metadata={"amount": float(rec.amount)},
    )

    seen_categories = []

    class InspectingModel:
        def estimate_probabilities(self, context):
            seen_categories.append(context.failure_category)
            return StubProbabilityModel().estimate_probabilities(context)

    strat = RecoverIQCtxAblationStrategy(probability_model=InspectingModel())
    strat.propose_action(rec, oracle_ctx)

    # In oracle_ctx, category is AUTHENTICATION_REJECTED
    assert oracle_ctx.failure_category == FailureCategory.AUTHENTICATION_REJECTED
    # But A1 context ablation strategy extracted UNKNOWN from "AUTH_REJECTED"
    assert len(seen_categories) == 1
    assert seen_categories[0] == FailureCategory.UNKNOWN
