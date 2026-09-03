"""Sprint 4 Tests — Feature preprocessor deterministic encoding and serialization."""

import pytest
import numpy as np

from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.model.preprocessing import FeaturePreprocessor


@pytest.fixture
def sample_feature_dicts():
    return [
        {
            "failure_category": "INSUFFICIENT_FUNDS",
            "failure_severity": "RECOVERABLE",
            "customer_tier": "STANDARD",
            "payment_method": "UPI",
            "amount": 500.0,
            "attempt_count": 1,
            "failure_hour": 14,
            "failure_day_of_week": 2,
        },
        {
            "failure_category": "NETWORK_TIMEOUT",
            "failure_severity": "TRANSIENT",
            "customer_tier": "VIP",
            "payment_method": "CREDIT_CARD",
            "amount": 2500.0,
            "attempt_count": 2,
            "failure_hour": 9,
            "failure_day_of_week": 4,
        },
    ]


def test_preprocessor_requires_fit_before_transform(sample_feature_dicts):
    """Transforming before fit must raise RuntimeError."""
    p = FeaturePreprocessor()
    with pytest.raises(RuntimeError):
        p.transform(sample_feature_dicts)


def test_preprocessor_fit_and_transform_shape(sample_feature_dicts):
    """Preprocessor must produce a 2D float array with exact expected dimensions."""
    p = FeaturePreprocessor()
    p.fit(sample_feature_dicts)

    assert p.is_fitted is True
    # 9 categories + 4 severities + 4 tiers + 5 methods + 4 numericals = 26 dimensions
    assert p.num_dimensions == 26
    assert len(p.feature_names) == 26

    X = p.transform(sample_feature_dicts)
    assert isinstance(X, np.ndarray)
    assert X.shape == (2, 26)


def test_preprocessor_transform_context():
    """Runtime PaymentContext must transform cleanly into a (1, 26) array."""
    p = FeaturePreprocessor()
    p.fit([
        {
            "failure_category": "INSUFFICIENT_FUNDS",
            "failure_severity": "RECOVERABLE",
            "customer_tier": "PREMIUM",
            "payment_method": "DEBIT_CARD",
            "amount": 1000.0,
            "attempt_count": 1,
            "failure_hour": 12,
            "failure_day_of_week": 1,
        }
    ])

    ctx = PaymentContext(
        payment_id="pay_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.PREMIUM,
        payment_method=PaymentMethod.DEBIT_CARD,
        raw_error_code="LOW_BALANCE",
        raw_error_message="Balance insufficient",
        failure_category=FailureCategory.INSUFFICIENT_FUNDS,
        failure_severity=FailureSeverity.RECOVERABLE,
        attempt_count=1,
    )

    X_ctx = p.transform_context(ctx, amount=1000.0)
    assert X_ctx.shape == (1, 26)


def test_preprocessor_serialization_roundtrip(sample_feature_dicts):
    """Serialized preprocessor state must restore cleanly with identical transformation outputs."""
    p1 = FeaturePreprocessor()
    p1.fit(sample_feature_dicts)
    X1 = p1.transform(sample_feature_dicts)

    data = p1.to_dict()
    p2 = FeaturePreprocessor.from_dict(data)
    X2 = p2.transform(sample_feature_dicts)

    np.testing.assert_array_almost_equal(X1, X2)
    assert p2.feature_names == p1.feature_names


def test_preprocessor_handles_unknown_category_safely():
    """Unrecognized category strings must not raise; their one-hot slots remain 0."""
    p = FeaturePreprocessor()
    p.fit([{"failure_category": "INSUFFICIENT_FUNDS", "amount": 100.0}])

    ex_unknown = {
        "failure_category": "ALIEN_SIGNAL_LOST",
        "failure_severity": "UNKNOWN",
        "customer_tier": "NEW",
        "payment_method": "WALLET",
        "amount": 100.0,
    }
    X = p.transform([ex_unknown])
    assert X.shape == (1, 26)
    # The first 9 features are categories; none of them should be 1.0 for ALIEN_SIGNAL_LOST
    assert np.sum(X[0, :9]) == 0.0
