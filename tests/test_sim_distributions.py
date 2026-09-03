"""Sprint 2 Tests — Distribution sanity checks."""

from collections import Counter
import pytest
from recoveriq.domain.models import CustomerTier, FailureCategory
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.schema import RecoverabilityProfile


@pytest.fixture(scope="module")
def large_dataset():
    """Larger dataset for distribution tests (more statistical power)."""
    cfg = SimulationConfig(n_payments=2000, n_customers=500)
    gen = SyntheticPaymentGenerator(cfg)
    return gen.generate(seed=42)


def test_all_failure_categories_present(large_dataset):
    """Every FailureCategory configured in the default distribution must appear."""
    categories = {r.failure_category for r in large_dataset.observable_records}
    for cat in FailureCategory:
        assert cat in categories, f"FailureCategory.{cat.name} absent from 2000-record dataset"


def test_all_customer_tiers_present(large_dataset):
    """Every CustomerTier must appear in the generated data."""
    tiers = {r.customer_tier for r in large_dataset.observable_records}
    for tier in CustomerTier:
        assert tier in tiers, f"CustomerTier.{tier.name} absent from 2000-record dataset"


def test_all_recoverability_profiles_present(large_dataset):
    """All five RecoverabilityProfile values should appear across 2000 payments."""
    profiles = {gt.latent_recoverability_profile for gt in large_dataset.ground_truth_records}
    for profile in RecoverabilityProfile:
        assert profile in profiles, f"RecoverabilityProfile.{profile.name} absent"


def test_failure_category_distribution_roughly_matches_config(large_dataset):
    """Dominant categories should dominate; UNKNOWN should be rare."""
    counts = Counter(r.failure_category for r in large_dataset.observable_records)
    n = len(large_dataset)
    # INSUFFICIENT_FUNDS is the highest-weight category (~28%) — must be > 10%
    insuf = counts.get(FailureCategory.INSUFFICIENT_FUNDS, 0) / n
    assert insuf > 0.10, f"INSUFFICIENT_FUNDS coverage too low: {insuf:.3f}"
    # UNKNOWN is the lowest-weight category (~2%) — must be < 15%
    unknown = counts.get(FailureCategory.UNKNOWN, 0) / n
    assert unknown < 0.15, f"UNKNOWN coverage too high: {unknown:.3f}"


def test_customer_tier_distribution_roughly_matches_config(large_dataset):
    """STANDARD should be most common; VIP least common."""
    counts = Counter(r.customer_tier for r in large_dataset.observable_records)
    n = len(large_dataset)
    standard_frac = counts.get(CustomerTier.STANDARD, 0) / n
    vip_frac = counts.get(CustomerTier.VIP, 0) / n
    assert standard_frac > vip_frac, "STANDARD should be more common than VIP"
    assert standard_frac > 0.25, f"STANDARD fraction too low: {standard_frac:.3f}"


def test_amount_ranges_respect_tier_bounds(large_dataset):
    """Amounts must fall within configured tier bounds."""
    from recoveriq.simulation.config import SimulationConfig
    cfg = SimulationConfig()
    for record in large_dataset.observable_records:
        lo, hi = cfg.amount_distribution.tier_amount_ranges[record.customer_tier]
        assert lo <= record.amount <= hi, (
            f"Amount {record.amount} out of range [{lo}, {hi}] for tier {record.customer_tier}"
        )


def test_hard_decline_recovery_profile_skews_low(large_dataset):
    """Hard declines in ground truth should be concentrated in low-recoverability profiles."""
    hard_decline_gts = [
        gt for obs, gt in zip(large_dataset.observable_records, large_dataset.ground_truth_records)
        if obs.failure_category == FailureCategory.HARD_DECLINE
    ]
    if len(hard_decline_gts) < 5:
        pytest.skip("Not enough hard decline records for distribution test")

    low_profiles = {RecoverabilityProfile.VERY_LOW, RecoverabilityProfile.LOW}
    low_count = sum(
        1 for gt in hard_decline_gts
        if gt.latent_recoverability_profile in low_profiles
    )
    low_fraction = low_count / len(hard_decline_gts)
    assert low_fraction >= 0.70, (
        f"Hard decline records should mostly have low recoverability profiles, "
        f"got {low_fraction:.3f}"
    )
