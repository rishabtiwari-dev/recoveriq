"""Sprint 2 Tests — Reproducibility and seed behaviour."""

import pytest
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.generator import SyntheticPaymentGenerator


def _generate_pair(seed: int, n: int = 200):
    cfg = SimulationConfig(n_payments=n, n_customers=50)
    gen = SyntheticPaymentGenerator(cfg)
    a = gen.generate(seed=seed)
    b = gen.generate(seed=seed)
    return a, b


def test_same_seed_produces_identical_records():
    """Same seed must produce byte-identical records."""
    a, b = _generate_pair(seed=42)
    assert len(a) == len(b)
    for oa, ob in zip(a.observable_records, b.observable_records):
        assert oa.payment_id == ob.payment_id
        assert oa.amount == ob.amount
        assert oa.failure_category == ob.failure_category
        assert oa.customer_tier == ob.customer_tier
        assert oa.payment_method == ob.payment_method


def test_same_seed_produces_identical_ground_truth():
    """Same seed must produce identical hidden ground-truth profiles."""
    a, b = _generate_pair(seed=100)
    for ga, gb in zip(a.ground_truth_records, b.ground_truth_records):
        assert ga.payment_id == gb.payment_id
        assert ga.latent_recoverability_profile == gb.latent_recoverability_profile
        for action in ga.action_base_probabilities:
            assert ga.action_base_probabilities[action] == pytest.approx(
                gb.action_base_probabilities[action], abs=1e-12
            )


def test_different_seeds_produce_different_datasets():
    """Different seeds must produce different observable records."""
    cfg = SimulationConfig(n_payments=200, n_customers=50)
    gen = SyntheticPaymentGenerator(cfg)
    d42 = gen.generate(seed=42)
    d99 = gen.generate(seed=99)
    # At least one field must differ (extremely unlikely to be identical)
    amounts_42 = [r.amount for r in d42.observable_records]
    amounts_99 = [r.amount for r in d99.observable_records]
    assert amounts_42 != amounts_99


def test_seed_stored_on_dataset():
    """Generated dataset must store the seed used."""
    cfg = SimulationConfig(n_payments=10, n_customers=10)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=777)
    assert ds.seed == 777


def test_config_default_seed_used_when_no_seed_passed():
    """Default seed from config should be applied when generate() called without seed."""
    cfg = SimulationConfig(n_payments=50, n_customers=20, default_seed=2024)
    gen = SyntheticPaymentGenerator(cfg)
    ds_default = gen.generate()
    ds_explicit = gen.generate(seed=2024)
    amounts_default = [r.amount for r in ds_default.observable_records]
    amounts_explicit = [r.amount for r in ds_explicit.observable_records]
    assert amounts_default == amounts_explicit
