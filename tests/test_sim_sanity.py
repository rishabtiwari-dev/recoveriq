"""Sprint 2 Tests — Full sanity check pipeline and multi-seed verification."""

import pytest
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.sanity import check_dataset_sanity


@pytest.fixture(scope="module")
def full_sanity_result():
    """Run the full sanity check pipeline on a 2000-record dataset."""
    cfg = SimulationConfig(n_payments=2000, n_customers=500)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    env = SimulationEnvironment(ds.ground_truth_records, seed=42)
    return check_dataset_sanity(
        dataset=ds,
        partitioned=partitioned,
        env=env,
    )


def test_full_sanity_check_passes(full_sanity_result):
    """The full sanity check suite must pass on a properly generated dataset."""
    assert full_sanity_result.passed, (
        f"Sanity check failed:\n{full_sanity_result.report()}"
    )


def test_sanity_no_critical_failures(full_sanity_result):
    """Sanity check must have zero hard failures."""
    assert len(full_sanity_result.failures) == 0, (
        f"Sanity check hard failures: {full_sanity_result.failures}"
    )


def test_sanity_stats_populated(full_sanity_result):
    """Sanity check stats dict must contain expected keys."""
    stats = full_sanity_result.stats
    assert "n_total" in stats
    assert "unique_payment_ids" in stats
    assert "n_train" in stats
    assert "n_test" in stats
    assert stats["leakage_detected"] is False


def test_multi_seed_evaluation():
    """Run across 5 seeds and verify all produce valid datasets and no leakage."""
    seeds = [42, 100, 777, 999, 2024]
    results = {}
    cfg = SimulationConfig(n_payments=500, n_customers=100)
    gen = SyntheticPaymentGenerator(cfg)

    for seed in seeds:
        ds = gen.generate(seed=seed)
        partitioned = partition_dataset(ds, train_fraction=0.75)
        env = SimulationEnvironment(ds.ground_truth_records, seed=seed)
        result = check_dataset_sanity(
            dataset=ds,
            partitioned=partitioned,
            env=env,
        )
        results[seed] = result

    for seed, result in results.items():
        assert result.passed, (
            f"Sanity check failed for seed={seed}:\n{result.report()}"
        )
        assert not result.stats.get("leakage_detected", True), (
            f"Leakage detected for seed={seed}"
        )


def test_datasets_differ_across_seeds():
    """Each seed must produce a distinct dataset (confirming seed-driven variation)."""
    cfg = SimulationConfig(n_payments=200, n_customers=50)
    gen = SyntheticPaymentGenerator(cfg)

    seeds = [42, 100, 777, 999, 2024]
    first_amounts = {}
    for seed in seeds:
        ds = gen.generate(seed=seed)
        first_amounts[seed] = [str(r.amount) for r in ds.observable_records[:10]]

    # Verify that not all seeds produce the same first 10 amounts
    unique_sequences = len(set(tuple(v) for v in first_amounts.values()))
    assert unique_sequences == len(seeds), (
        "Multiple seeds produced identical datasets — seeding is not working correctly!"
    )
