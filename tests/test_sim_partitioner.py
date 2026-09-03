"""Sprint 2 Tests — Anti-leakage and train/test partition integrity."""

import pytest
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import PartitionedDataset, partition_dataset
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


@pytest.fixture(scope="module")
def partitioned():
    cfg = SimulationConfig(n_payments=1000, n_customers=200)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


def test_no_payment_id_leakage(partitioned):
    """No payment_id must appear in both train and test splits."""
    assert not partitioned.has_leakage(), (
        "LEAKAGE DETECTED: payment_ids overlap between train and test partitions!"
    )


def test_train_plus_test_equals_total(partitioned):
    """Train + test records must equal total records generated."""
    assert partitioned.n_train + partitioned.n_test == partitioned.n_total


def test_train_fraction_approximately_correct(partitioned):
    """Actual train fraction should be close to 0.75 (within 5%)."""
    actual = partitioned.n_train / partitioned.n_total
    assert abs(actual - 0.75) < 0.05, (
        f"Train fraction {actual:.4f} deviates too far from 0.75"
    )


def test_observable_and_ground_truth_aligned_in_train(partitioned):
    """Train split: observable and ground_truth records must be aligned."""
    for obs, gt in zip(partitioned.train_observable, partitioned.train_ground_truth):
        assert obs.payment_id == gt.payment_id


def test_observable_and_ground_truth_aligned_in_test(partitioned):
    """Test split: observable and ground_truth records must be aligned."""
    for obs, gt in zip(partitioned.test_observable, partitioned.test_ground_truth):
        assert obs.payment_id == gt.payment_id


def test_ground_truth_absent_from_observable_fields(partitioned):
    """SyntheticPaymentRecord must not have latent_recoverability_profile attribute."""
    for record in partitioned.train_observable + partitioned.test_observable:
        assert not hasattr(record, "latent_recoverability_profile"), (
            "Observable record must not expose hidden ground-truth fields!"
        )
        assert not hasattr(record, "action_base_probabilities"), (
            "Observable record must not expose action_base_probabilities!"
        )


def test_partition_is_deterministic_across_calls():
    """Re-partitioning the same dataset must produce identical splits."""
    cfg = SimulationConfig(n_payments=400, n_customers=100)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=2024)

    p1 = partition_dataset(ds, train_fraction=0.75)
    p2 = partition_dataset(ds, train_fraction=0.75)

    ids1 = [r.payment_id for r in p1.train_observable]
    ids2 = [r.payment_id for r in p2.train_observable]
    assert ids1 == ids2, "Partitioning is not deterministic!"


def test_different_train_fractions_produce_different_splits():
    """Different train_fraction values should produce differently-sized splits."""
    cfg = SimulationConfig(n_payments=400, n_customers=100)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)

    p_60 = partition_dataset(ds, train_fraction=0.60)
    p_80 = partition_dataset(ds, train_fraction=0.80)

    assert p_60.n_train < p_80.n_train


def test_invalid_train_fraction_raises():
    """Partition should reject invalid train_fraction values."""
    cfg = SimulationConfig(n_payments=100, n_customers=20)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)

    with pytest.raises(ValueError):
        partition_dataset(ds, train_fraction=0.0)

    with pytest.raises(ValueError):
        partition_dataset(ds, train_fraction=1.0)
