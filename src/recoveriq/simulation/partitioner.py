"""Deterministic, payment-level train/test partitioner.

ANTI-LEAKAGE INVARIANT:
Partitioning is done at the PAYMENT level (by payment_id), NOT at the
record/row level. This ensures that if a customer appears in multiple records,
all their payments are assigned to the SAME partition — preventing the model
from seeing the same customer's patterns in both train and test.

For the current dataset (each payment_id is unique), payment-level splitting
is equivalent to ensuring reproducible, seed-deterministic splits with no
cross-contamination between train and test.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple

from recoveriq.simulation.generator import SyntheticDataset
from recoveriq.simulation.schema import GroundTruthRecord, SyntheticPaymentRecord


def _partition_index(payment_id: str, train_fraction: float) -> str:
    """Deterministically assign a payment_id to 'train' or 'test'.

    Uses a hash of the payment_id to ensure reproducibility regardless of
    generation order. The same payment_id always maps to the same partition.
    """
    digest = hashlib.md5(payment_id.encode("utf-8")).hexdigest()
    # Take first 8 hex chars (32 bits) and normalize to [0, 1)
    hash_val = int(digest[:8], 16) / (16 ** 8)
    return "train" if hash_val < train_fraction else "test"


@dataclass
class PartitionedDataset:
    """A train/test split of a SyntheticDataset with strict payment-level separation."""

    train_observable: List[SyntheticPaymentRecord] = field(default_factory=list)
    train_ground_truth: List[GroundTruthRecord] = field(default_factory=list)
    test_observable: List[SyntheticPaymentRecord] = field(default_factory=list)
    test_ground_truth: List[GroundTruthRecord] = field(default_factory=list)
    seed: int = 42
    train_fraction: float = 0.75

    @property
    def n_train(self) -> int:
        return len(self.train_observable)

    @property
    def n_test(self) -> int:
        return len(self.test_observable)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_test

    def train_payment_ids(self) -> List[str]:
        return [r.payment_id for r in self.train_observable]

    def test_payment_ids(self) -> List[str]:
        return [r.payment_id for r in self.test_observable]

    def has_leakage(self) -> bool:
        """Return True if any payment_id appears in both train and test."""
        train_ids = set(self.train_payment_ids())
        test_ids = set(self.test_payment_ids())
        return bool(train_ids & test_ids)


def partition_dataset(
    dataset: SyntheticDataset,
    train_fraction: float = 0.75,
) -> PartitionedDataset:
    """Split a SyntheticDataset into train and test partitions at payment level.

    Args:
        dataset: Generated synthetic dataset.
        train_fraction: Fraction of payments to include in training set.

    Returns:
        PartitionedDataset with strict payment-level separation.
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    train_obs: List[SyntheticPaymentRecord] = []
    train_gt: List[GroundTruthRecord] = []
    test_obs: List[SyntheticPaymentRecord] = []
    test_gt: List[GroundTruthRecord] = []

    for obs, gt in zip(dataset.observable_records, dataset.ground_truth_records):
        partition = _partition_index(obs.payment_id, train_fraction)
        if partition == "train":
            train_obs.append(obs)
            train_gt.append(gt)
        else:
            test_obs.append(obs)
            test_gt.append(gt)

    return PartitionedDataset(
        train_observable=train_obs,
        train_ground_truth=train_gt,
        test_observable=test_obs,
        test_ground_truth=test_gt,
        seed=dataset.seed,
        train_fraction=train_fraction,
    )
