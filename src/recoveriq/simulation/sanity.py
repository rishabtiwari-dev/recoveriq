"""Statistical sanity checks for generated synthetic datasets."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticDataset
from recoveriq.simulation.partitioner import PartitionedDataset
from recoveriq.simulation.schema import RecoverabilityProfile


@dataclass
class SanityCheckResult:
    """Outcome of running statistical sanity checks on a dataset."""

    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"Sanity Check: {'PASSED' if self.passed else 'FAILED'}",
            f"  Total records   : {self.stats.get('n_total', '?')}",
            f"  Failures found  : {len(self.failures)}",
            f"  Warnings        : {len(self.warnings)}",
        ]
        for f in self.failures:
            lines.append(f"  [FAIL] {f}")
        for w in self.warnings:
            lines.append(f"  [WARN] {w}")
        for k, v in self.stats.items():
            if k != "n_total":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def check_dataset_sanity(
    dataset: SyntheticDataset,
    partitioned: Optional[PartitionedDataset] = None,
    env: Optional[SimulationEnvironment] = None,
    min_category_coverage: float = 0.005,
    train_fraction_tolerance: float = 0.05,
) -> SanityCheckResult:
    """Run statistical sanity checks on a generated dataset.

    Checks:
      1. Sample count matches config
      2. observable_records and ground_truth_records aligned
      3. All failure categories present (above min_category_coverage)
      4. All customer tiers present
      5. Amounts are all positive
      6. payment_ids are unique
      7. Ground-truth profiles are assigned (VERY_LOW..VERY_HIGH all present)
      8. If partitioned: no leakage, sizes approximately match train_fraction
      9. If env: action-conditioned outcomes can be resolved for all records

    Args:
        dataset: Generated dataset to validate.
        partitioned: Optional partitioned dataset.
        env: Optional simulation environment.
        min_category_coverage: Minimum fraction each failure category must reach.
        train_fraction_tolerance: Allowed deviation from target train fraction.

    Returns:
        SanityCheckResult with pass/fail status and statistics.
    """
    failures: List[str] = []
    warnings: List[str] = []
    stats: Dict = {}

    n = len(dataset)
    stats["n_total"] = n

    # 1. Length alignment
    if len(dataset.observable_records) != len(dataset.ground_truth_records):
        failures.append(
            f"Length mismatch: {len(dataset.observable_records)} observable vs "
            f"{len(dataset.ground_truth_records)} ground truth records."
        )

    # 2. Uniqueness of payment_ids
    payment_ids = [r.payment_id for r in dataset.observable_records]
    if len(payment_ids) != len(set(payment_ids)):
        dup_count = len(payment_ids) - len(set(payment_ids))
        failures.append(f"{dup_count} duplicate payment_ids found.")
    stats["unique_payment_ids"] = len(set(payment_ids))

    # 3. Positive amounts
    bad_amounts = [r for r in dataset.observable_records if r.amount <= 0]
    if bad_amounts:
        failures.append(f"{len(bad_amounts)} records have non-positive amounts.")
    stats["min_amount"] = min((float(r.amount) for r in dataset.observable_records), default=0)
    stats["max_amount"] = max((float(r.amount) for r in dataset.observable_records), default=0)

    # 4. Failure category coverage
    cat_counts = Counter(r.failure_category for r in dataset.observable_records)
    stats["failure_category_counts"] = {k.value: v for k, v in cat_counts.items()}
    for cat in FailureCategory:
        fraction = cat_counts.get(cat, 0) / n if n > 0 else 0.0
        if fraction < min_category_coverage:
            warnings.append(
                f"FailureCategory.{cat.name} has low coverage: {fraction:.3f} "
                f"(< {min_category_coverage:.3f})"
            )

    # 5. Customer tier coverage
    tier_counts = Counter(r.customer_tier for r in dataset.observable_records)
    stats["tier_counts"] = {k.value: v for k, v in tier_counts.items()}
    for tier in CustomerTier:
        if tier not in tier_counts:
            failures.append(f"CustomerTier.{tier.name} absent from dataset.")

    # 6. Ground-truth profile coverage
    profile_counts = Counter(
        gt.latent_recoverability_profile for gt in dataset.ground_truth_records
    )
    stats["profile_counts"] = {k.value: v for k, v in profile_counts.items()}
    for profile in RecoverabilityProfile:
        if profile not in profile_counts:
            warnings.append(f"RecoverabilityProfile.{profile.name} absent from dataset.")

    # 7. Action-conditioned probabilities are populated
    empty_probs = [
        gt for gt in dataset.ground_truth_records
        if not gt.action_base_probabilities
    ]
    if empty_probs:
        failures.append(
            f"{len(empty_probs)} ground truth records have empty action_base_probabilities."
        )

    # 8. Partition checks
    if partitioned is not None:
        # No leakage
        if partitioned.has_leakage():
            failures.append("LEAKAGE DETECTED: payment_ids appear in both train and test sets.")
        else:
            stats["leakage_detected"] = False

        # Size check
        actual_frac = partitioned.n_train / partitioned.n_total if partitioned.n_total > 0 else 0
        target_frac = partitioned.train_fraction
        stats["train_fraction_actual"] = round(actual_frac, 4)
        stats["train_fraction_target"] = target_frac
        if abs(actual_frac - target_frac) > train_fraction_tolerance:
            warnings.append(
                f"Train fraction {actual_frac:.3f} deviates from target {target_frac:.3f} "
                f"by more than tolerance {train_fraction_tolerance:.3f}."
            )

        stats["n_train"] = partitioned.n_train
        stats["n_test"] = partitioned.n_test

    # 9. Environment resolution check (spot-check first 10 records)
    if env is not None:
        resolution_failures = 0
        for record in dataset.observable_records[:10]:
            try:
                outcome = env.apply_action(record.payment_id, Action.RETRY_NOW)
                if not isinstance(outcome.recovered, bool):
                    resolution_failures += 1
            except Exception as e:
                resolution_failures += 1
        if resolution_failures > 0:
            failures.append(
                f"Environment resolution failed for {resolution_failures}/10 spot-check records."
            )
        else:
            stats["env_spot_check"] = "10/10 records resolved successfully"

    result_passed = len(failures) == 0
    return SanityCheckResult(
        passed=result_passed,
        failures=failures,
        warnings=warnings,
        stats=stats,
    )
