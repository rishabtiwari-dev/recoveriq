"""Sprint 2 verification script: multi-seed simulation run with sanity checks.

Runs generation, partitioning, environment construction, and statistical sanity
checks across all required seeds (minimum 3, preferred 5).

Usage:
    python scripts/verify_simulation.py
"""

import sys
from recoveriq.domain.actions import Action
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset
from recoveriq.simulation.sanity import check_dataset_sanity


SEEDS = [42, 100, 777, 999, 2024]  # 5 seeds (SPEC preferred target)
N_PAYMENTS = 2000


def run_seed(seed: int, cfg: SimulationConfig) -> bool:
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=seed)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    env = SimulationEnvironment(ds.ground_truth_records, seed=seed)

    result = check_dataset_sanity(
        dataset=ds,
        partitioned=partitioned,
        env=env,
    )

    # Compute action-conditioned recovery rates on test set
    test_ids = [r.payment_id for r in partitioned.test_observable]
    rates = {}
    for action in Action:
        outcomes = env.batch_apply_action(test_ids, action)
        rates[action] = sum(o.recovered for o in outcomes) / len(outcomes) if outcomes else 0.0

    print(f"\n{'='*60}")
    print(f"Seed: {seed} | Total: {len(ds)} | Train: {partitioned.n_train} | Test: {partitioned.n_test}")
    print(f"Sanity: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Leakage: {result.stats.get('leakage_detected', 'N/A')}")
    print(f"Profile distribution: {result.stats.get('profile_counts', {})}")
    print("Action-conditioned recovery rates (test set):")
    for action, rate in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {action.value:<15} : {rate:.4f}")

    if result.failures:
        print("FAILURES:")
        for f in result.failures:
            print(f"  [FAIL] {f}")
    if result.warnings:
        for w in result.warnings:
            print(f"  [WARN] {w}")

    return result.passed


def main() -> int:
    cfg = SimulationConfig(n_payments=N_PAYMENTS, n_customers=500)
    all_passed = True

    print("RecoverIQ Sprint 2 — Simulation Verification")
    print(f"Seeds: {SEEDS}  |  N_payments per seed: {N_PAYMENTS}")

    for seed in SEEDS:
        passed = run_seed(seed, cfg)
        if not passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"ALL {len(SEEDS)} SEEDS PASSED. Sprint 2 simulation verified.")
    else:
        print("ONE OR MORE SEEDS FAILED. See details above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
