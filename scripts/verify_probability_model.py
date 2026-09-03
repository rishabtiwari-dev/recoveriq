"""RecoverIQ Sprint 4 — Probability Model Verification Script.

Verifies:
1. Counterfactual dataset generation (N payments x 6 actions).
2. Training of 6 action models with STOP=0.0 invariant.
3. Artifact save/load round-trip & fail-fast on missing artifact.
4. Inference adhering to RecoveryProbabilityModel protocol.
5. Zero leakage of hidden ground-truth fields.
6. Held-out test set evaluation metrics.
"""

import sys
import tempfile
from decimal import Decimal
from pathlib import Path

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.model.dataset import CounterfactualDatasetBuilder, extract_observable_features
from recoveriq.model.evaluation import ModelEvaluator
from recoveriq.model.probability import RecoveryProbabilityModel
from recoveriq.model.trained_model import TrainedRecoveryProbabilityModel
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


def main() -> int:
    print("=" * 70)
    print("RecoverIQ Sprint 4 — Statistical Recovery Probability Model Verification")
    print("=" * 70)

    # 1. Generate synthetic dataset
    print("\n1. Generating synthetic dataset (N=1000, seed=42)...")
    cfg = SimulationConfig(n_payments=1000, n_customers=200, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    print(f"   Total: {len(ds)} | Train: {partitioned.n_train} | Test: {partitioned.n_test}")

    # 2. Build counterfactual dataset on training partition
    print("\n2. Building full factorial counterfactual dataset on train split...")
    train_env = SimulationEnvironment(partitioned.train_ground_truth, seed=42)
    builder = CounterfactualDatasetBuilder()
    cf_dataset = builder.build_dataset(partitioned.train_observable, train_env, seed=42)
    expected_rows = partitioned.n_train * len(Action)
    assert len(cf_dataset) == expected_rows, f"Expected {expected_rows} rows, got {len(cf_dataset)}"
    print(f"   [OK] Constructed {len(cf_dataset)} counterfactual observations ({partitioned.n_train} x 6 actions).")

    # 3. Verify anti-leakage in feature extraction
    print("\n3. Verifying anti-leakage invariants...")
    sample_feat = extract_observable_features(partitioned.train_observable[0])
    forbidden = ["true_probability", "latent_recoverability_profile", "action_base_probabilities"]
    for f in forbidden:
        assert f not in sample_feat, f"Forbidden field '{f}' leaked into features!"
    print("   [OK] Zero hidden ground-truth fields in observable feature representations.")

    # 4. Train 6-action model
    print("\n4. Training independent action logistic regressions...")
    trainer = ModelTrainer(c_regularization=1.0, random_state=42, model_version="logistic-regression-v1")
    model = trainer.train(partitioned.train_observable, train_env)
    assert isinstance(model, RecoveryProbabilityModel), "Model must implement RecoveryProbabilityModel protocol"
    assert len(model.action_models) == 6, "All 6 actions must have an action model"
    print("   [OK] Successfully trained 6 action models.")

    # 5. Verify STOP invariant
    print("\n5. Verifying STOP = 0.0 domain invariant...")
    test_context = PaymentContext(
        payment_id="pay_test_001",
        customer_id="cust_001",
        customer_tier=CustomerTier.VIP,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        attempt_count=1,
    )
    probs = model.estimate_probabilities(test_context)
    assert probs[Action.STOP].probability == Decimal("0.00"), f"STOP probability must be 0.00, got {probs[Action.STOP].probability}"
    print("   [OK] Action.STOP returns Decimal('0.00') strictly.")

    # 6. Verify artifact save/load and fail-fast
    print("\n6. Verifying artifact persistence & fail-fast behavior...")
    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = Path(tmpdir) / "model_artifact.json"
        model.save(art_path)
        assert art_path.exists(), "Artifact file must exist after save"

        loaded_model = TrainedRecoveryProbabilityModel.load(art_path)
        probs_loaded = loaded_model.estimate_probabilities(test_context)
        for act in Action:
            assert probs[act].probability == probs_loaded[act].probability, f"Mismatch on {act.value}"
        print("   [OK] Save and load round-trip succeeded with identical predictions.")

        # Fail-fast test
        missing_path = Path(tmpdir) / "does_not_exist.json"
        try:
            TrainedRecoveryProbabilityModel.load(missing_path)
            raise AssertionError("Should have raised FileNotFoundError for missing artifact")
        except FileNotFoundError:
            print("   [OK] Missing artifact properly raised FileNotFoundError (no silent fallback).")

    # 7. Evaluate on held-out test partition
    print("\n7. Evaluating on held-out test partition...")
    test_env = SimulationEnvironment(partitioned.test_ground_truth, seed=42)
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, partitioned.test_observable, test_env)
    print("\n" + report.summary_table())
    print(f"\nOverall Mean Brier Score: {report.overall_mean_brier_score:.4f}")
    print(f"Overall Mean Log Loss:    {report.overall_mean_log_loss:.4f}")

    print("\n" + "=" * 70)
    print("ALL SPRINT 4 PROBABILITY MODEL CHECKS PASSED.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
