"""Sprint 4 Tests — Model persistence, artifact save/load roundtrip, and fail-fast guarantees."""

from decimal import Decimal
import tempfile
from pathlib import Path
import pytest

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory, FailureSeverity, PaymentContext, PaymentMethod
from recoveriq.model.trained_model import TrainedRecoveryProbabilityModel
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def trained_model():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)
    env = SimulationEnvironment(partitioned.train_ground_truth, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    return trainer.train(partitioned.train_observable, env)


def test_save_and_load_roundtrip(trained_model):
    """Saving and loading an artifact must preserve exact predictions and metadata."""
    ctx = PaymentContext(
        payment_id="pay_save_test",
        customer_id="cust_save_test",
        customer_tier=CustomerTier.PREMIUM,
        payment_method=PaymentMethod.UPI,
        raw_error_code="504",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
        attempt_count=1,
    )
    original_estimates = trained_model.estimate_probabilities(ctx)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = Path(tmpdir) / "test_artifact.json"
        trained_model.save(art_path)
        assert art_path.exists()

        loaded_model = TrainedRecoveryProbabilityModel.load(art_path)
        loaded_estimates = loaded_model.estimate_probabilities(ctx)

        assert loaded_model.model_version == trained_model.model_version
        assert loaded_model.metadata["n_training_payments"] == trained_model.metadata["n_training_payments"]

        for action in Action:
            assert loaded_estimates[action].probability == original_estimates[action].probability
            assert loaded_estimates[action].action == action


def test_load_missing_artifact_fails_fast():
    """Missing model artifact file must raise FileNotFoundError immediately."""
    missing_path = Path("this/path/does/not/exist/model.json")
    with pytest.raises(FileNotFoundError) as exc_info:
        TrainedRecoveryProbabilityModel.load(missing_path)

    assert "Failing fast" in str(exc_info.value)


def test_load_corrupted_json_fails_fast():
    """Corrupted non-JSON file must raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write("{this is not valid json")
        corrupted_path = Path(f.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            TrainedRecoveryProbabilityModel.load(corrupted_path)
        assert "Corrupt or unparseable" in str(exc_info.value)
    finally:
        corrupted_path.unlink(missing_ok=True)


def test_load_invalid_schema_fails_fast():
    """Valid JSON missing required model sections must raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write('{"model_version": "v1"}')
        invalid_path = Path(f.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            TrainedRecoveryProbabilityModel.load(invalid_path)
        assert "Invalid model artifact schema" in str(exc_info.value)
    finally:
        invalid_path.unlink(missing_ok=True)
