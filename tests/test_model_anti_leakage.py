"""Sprint 4 Tests — Anti-leakage invariants and hidden ground truth exclusion checks."""

import ast
import pathlib
import pytest

from recoveriq.domain.actions import Action
from recoveriq.model.dataset import CounterfactualDatasetBuilder, extract_observable_features
from recoveriq.model.preprocessing import FeaturePreprocessor
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def sim_partition():
    cfg = SimulationConfig(n_payments=100, n_customers=20, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    return partition_dataset(ds, train_fraction=0.75)


def test_hidden_fields_never_enter_feature_extractor(sim_partition):
    """extract_observable_features must never return hidden simulation ground truth."""
    for record in sim_partition.train_observable:
        features = extract_observable_features(record)
        assert "true_probability" not in features
        assert "latent_recoverability_profile" not in features
        assert "action_base_probabilities" not in features


def test_hidden_fields_never_enter_counterfactual_dataset(sim_partition):
    """CounterfactualDataset training examples must never contain hidden ground-truth fields."""
    train_env = SimulationEnvironment(sim_partition.train_ground_truth, seed=42)
    builder = CounterfactualDatasetBuilder()
    cf_dataset = builder.build_dataset(sim_partition.train_observable, train_env, seed=42)

    for ex in cf_dataset.examples:
        assert "true_probability" not in ex.features
        assert "latent_recoverability_profile" not in ex.features
        assert "action_base_probabilities" not in ex.features
        assert hasattr(ex, "label")
        assert not hasattr(ex, "true_probability")


def test_preprocessor_feature_vocabulary_contains_no_hidden_names():
    """FeaturePreprocessor feature_names must strictly consist of observable categorical and numerical features."""
    p = FeaturePreprocessor()
    for name in p.feature_names:
        assert "true_prob" not in name.lower()
        assert "latent" not in name.lower()
        assert "ground_truth" not in name.lower()
        assert "profile" not in name.lower()


def test_training_uses_strictly_train_partition_payment_ids(sim_partition):
    """Payment IDs present in the training counterfactual dataset must have zero overlap with test partition."""
    train_env = SimulationEnvironment(sim_partition.train_ground_truth, seed=42)
    builder = CounterfactualDatasetBuilder()
    cf_dataset = builder.build_dataset(sim_partition.train_observable, train_env, seed=42)

    train_pids = {ex.payment_id for ex in cf_dataset.examples}
    test_pids = set(sim_partition.test_payment_ids())

    overlap = train_pids & test_pids
    assert len(overlap) == 0, f"Leakage detected: {len(overlap)} test payments appeared in training!"


def test_ast_scan_no_ground_truth_internal_table_imports_in_model_package():
    """Verify via AST that recoveriq.model does NOT import GROUND_TRUTH_RECOVERY_TABLE or _PROFILE_WEIGHTS."""
    model_dir = pathlib.Path(__file__).parent.parent / "src" / "recoveriq" / "model"

    forbidden_symbols = {"GROUND_TRUTH_RECOVERY_TABLE", "_PROFILE_WEIGHTS", "RecoverabilityProfile"}

    for py_file in model_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in forbidden_symbols, (
                        f"{py_file.name} line {node.lineno}: forbidden import of {alias.name}"
                    )
