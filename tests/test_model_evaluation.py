"""Sprint 4 Tests — Model evaluation, metrics calculation, and ground-truth comparison diagnostics."""

import pytest

from recoveriq.domain.actions import Action
from recoveriq.model.evaluation import ModelEvaluationReport, ModelEvaluator
from recoveriq.model.trainer import ModelTrainer
from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import SimulationEnvironment
from recoveriq.simulation.generator import SyntheticPaymentGenerator
from recoveriq.simulation.partitioner import partition_dataset


@pytest.fixture
def trained_model_and_test_data():
    cfg = SimulationConfig(n_payments=200, n_customers=40, default_seed=42)
    gen = SyntheticPaymentGenerator(cfg)
    ds = gen.generate(seed=42)
    partitioned = partition_dataset(ds, train_fraction=0.75)

    train_env = SimulationEnvironment(partitioned.train_ground_truth, seed=42)
    trainer = ModelTrainer(c_regularization=1.0, random_state=42)
    model = trainer.train(partitioned.train_observable, train_env)

    test_env = SimulationEnvironment(partitioned.test_ground_truth, seed=42)
    return model, partitioned.test_observable, test_env


def test_evaluator_computes_metrics_for_all_actions(trained_model_and_test_data):
    """ModelEvaluator must produce valid metrics across all 6 candidate actions."""
    model, test_obs, test_env = trained_model_and_test_data
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, test_obs, test_env)

    assert isinstance(report, ModelEvaluationReport)
    assert len(report.metrics_per_action) == 6

    for action in Action:
        m = report.metrics_per_action[action]
        assert m.sample_count == len(test_obs)
        assert 0.0 <= m.empirical_recovery_rate <= 1.0
        assert 0.0 <= m.mean_predicted_probability <= 1.0
        assert 0.0 <= m.brier_score <= 1.0
        assert m.log_loss_val >= 0.0
        assert m.mae_vs_ground_truth >= 0.0


def test_stop_action_metrics_are_strictly_zero(trained_model_and_test_data):
    """STOP action has empirical rate 0, mean predicted 0, brier 0, log loss 0, and MAE 0."""
    model, test_obs, test_env = trained_model_and_test_data
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, test_obs, test_env)

    stop_m = report.metrics_per_action[Action.STOP]
    assert stop_m.empirical_recovery_rate == 0.0
    assert stop_m.mean_predicted_probability == 0.0
    assert stop_m.brier_score == 0.0
    assert stop_m.log_loss_val == 0.0
    assert stop_m.roc_auc is None
    assert stop_m.mae_vs_ground_truth == 0.0


def test_evaluator_summary_table(trained_model_and_test_data):
    """Summary table formatting must produce readable table with all actions."""
    model, test_obs, test_env = trained_model_and_test_data
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, test_obs, test_env)

    table_str = report.summary_table()
    for action in Action:
        assert action.value in table_str
    assert "Empirical" in table_str
    assert "Brier" in table_str
    assert "ROC-AUC" in table_str
