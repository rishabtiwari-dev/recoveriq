"""RecoverIQ end-to-end evaluation harness package."""

from recoveriq.evaluation.model_error import (
    ALL_DISTRIBUTION_SHIFT_CONDITIONS,
    ALL_MODEL_ERROR_CONDITIONS,
    DistributionShiftCondition,
    ModelErrorCondition,
    PerturbedProbabilityModel,
    apply_distribution_shift,
    get_perturbation_description,
)
from recoveriq.evaluation.model_free_policy import (
    FittedQIterationPolicy,
    ModelFreeRecoverIQStrategy,
    train_model_free_policy,
)
from recoveriq.evaluation.hybrid_policy import (
    HybridActionEvaluation,
    HybridDecision,
    HybridRecoverIQStrategy,
    HybridRegime,
    UncertaintyEstimator,
)
from recoveriq.evaluation.demo_data import (
    ATTEMPT_3_ACTION_DISTRIBUTION,
    BASELINE_BENCHMARK_M0_D0,
    DEGRADATION_M0_TO_M3,
    DISTRIBUTION_SHIFT_RESULTS,
    MODEL_ERROR_RESULTS,
    PAIRED_CRN_STATISTICS,
    RESEARCH_HYPOTHESES_VERDICTS,
)
from recoveriq.evaluation.demo_engine import DemoEngine

from recoveriq.evaluation.ablation_strategies import (
    GreedyProbabilitySelector,
    RecoverIQCtxAblationStrategy,
    RecoverIQNoEconStrategy,
)
from recoveriq.evaluation.metrics import (
    MultiSeedBenchmarkReport,
    MultiSeedStrategyMetrics,
    PaymentEvaluationRecord,
    StrategyMetrics,
)
from recoveriq.evaluation.bellman_policy import (
    BellmanActionEvaluation,
    BellmanDecision,
    BellmanRecoverIQStrategy,
)
from recoveriq.evaluation.robustness import (
    SPRINT12_EXPANDED_SEEDS,
    BreakEvenDiagnostic,
    PairedComparisonResult,
    ValueStratumResult,
    calculate_human_ops_valuation_sweep,
    compute_break_even_diagnostic,
    compute_paired_crn_differences,
    stratify_payments_by_value,
)
from recoveriq.evaluation.runner import EvaluationRunner
from recoveriq.evaluation.sequential_policy import TieredRecoverIQStrategy
from recoveriq.evaluation.strategies import (
    FixedRetryStrategy,
    RecoveryStrategy,
    RecoverIQStrategy,
    RuleBasedStrategy,
)
from recoveriq.evaluation.trajectory import (
    AlwaysStopStrategy,
    MultiSeedTrajectoryMetrics,
    TrajectoryEpisode,
    TrajectoryEvaluationRunner,
    TrajectoryStep,
    TrajectoryStrategyMetrics,
)

__all__ = [
    "RecoveryStrategy",
    "FixedRetryStrategy",
    "RuleBasedStrategy",
    "RecoverIQStrategy",
    "TieredRecoverIQStrategy",
    "BellmanRecoverIQStrategy",
    "BellmanActionEvaluation",
    "BellmanDecision",
    "AlwaysStopStrategy",
    "GreedyProbabilitySelector",
    "RecoverIQCtxAblationStrategy",
    "RecoverIQNoEconStrategy",
    "EvaluationRunner",
    "TrajectoryEvaluationRunner",
    "TrajectoryStep",
    "TrajectoryEpisode",
    "TrajectoryStrategyMetrics",
    "MultiSeedTrajectoryMetrics",
    "PaymentEvaluationRecord",
    "StrategyMetrics",
    "MultiSeedStrategyMetrics",
    "MultiSeedBenchmarkReport",
    "SPRINT12_EXPANDED_SEEDS",
    "calculate_human_ops_valuation_sweep",
    "stratify_payments_by_value",
    "compute_paired_crn_differences",
    "compute_break_even_diagnostic",
    "ValueStratumResult",
    "PairedComparisonResult",
    "BreakEvenDiagnostic",
    # Sprint 14
    "ModelErrorCondition",
    "ALL_MODEL_ERROR_CONDITIONS",
    "DistributionShiftCondition",
    "ALL_DISTRIBUTION_SHIFT_CONDITIONS",
    "PerturbedProbabilityModel",
    "apply_distribution_shift",
    "get_perturbation_description",
    "FittedQIterationPolicy",
    "ModelFreeRecoverIQStrategy",
    "train_model_free_policy",
    # Sprint 15
    "HybridRecoverIQStrategy",
    "HybridRegime",
    "HybridActionEvaluation",
    "HybridDecision",
    "UncertaintyEstimator",
    # Sprint 16
    "BASELINE_BENCHMARK_M0_D0",
    "MODEL_ERROR_RESULTS",
    "DEGRADATION_M0_TO_M3",
    "DISTRIBUTION_SHIFT_RESULTS",
    "PAIRED_CRN_STATISTICS",
    "ATTEMPT_3_ACTION_DISTRIBUTION",
    "RESEARCH_HYPOTHESES_VERDICTS",
    "DemoEngine",
]
