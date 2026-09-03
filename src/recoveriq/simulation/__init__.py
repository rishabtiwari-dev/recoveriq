"""Simulation package for RecoverIQ Sprint 2."""

from recoveriq.simulation.config import SimulationConfig
from recoveriq.simulation.environment import ActionOutcome, SimulationEnvironment
from recoveriq.simulation.generator import SyntheticDataset, SyntheticPaymentGenerator
from recoveriq.simulation.ground_truth import assign_ground_truth, resolve_outcome
from recoveriq.simulation.partitioner import PartitionedDataset, partition_dataset
from recoveriq.simulation.sanity import SanityCheckResult, check_dataset_sanity
from recoveriq.simulation.schema import (
    GroundTruthRecord,
    RecoverabilityProfile,
    SyntheticPaymentRecord,
)

__all__ = [
    "ActionOutcome",
    "GroundTruthRecord",
    "PartitionedDataset",
    "RecoverabilityProfile",
    "SanityCheckResult",
    "SimulationConfig",
    "SimulationEnvironment",
    "SyntheticDataset",
    "SyntheticPaymentGenerator",
    "SyntheticPaymentRecord",
    "assign_ground_truth",
    "check_dataset_sanity",
    "partition_dataset",
    "resolve_outcome",
]
