# RecoverIQ Sprint 16 — System Architecture & Demonstration Layer

## 1. System Overview & Component Inventory

Sprint 16 adds an interactive, scientifically grounded demonstration and presentation layer (`app/demo.py`) to the established RecoverIQ research repository.

### Layer Architecture:
```
┌────────────────────────────────────────────────────────────────────────┐
│             DEMONSTRATION & PRESENTATION LAYER (Sprint 16)             │
│   app/demo.py  •  Interactive Walkthrough  •  Policy Comparison        │
│   Research Experiment Dashboards (M0–M3, D0–D3, Arbitration Weights)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     EVALUATION & RESEARCH ENGINE                       │
│  src/recoveriq/evaluation/                                             │
│  ├── trajectory.py        (Sequential Trajectory Evaluation Runner)    │
│  ├── strategies.py        (Fixed-Retry, Rule-Based, RecoverIQ-Uncon)   │
│  ├── sequential_policy.py (RecoverIQ-Tiered, Human-Ops Valuation)      │
│  ├── bellman_policy.py    (RecoverIQ-Bellman DP & Option Value)        │
│  ├── model_free_policy.py (RecoverIQ-ModelFree Fitted Q-Iteration)     │
│  ├── hybrid_policy.py     (RecoverIQ-Hybrid Uncertainty Arbitration)   │
│  ├── model_error.py       (M0–M3 Misspecification, D0–D3 Shifts)      │
│  └── robustness.py        (Paired CRN, Bootstrap CIs, Stratification)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   FROZEN CORE RESEARCH FOUNDATION                      │
│  domain/ • policy/ • ai/ • economics/ • model/ • simulation/           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Audit Findings

| Component Area | Existing Implementation | Sprint 16 Integration Role |
|:---|:---|:---|
| **Simulation** | `SyntheticPaymentGenerator`, `SimulationEnvironment`, `partition_dataset` | Generates realistic, reproducible observable payment records and evaluates candidate actions. |
| **Model Training** | `ModelTrainer` fits logistic regression on observable training partitions | Provides trained probability models for Bellman DP and evaluation baselines. |
| **Rule-Based** | `RuleBasedStrategy` maps failure categories deterministically | Benchmark heuristic reflecting standard industry practice. |
| **Bellman DP** | `BellmanRecoverIQStrategy` (Sprint 13) | Computes $Q_t(a, x) = \text{Immediate\_EV} + \text{Future\_Option\_Value}$ across finite horizon $H=3$. |
| **Model-Free Q** | `ModelFreeRecoverIQStrategy` (Sprint 14) | Fitted Q-Iteration tabular policy learned from offline exploration trajectories; immune to probability model misspecification. |
| **Hybrid** | `HybridRecoverIQStrategy` (Sprint 15) | Arbitrates between Bellman and ModelFree Q-values via Equal-Weight, Fixed-Weight, or Uncertainty-Aware weighting. |
| **Model Perturbation** | `PerturbedProbabilityModel` (M0–M3) | Deterministically distorts probability estimates to evaluate policy robustness. |
| **Distribution Shift** | `apply_distribution_shift` (D0–D3) | Covariate transformations (2x value shift, +1 customer tier) on evaluation sets. |

---

## 3. Strict Safety & Anti-Leakage Boundaries

1. **Frozen Production Boundaries:** `SPEC.md`, `src/recoveriq/domain/*`, `policy/*`, `ai/*`, `economics/*`, `model/*`, and `simulation/*` remain untouched.
2. **Observable State Isolation:** The demonstration UI exposes only decision-time context (`payment_amount`, `failure_category`, `customer_tier`, `attempt_count`, `raw_error_code`, `payment_method`). Hidden ground truth (`GroundTruthRecord`, actual success probabilities) is strictly isolated within the evaluation environment.
3. **No External API Dependencies:** The demo runs 100% locally on synthetic simulation environments with zero external network or payment gateway dependencies.
4. **Authentic Results Data:** All dashboard charts and statistical confidence intervals display verified research outputs from 20-seed Common Random Numbers experiments.
