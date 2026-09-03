# RecoverIQ: Economic & Context-Aware Payment Recovery Engine

RecoverIQ is an experimental research prototype investigating whether a context-aware, economically optimized payment recovery engine can improve net recovered value over fixed-retry and rule-based strategies while enforcing deterministic payment-safety constraints.

---

## Current Status: Sprint 6 Complete — Component Attribution & Ablation Study (SPEC §17)

| Sprint | Status | Description |
|--------|--------|-------------|
| Sprint 1 | ✅ Locked | Foundation & Architecture — domain models, state machine, policy gate, audit, interfaces |
| Sprint 2 | ✅ Locked | Synthetic Data Generation & Simulation Environment — multi-seed reproducibility, zero leakage |
| Sprint 3 | ✅ Locked | LLM Context Integration — Gemini 3.8 Flash technical diagnostics, strict privacy & anti-authority boundaries |
| Sprint 4 | ✅ Locked | Statistical Recovery Probability Model — 6 independent action models, counterfactual dataset, STOP invariant |
| Sprint 5 | ✅ Complete | End-to-End Evaluation Harness — RecoverIQ vs Fixed-Retry vs Rule-Based, Common Random Numbers (CRN), Common Policy Gate |
| Sprint 6 | ✅ Complete | Component Attribution & Ablation Study — Context-Source Ablation (A1), Economic Engine Ablation (A2) |

---

## Sprint 6 Ablation Study (SPEC.md Section 17)

Sprint 6 isolates the individual contributions of RecoverIQ's core architectural components:

### 1. Ablation A1 — Context-Source Ablation (Generator-Oracle vs Rule-Based Extraction)
* **What it measures:** In Sprint 5, the full RecoverIQ system passed generator-assigned `failure_category` directly into `PaymentContext` (an oracle shortcut). A1 measures the effect of replacing this oracle context with context extracted via `RuleBasedContextExtractor` from raw error strings.
* **Important Research Clarification:** A1 does **NOT** measure "Gemini vs Rules". Because the Sprint 5 evaluation harness bypassed the AI layer, A1 measures **Generator-Oracle Context vs Rule-Based Keyword Extraction**.
* **Empirical Category Agreement:** Measured at **86.8% mean agreement** (13.2% divergence) across the 5 evaluation seeds. The divergence is structurally concentrated in `AUTHENTICATION_REJECTED -> UNKNOWN`, `NETWORK_TIMEOUT -> UNKNOWN`, and `INVALID_DETAILS -> UNKNOWN`.

### 2. Ablation A2 — Economic Engine Ablation (Greedy Probability vs EV Optimization)
* **What it measures:** Replaces Net Expected Value (EV) optimization ($\arg\max_a \text{EV}(a)$) with greedy probability maximization ($\arg\max_a \hat{P}(a)$ over non-STOP actions).
* **Behavioral Finding:** Greedy probability over-selects `ESCALATE` (61.7% vs 60.5%), incurring higher direct operational intervention costs (570.48 vs 560.55 INR, $\Delta = +9.93$ INR).

### Multi-Seed Ablation Comparison (Mean ± Std across 5 seeds: `[42, 100, 777, 999, 2024]`):

| Strategy | Net Recovered Value (NRV) | NRV / Payment | Recovery Rate | Direct Cost | Block Rate | Policy Violation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Retry** | 714,852.24 ± 117,275.74 | 2,803.63 ± 488.44 | 32.00% ± 2.47% | 33.90 ± 0.78 | 11.48% | **0.00% (PASSED)** |
| **Rule-Based** | 775,936.89 ± 121,714.84 | 3,045.08 ± 524.26 | 36.64% ± 1.37% | 48.55 ± 1.33 | 0.00% | **0.00% (PASSED)** |
| **RecoverIQ (Oracle Ctx)** | **983,030.03 ± 200,330.18** | **3,858.73 ± 833.93** | **46.42% ± 3.12%** | **560.55 ± 144.38** | **2.87%** | **0.00% (PASSED)** |
| **RecoverIQ-CtxAblation (A1)** | 981,951.50 ± 201,585.21 | 3,854.61 ± 838.93 | 46.19% ± 3.23% | 560.83 ± 150.32 | 2.94% | **0.00% (PASSED)** |
| **RecoverIQ-NoEcon (A2)** | 983,328.05 ± 200,502.34 | 3,859.91 ± 834.59 | 46.74% ± 3.20% | 570.48 ± 142.92 | 2.71% | **0.00% (PASSED)** |

### Component Attribution Deltas (relative to Full RecoverIQ):
* **Context-Source Contribution (Oracle vs Rule-Based):** $\Delta \text{NRV} = +1,078.54$ INR (+0.11%).
* **Economic Engine Contribution (EV Optimization vs Greedy Prob):** $\Delta \text{NRV} = -298.02$ INR (-0.03%), with greedy prob incurring +9.93 INR higher direct cost.
* **Primary RecoverIQ Lift:** Maintained at **+268,177.79 INR (+37.5%)** over Fixed-Retry and **+207,093.15 INR (+26.7%)** over Rule-Based baselines.

---

## Repository Structure

```
recoveriq/
├── SPEC.md                    # Authoritative system specification
├── README.md                  # Project documentation & status
├── pyproject.toml             # Project metadata & test configuration
├── .gitignore                 # Version control ignore patterns
├── .env.example               # Environment variables template
├── docs/
│   └── architecture.md        # Modular architecture design
├── scripts/
│   ├── verify_foundation.py   # Sprint 1 sanity verification script
│   ├── verify_simulation.py   # Sprint 2 multi-seed simulation verification script
│   ├── verify_ai_boundaries.py# Sprint 3 AI boundary & safety verification script
│   ├── verify_probability_model.py # Sprint 4 probability model verification script
│   ├── verify_evaluation.py   # Sprint 5 end-to-end evaluation harness verification script
│   └── verify_ablation.py     # Sprint 6 ablation study & attribution verification script
├── src/
│   └── recoveriq/
│       ├── domain/            # Core domain models, state machine, actions, events, idempotency
│       ├── context/           # Context ingestion & rule-based extractors
│       ├── ai/                # Sprint 3: Gemini 3.8 Flash context interpretation & guardrails
│       │   ├── gemini_context_layer.py # Production AIContextLayer with modern google-genai
│       │   ├── prompt_template.py      # Strict 4-field privacy prompt renderer
│       │   ├── schema_validator.py     # Anti-authority blacklist schema validator
│       │   └── context_layer.py        # Protocol & StubAIContextLayer
│       ├── model/             # Sprint 4: Statistical recovery probability model
│       │   ├── dataset.py              # Counterfactual dataset builder (N x 6 actions)
│       │   ├── preprocessing.py        # Deterministic one-hot & numerical standardizer
│       │   ├── logistic_regression.py  # Action-specific L2 logistic regression & STOP handler
│       │   ├── trained_model.py        # TrainedRecoveryProbabilityModel implementation
│       │   ├── trainer.py              # End-to-end model trainer
│       │   ├── evaluation.py           # Test set evaluation & diagnostic MAE metrics
│       │   └── probability.py          # Protocol & StubProbabilityModel
│       ├── economics/         # Expected value optimizer & cost/penalty models
│       ├── policy/            # Deterministic policy authorization gate & invariants
│       ├── executor/          # Action executor & idempotency enforcement
│       ├── audit/             # Structured immutable audit logging
│       ├── config/            # Strongly typed settings (costs, penalties, policies, LLM)
│       ├── simulation/        # Sprint 2: Synthetic data & simulation environment
│       │   ├── config.py      # Simulation configuration & distributions
│       │   ├── schema.py      # Observable record vs hidden ground-truth record (strict separation)
│       │   ├── ground_truth.py# Independent stochastic world model (no RecoverIQ logic)
│       │   ├── generator.py   # Reproducible synthetic payment generator
│       │   ├── partitioner.py # Payment-level train/test splitting (no leakage)
│       │   ├── environment.py # Action-conditioned outcome resolver
│       │   └── sanity.py      # Statistical sanity checks
│       ├── evaluation/        # Sprint 5 & 6: Evaluation & ablation harness
│       │   ├── strategies.py  # RecoverIQ, Fixed-Retry, and Rule-Based strategy definitions
│       │   ├── ablation_strategies.py # Sprint 6: A1 Context-Ablation and A2 No-Econ strategies
│       │   ├── metrics.py     # NRV, gross revenue, recovery rate, and multi-seed aggregation
│       │   └── runner.py      # CRN benchmark & ablation runner
│       └── engine.py          # Unified pipeline coordinator
└── tests/                     # Unit & contract tests (169 tests across 29 suites)
    ├── test_actions.py
    ├── test_state_machine.py
    ├── test_models.py
    ├── test_config.py
    ├── test_idempotency.py
    ├── test_economics.py
    ├── test_policy_gate.py
    ├── test_audit.py
    ├── test_pipeline_contract.py
    ├── test_sim_reproducibility.py
    ├── test_sim_schema.py
    ├── test_sim_distributions.py
    ├── test_sim_partitioner.py
    ├── test_sim_outcomes.py
    ├── test_sim_sanity.py
    ├── test_ai_prompt_template.py
    ├── test_ai_schema_validator.py
    ├── test_ai_gemini_layer.py
    ├── test_ai_boundary_contract.py
    ├── test_ai_integration.py
    ├── test_model_dataset.py
    ├── test_model_preprocessing.py
    ├── test_model_training.py
    ├── test_model_persistence.py
    ├── test_model_anti_leakage.py
    ├── test_model_evaluation.py
    ├── test_model_engine_integration.py
    ├── test_eval_strategies.py
    ├── test_eval_metrics.py
    ├── test_eval_common_random_numbers.py
    ├── test_eval_common_policy_gate.py
    ├── test_eval_held_out_partition.py
    ├── test_ablation_strategies.py
    ├── test_ablation_confounder_controls.py
    ├── test_resilience_harness.py
    └── test_failure_resilience.py
```

---

## Installation & Setup

### Prerequisites
* Python 3.11+ (Tested on Python 3.13)
* `pytest` for running test suites

### Development Install (Base — Pure Standard Library)
```bash
# Clone and enter directory
cd RECOVERIQ

# Base install with zero mandatory dependencies
pip install -e .[dev]
```

### Optional LLM Integration Install
```bash
# Install modern google-genai SDK for live Gemini calling
pip install -e .[llm]
```

---

## Running Tests & Verification

### 1. Run Offline Test Suite (Default — 190 Passed, 1 Skipped, 1 Xfailed)
```bash
pytest
```

### 2. Run Sprint 8 Economic Engine Verification Script
```bash
python scripts/verify_economics.py
```

### 3. Run Sprint 7 Failure & Resilience Verification Script
```bash
python scripts/verify_resilience.py
```

### 4. Run Sprint 6 Component Attribution & Ablation Verification Script
```bash
python scripts/verify_ablation.py
```

### 5. Run Sprint 5 End-to-End Evaluation Harness Verification Script
```bash
python scripts/verify_evaluation.py
```

### 6. Run Sprint 4 Probability Model Verification Script
```bash
python scripts/verify_probability_model.py
```

### 7. Run Sprint 3 AI Boundary & Safety Invariants Script
```bash
python scripts/verify_ai_boundaries.py
```

### 8. Run Sprint 2 Multi-Seed Simulation Verification Script
```bash
python scripts/verify_simulation.py
```

### 9. Run Sprint 1 Foundation Verification Script
```bash
python scripts/verify_foundation.py
```

### 10. Run Live Gemini Integration Test (Optional — Requires GEMINI_API_KEY)
```bash
pytest -m llm_integration
```

---

## Limitations & Boundaries
* **Research Prototype:** This project is an experimental research system evaluated on controlled synthetic scenarios.
* **Economic Engine Parameter Regime (Sprint 8 / Sprint 6 A2):**
  - **Intervention Cost vs Ticket Size:** In the current synthetic benchmark regime, payment amounts (mean ~INR 3,800) heavily dominate intervention costs (max cost = INR 3.50 for ESCALATE, i.e., ~0.09% of payment).
  - **Attribution Finding:** Sprint 6 Ablation 2 (NoEcon) achieved +298.02 INR higher NRV than full RecoverIQ across 5 seeds because greedy probability over-selects ESCALATE on high-value payments where EV remains strongly positive despite human costs.
  - **Honest Parameter Reporting:** The economic engine's cost-subtraction term therefore exerts a relatively small quantitative selection pressure under these specific synthetic parameter distributions. Per scientific integrity rules, cost and penalty parameters are not artificially manipulated to inflate the economic engine's contribution.
* **Failure & Resilience Scope (Sprint 7):**
  - **§18 Resilience Scenarios:** The six specified failure/resilience scenarios passed.
  - **§11.3 Cooldown Invariant:** The Policy Gate cooldown invariant remains unsatisfied in the current implementation (`policy/gate.py` lines 130–153).
  - **Sequential Replay Scope:** Concurrency and deduplication tests exercise sequential event replay and SHA-256 idempotency keys; they do not prove thread safety across distributed concurrent workers.
  - **Persistence Boundary:** The production engine does not persist payment state; multi-step replay relies on the test-only `PaymentReplayHarness`.
  - **Timeout Model:** Timeout is conservatively modeled as "dispatch did not complete" and enters a safe non-transitioned state without modeling downstream response loss.
* **Generator-Oracle Context Limitation:** Sprint 5 full RecoverIQ passed generator-assigned category labels directly into PaymentContext. Sprint 6 Ablation A1 establishes that rule-based extraction achieves 99.89% of oracle NRV (86.8% category agreement). This is an oracle-vs-rules comparison, not an LLM-vs-rules comparison.
* **Single-Decision-Point Scope:** Evaluates single-step recovery decisions. Multi-step schedules, retry cooldown trajectories, and production integrations are explicitly deferred.
* **Context Interpretation Only:** Gemini is strictly an interpretive context translator; it does not make payment decisions or recommend actions.
* **Statistical Estimation Only:** The probability model estimates recovery odds; it does not authorize or dispatch transactions.
* **No Production Gateway Claims:** RecoverIQ does not connect to live production payment gateways (e.g., Razorpay) in this repository.
* **No Claim of Inventing Retries:** The focus is on the decoupled context-economic-policy architecture.

