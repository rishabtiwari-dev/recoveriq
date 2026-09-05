# RecoverIQ: Sequential Decision Economics for Payment Recovery

**A controlled experimental framework for studying sequential recovery decisions, option value pricing, robustness under model misspecification, and failure modes in AI-assisted payment recovery.**

> RecoverIQ is not presented as a production payment recovery system. It is a research prototype investigating whether context-aware, economically optimized sequential policies can improve net recovered value over fixed-retry and rule-based strategies while maintaining strict payment-safety constraints.

---

## Problem Motivation

In digital payment systems, transaction failures are routine — gateway timeouts, insufficient funds, expired cards, authentication drops, and regulatory declines generate a stream of failed payments that require intelligent recovery decisions.

**The industry standard** is either:
- **Fixed-retry schedules** — blind, context-agnostic retries that burn processing fees, trigger fraud flags, and degrade customer trust.
- **Static rule tables** — rigid if-else heuristics that cannot adapt to complex error signals, customer context, or economic trade-offs.

**The research question:** Can a modular architecture that separates *context interpretation* (AI), *probability estimation* (statistical model), *economic valuation* (expected value optimization), and *safety enforcement* (deterministic policy gate) produce measurably better recovery outcomes — and can we rigorously characterize when it does and when it fails?

---

## Research Questions

### Primary
> Can a context-aware, economically optimized payment recovery policy improve net recovered value over fixed-retry and rule-based strategies while maintaining strict payment-safety constraints?

### Sequential (Sprints 11–15)
> Does explicitly pricing the **option value** of future recovery attempts (via Bellman DP, model-free Q-learning, or hybrid arbitration) prevent premature escalation and improve automated recovery trajectories?

> How robust are these sequential policies under **model misspecification** and **distribution shift**?

---

## System Architecture

```
[Failed Payment Event]
         │
         ▼
┌─────────────────────────────────┐
│      Context Ingestion          │
│  (Metadata, Error, History)     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      AI Context Layer           │
│  Gemini 3.8 Flash: interprets   │
│  unstructured errors → taxonomy │
│  NO execution authority         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Recovery Probability Model     │
│  P(recovery | context, action)  │
│  Interpretable logistic models  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Economic Engine            │
│  EV = P·Amount − Cost − Penalty │
│  Proposes argmax(EV) candidate  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Deterministic Policy Gate     │
│  State machine, retry budget,   │
│  cooldowns, hard-decline block  │
│  AUTHORIZES or REJECTS action   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Executor + Audit Trail        │
│  Idempotent dispatch, immutable │
│  append-only audit logging      │
└─────────────────────────────────┘
```

### Core Architectural Principle

| Layer | Role | Authority |
|---|---|---|
| **AI (Gemini 3.8 Flash)** | Interprets unstructured error messages into structured failure taxonomy | Context interpretation only — **no execution authority** |
| **Statistical Model** | Estimates `P(recovery \| context, action)` via per-action logistic regression | Probability estimation only — **no action authorization** |
| **Economic Engine** | Computes `EV(a) = P̂·V − C(a) − Ω(a,x)` and proposes `argmax(EV)` | Proposal only — **subject to policy gate** |
| **Policy Gate** | Enforces state machine, retry budgets, cooldowns, hard-decline exclusion | **Final authority** — can override any proposal |
| **Executor** | Dispatches authorized actions with SHA-256 idempotency keys | Execution only — **policy-gated** |

---

## Sequential Decision Approaches (Sprints 11–15)

Sprint 10 revealed that RecoverIQ's single-step EV optimizer frequently selects `ESCALATE` at the first decision point because escalation carries high estimated recovery probability (~0.80) and low relative cost. In sequential trajectories (N_max = 3 attempts), this terminates automated recovery prematurely, forfeiting the **option value** of cheaper retry actions.

Sprints 11–15 investigate sequential policies that explicitly price this option value:

### Bellman Dynamic Programming (Sprint 13)
Finite-horizon backward induction computes value-to-go `V(s, attempt)` at each decision point, incorporating the expected value of *future* recovery attempts. This endogenously suppresses premature `ESCALATE` without hardcoded attempt-gating rules.

### Model-Free Q-Learning (Sprint 14)
Fitted Q-Iteration learns state-action values from disjoint training trajectories without depending on the probability model. Tests whether a **model-free** policy remains robust when the probability model is deliberately misspecified (M0–M3 perturbation regimes) or the evaluation distribution shifts (D0–D3 shift regimes).

### Uncertainty-Aware Hybrid (Sprint 15)
Combines Bellman and Model-Free action values via uncertainty-weighted arbitration:

```
Q_hybrid(s,a) = w(s,a)·Q_bellman(s,a) + (1−w(s,a))·Q_modelfree(s,a)
```

Three regimes tested: Equal-weight (`w=0.50`), Fixed-weight, and Uncertainty-adaptive weighting derived from model confidence and Q-table visitation counts.

---

## Experimental Methodology

All experiments follow a rigorous controlled protocol:

- **Synthetic Data**: Independent stochastic ground-truth generator — labels are NOT generated by RecoverIQ's own models or rules
- **Train/Test Partition**: Payment-level splitting with zero data leakage
- **Common Random Numbers (CRN)**: All strategies evaluated on identical payment sequences per seed
- **Multi-Seed Evaluation**: 20 deterministic seeds for robustness validation (Sprints 12–15)
- **Paired Comparisons**: CRN-paired bootstrap with 95% confidence intervals
- **Model Misspecification**: Controlled M0–M3 perturbation regimes (identity → severe bias injection)
- **Distribution Shift**: Controlled D0–D3 shift regimes (in-distribution → out-of-distribution)
- **Frozen Safety Gate**: Policy gate operates identically across all strategies — 0.00% policy violation rate

---

## Key Experimental Results

### Sequential Policy Comparison (20-seed CRN evaluation)

| Policy | Recovery Rate | NRV (INR) | Escalation Rate |
|---|---|---|---|
| **Bellman DP** | 60.44% ± 2.93% | 1,367,896.21 ± 235,279.97 | 22.48% |
| **Model-Free Q** | 67.80% ± 2.25% | 1,481,173.68 ± 218,892.84 | 0.00% |
| **Hybrid-Equal** | 68.87% ± 2.57% | 1,476,069.85 ± 229,886.80 | 3.99% |

**Hybrid-Equal had the strongest observed recovery rate** among tested policies at 68.87% ± 2.57%.

### Model-Free vs Bellman Paired CRN Comparison
- **Per-payment NRV lift**: +INR 454.11/payment
- **95% CI**: [+INR 216.42, +INR 707.12]
- **Recovery rate lift**: +7.38 percentage points

### Robustness Under Model Misspecification (M0 → M3)

| Policy | M0→M3 NRV Degradation |
|---|---|
| **Bellman DP** | −5.48% (−INR 74,988.63) |
| **Model-Free Q** | 0.00% (model-independent) |
| **Hybrid-Uncertainty** | −10.35% (the "confidently wrong" failure) |

### Sprint 14 Hypothesis Results (Model-Free Robustness)
| Hypothesis | Result |
|---|---|
| H1: Model-Free matches/exceeds Bellman under correct model (M0) | ✅ SUPPORTED |
| H2: Model-Free degrades less than Bellman under misspecification (M1–M3) | ✅ SUPPORTED |
| H3: Model-Free resists distribution shift better than Bellman (D1–D3) | ✅ SUPPORTED |
| H4: Model-Free achieves lower escalation rate than Bellman | ✅ SUPPORTED |

### Sprint 15 Hypothesis Results (Hybrid Policy)
| Hypothesis | Result |
|---|---|
| H1: Hybrid-Uncertainty outperforms both components | ❌ NOT SUPPORTED |
| H2: Hybrid-Uncertainty degrades less than Bellman under M3 | ❌ NOT SUPPORTED |
| H3: Hybrid-Uncertainty outperforms Equal-weight under M3 | ❌ NOT SUPPORTED |
| H4: Hybrid-Equal outperforms pure Bellman | ✅ SUPPORTED |
| H5: Uncertainty-weighting suppresses escalation better | ❌ NOT SUPPORTED |

---

## Key Research Discoveries

### 1. Sequential Option Value Is Real (Sprint 11)
Restricting `ESCALATE` until the final attempt preserves cheap retry option value:
- RecoverIQ-Unconstrained: 17.83% recovery, INR 630,655.80 NRV
- RecoverIQ-Tiered: 60.75% recovery, INR 1,333,193.10 NRV

### 2. The "Confidently Wrong" Failure (Sprint 15)
The Bellman confidence heuristic based on `|p − 0.5|` paradoxically *increases* under M3 misspecification because inflated `ESCALATE` probabilities (pushed toward 1.0) appear *more decisive* to the uncertainty estimator. This causes Hybrid-Uncertainty to degrade by −10.35% under M3 while Bellman only degrades −5.48%.

**Why Equal-weight wins:** Fixed equal weighting (`w = 0.50`) robustly balances both signals without being fooled by artificial model certainty. This is a cautionary finding about naive confidence-based arbitration.

### 3. Terminal Escalation Finding (Sprint 13)
Bellman DP concentrates escalation at the final attempt (attempt 3): **64.78%** of Bellman escalations occur at attempt 3. Hybrid-Uncertainty reduces this to **27.41%** at attempt 3.

### 4. Model-Free Zero Escalation (Sprint 14)
Model-Free Q-Learning achieves **0% escalation** at all attempts because its Q-table learns empirically that automated retries produce higher returns than terminal escalation — without any explicit escalation-suppression rule.

---

## Safety & Guardrails

- **Deterministic Policy Gate**: Hard safety invariants that can never be overridden by AI, ML, or EV optimization
- **0.00% Policy Violation Rate**: Verified across all strategies, all seeds, all experiments
- **AI Has No Execution Authority**: Gemini is strictly an interpretive context translator
- **Retry Budget Enforcement**: Maximum N_max = 3 attempts per payment, then forced STOP/ESCALATE
- **Hard Decline Exclusion**: Stolen card, fraudulent card, account closed → retries permanently banned
- **Cooldown Windows**: Minimum 15-minute intervals between consecutive interventions (SPEC §11.3)
- **Idempotency**: SHA-256 keyed deduplication prevents double execution
- **Immutable Audit Trail**: Every decision logged to append-only audit record

---

## Limitations

- **Research Prototype**: Evaluated on controlled synthetic scenarios only — no production performance claims
- **No Production Gateway**: Does not connect to live payment gateways (Razorpay or otherwise)
- **No Real Customer Data**: All data is synthetically generated — no PII or proprietary datasets
- **Synthetic Parameter Regime**: Payment amounts (~INR 3,800 mean) heavily dominate intervention costs (max INR 3.50 for ESCALATE), limiting the economic engine's discriminative pressure
- **Generator-Oracle Context**: Sprint 5 evaluation used generator-assigned category labels (oracle shortcut); Sprint 6 A1 showed rule-based extraction achieves 99.89% of oracle NRV
- **Sequential Horizon**: N_max = 3 attempts — longer retry horizons are not evaluated
- **No Claim of Inventing Retries**: Payment retries are standard industry practice; the contribution is the modular context-economic-policy architecture and the sequential option value analysis

---

## Interactive Demo (Sprint 16)

An interactive Streamlit research demo exposes the complete experimental system:

```bash
# Install demo dependencies
pip install -e .[demo]

# Launch the interactive demo
streamlit run app/demo.py
```

**Demo modes:**
1. **Presentation Mode** — Guided walkthrough of research findings with cached verified results
2. **Single Payment Explorer** — Step through one payment's recovery trajectory decision-by-decision
3. **Strategy Comparison** — Side-by-side evaluation of all sequential policies
4. **Research Dashboards** — Robustness, misspecification, and ablation analysis
5. **Architecture Viewer** — Interactive system architecture and module relationships

---

## Repository Structure

```
recoveriq/
├── SPEC.md                              # Authoritative system specification
├── README.md                            # This file
├── pyproject.toml                       # Project metadata & dependencies
├── .env.example                         # Environment variable template
│
├── app/
│   └── demo.py                          # Sprint 16: Streamlit interactive demo
│
├── docs/
│   ├── architecture.md                  # System architecture design
│   ├── sprint16_architecture.md         # Sprint 16 architecture audit
│   ├── sprint16_demo_guide.md           # Demo launch & usage guide
│   └── sprint16_presentation_flow.md    # 5-8 minute presentation script
│
├── scripts/
│   ├── verify_foundation.py             # Sprint 1 verification
│   ├── verify_simulation.py             # Sprint 2 verification
│   ├── verify_ai_boundaries.py          # Sprint 3 verification
│   ├── verify_probability_model.py      # Sprint 4 verification
│   ├── verify_evaluation.py             # Sprint 5 verification
│   ├── verify_ablation.py               # Sprint 6 verification
│   ├── verify_resilience.py             # Sprint 7 verification
│   ├── verify_economics.py              # Sprint 8 verification
│   ├── verify_trajectory.py             # Sprint 10 verification
│   ├── verify_sprint11_audit.py         # Sprint 11 verification
│   ├── verify_sprint12_robustness.py    # Sprint 12 verification
│   ├── verify_sprint13_bellman.py       # Sprint 13 verification
│   ├── verify_sprint14_model_free.py    # Sprint 14 verification
│   ├── verify_sprint15_hybrid.py        # Sprint 15 verification
│   └── verify_sprint16_demo.py          # Sprint 16 verification
│
├── src/recoveriq/
│   ├── domain/                          # Core domain models, state machine, actions
│   ├── context/                         # Context ingestion & rule-based extractors
│   ├── ai/                              # Gemini 3.8 Flash context layer & guardrails
│   ├── model/                           # Statistical recovery probability model
│   ├── economics/                       # Expected value optimizer & cost models
│   ├── policy/                          # Deterministic policy gate & invariants
│   ├── executor/                        # Idempotent action executor
│   ├── audit/                           # Structured immutable audit logging
│   ├── config/                          # Typed settings (costs, penalties, LLM)
│   ├── simulation/                      # Synthetic data & simulation environment
│   ├── evaluation/                      # Evaluation, ablation & research extensions
│   │   ├── strategies.py               #   RecoverIQ, Fixed-Retry, Rule-Based
│   │   ├── ablation_strategies.py      #   Sprint 6: A1/A2 ablation strategies
│   │   ├── metrics.py                  #   NRV, recovery rate, multi-seed aggregation
│   │   ├── runner.py                   #   CRN benchmark & ablation runner
│   │   ├── trajectory.py              #   Sprint 10: Sequential trajectory evaluation
│   │   ├── sequential_policy.py       #   Sprint 11: Tiered sequential policy
│   │   ├── robustness.py              #   Sprint 12: Statistical robustness validation
│   │   ├── bellman_policy.py          #   Sprint 13: Bellman DP option value
│   │   ├── model_error.py             #   Sprint 14: M0–M3 & D0–D3 perturbation
│   │   ├── model_free_policy.py       #   Sprint 14: Fitted Q-Iteration policy
│   │   ├── hybrid_policy.py           #   Sprint 15: Uncertainty-aware hybrid
│   │   ├── demo_data.py               #   Sprint 16: Cached verified results
│   │   └── demo_engine.py             #   Sprint 16: Anti-leakage demo adapter
│   └── engine.py                        # Unified pipeline coordinator
│
└── tests/                               # 270 tests across 44 suites
    ├── test_actions.py ... test_state_machine.py           # Core domain tests
    ├── test_eval_*.py                                      # Evaluation harness tests
    ├── test_sprint11_audit.py                              # Sequential policy tests
    ├── test_sprint12_robustness.py                         # Robustness validation tests
    ├── test_sprint13_bellman.py                             # Bellman DP tests
    ├── test_sprint14_model_free.py                          # Model-Free Q tests
    ├── test_sprint15_hybrid.py                              # Hybrid policy tests
    └── test_sprint16_demo.py                                # Demo layer tests
```

---

## Installation & Testing

### Prerequisites
- Python 3.11+ (tested on Python 3.13)
- No mandatory external dependencies — core system uses only the Python standard library

### Install
```bash
git clone https://github.com/Rishab-Tiwari/RECOVERIQ.git
cd RECOVERIQ

# Base install (pure standard library + pytest)
pip install -e .[dev]

# Optional: LLM integration (requires GEMINI_API_KEY)
pip install -e .[llm]

# Optional: Interactive demo
pip install -e .[demo]
```

### Run Tests
```bash
# Full offline test suite (270 passed, 1 skipped)
pytest

# The single skipped test requires a live GEMINI_API_KEY
# and is intentionally not part of the offline test suite
pytest -m llm_integration  # Optional: run with GEMINI_API_KEY set
```

### Run Verification Scripts
```bash
# Sprint verification scripts (each is self-contained)
python scripts/verify_foundation.py          # Sprint 1
python scripts/verify_simulation.py          # Sprint 2
python scripts/verify_ai_boundaries.py       # Sprint 3
python scripts/verify_probability_model.py   # Sprint 4
python scripts/verify_evaluation.py          # Sprint 5
python scripts/verify_ablation.py            # Sprint 6
python scripts/verify_resilience.py          # Sprint 7
python scripts/verify_economics.py           # Sprint 8
python scripts/verify_trajectory.py          # Sprint 10
python scripts/verify_sprint11_audit.py      # Sprint 11
python scripts/verify_sprint12_robustness.py # Sprint 12
python scripts/verify_sprint13_bellman.py    # Sprint 13
python scripts/verify_sprint14_model_free.py # Sprint 14
python scripts/verify_sprint15_hybrid.py     # Sprint 15
python scripts/verify_sprint16_demo.py       # Sprint 16
```

---

## Sprint History

| Sprint | Title | Key Contribution |
|---|---|---|
| 1 | Foundation & Architecture | Domain models, state machine, policy gate, audit, modular interfaces |
| 2 | Synthetic Data & Simulation | Multi-seed reproducible generator, zero-leakage train/test partitioning |
| 3 | LLM Context Integration | Gemini 3.8 Flash error interpretation, strict privacy & anti-authority guardrails |
| 4 | Statistical Probability Model | Per-action logistic regression, counterfactual dataset, STOP invariant |
| 5 | End-to-End Evaluation | RecoverIQ vs Fixed-Retry vs Rule-Based, CRN benchmarking |
| 6 | Component Attribution | Context-source ablation (A1), economic engine ablation (A2) |
| 7 | Failure & Resilience | §18 failure injection scenarios, idempotency & replay harness |
| 8 | Economic Engine Hardening | SPEC §8/§9 EV optimization audit & parameter verification |
| 9 | Safety Invariant Resolution | SPEC §11.3 mandatory cooldown gate enforcement |
| 10 | Sequential Trajectories | Multi-step trajectory evaluation (N_max = 3), terminal distribution analysis |
| 11 | Sequential Policy Ablation | Tiered escalation gating, human-ops valuation, option value evidence |
| 12 | Robustness Validation | 20-seed CRN, paired bootstrap, strata analysis, sensitivity testing |
| 13 | Bellman Option Value | Finite-horizon DP, endogenous escalation suppression, value-to-go |
| 14 | Model-Free Q-Learning | Fitted Q-Iteration, M0–M3 misspecification, D0–D3 distribution shift |
| 15 | Hybrid Policy | Uncertainty-aware arbitration, "confidently wrong" discovery, equal-weight robustness |
| 16 | Research Demo & Presentation | Interactive Streamlit UI, cached verified results, anti-leakage demo adapter |

---

## Research Positioning

RecoverIQ is a **controlled experimental framework** for studying:

1. **Sequential recovery decision economics** — how option value pricing changes optimal action selection
2. **Model robustness** — how model-based vs model-free policies degrade under misspecification
3. **Failure modes of confidence-based arbitration** — the "confidently wrong" phenomenon
4. **Safety-constrained AI systems** — modular separation of AI interpretation from deterministic authorization

It is not presented as a production payment recovery system. Results are valid within the bounds of the synthetic benchmark environment.

---

## License

MIT

---

## Acknowledgments

Built for the **Razorpay AI Buildathon**. RecoverIQ investigates the specific decoupled architecture combining context interpretation, statistical estimation, economic gating, and deterministic safety enforcement — applied to the payment recovery domain.
