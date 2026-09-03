# RecoverIQ: Economic & Context-Aware Payment Recovery Engine
## System Specification Document (SPEC.md)

---

## 1. Project Overview

**RecoverIQ** is an experimental, context-aware, and economically optimized payment recovery engine. In modern digital payment flows, transaction failures occur due to a diverse set of reasons ranging from transient technical faults (e.g., gateway timeouts, network drops) to structural constraints (e.g., insufficient funds, daily velocity limits, expired cards, regulatory/2FA drops). 

Standard industry practices rely either on naive fixed-retry schedules (which burn payment processing fees, trigger fraud flags, and degrade customer trust) or rigid, static rule tables (which fail to adapt to complex error signals and customer-level context).

RecoverIQ investigates a modular architecture where payment failure context is structured, recovery probability is estimated via interpretable statistical models, potential actions are ranked by net expected economic value, and all actions are strictly regulated by a deterministic policy gate before execution.

### Core Architecture Principle
* **AI interprets:** The LLM is restricted to parsing raw, unstructured decline codes, error strings, and metadata into structured context signals and explanations.
* **The statistical model estimates:** An interpretable statistical/machine learning model estimates recovery probability $P(\text{recovery} \mid \text{context}, \text{action})$.
* **The economic engine evaluates:** An expected-value optimizer calculates the net financial payoff of candidate actions.
* **The deterministic policy engine authorizes:** Invariant safety rules, budget caps, cooldown windows, and state constraints gate and authorize actions.
* **The executor performs:** A dedicated execution layer dispatches authorized actions with idempotency and retry protection.
* **The audit/evaluation layer records:** An immutable ledger captures every event, context extraction, probability estimate, EV score, policy decision, and final outcome for rigorous evaluation.

---

## 2. Research Question

> **Primary Research Question:**
> *Can a context-aware, economically optimized payment recovery policy improve net recovered value over fixed-retry and rule-based strategies while maintaining strict payment-safety constraints?*

---

## 3. Scope

RecoverIQ is focused strictly on:
* **Failed-payment revenue recovery** occurring after an initial transaction failure notification (e.g., webhook or API error event).
* Decision optimization over intervention selection (immediate retry, scheduled retry, payment link delivery, customer nudge, human/tier escalation, or abort).
* Safety and financial efficiency optimization under cost, penalty, and retry constraints.
* Benchmarking on rigorously partitioned, stochastic synthetic transaction environments across multiple random seeds.

---

## 4. Non-Goals (Out of Scope)

The following areas are explicitly **OUT OF SCOPE**:
* **Checkout Abandonment:** Handling pre-checkout cart abandonment or marketing re-engagement.
* **B2B Receivables / Invoicing:** Multi-week accounts receivable, dunning cycles, or enterprise credit lines.
* **Fraud Detection Platform:** RecoverIQ is not a fraud classifier or chargeback defense platform.
* **General Risk Manager:** Not a broad credit risk, underwriting, or solvency engine.
* **Multi-Agent Architecture:** RecoverIQ will not deploy autonomous, multi-agent negotiations or emergent agent frameworks.
* **Retrieval-Augmented Generation (RAG):** No vector databases or external document retrieval pipelines.
* **Unrelated AI Features:** No generative marketing copy, generative customer support chatbots, or conversational agents.
* **Production Razorpay Integration:** No live merchant credentials, live webhooks, or production network calls.
* **Real Customer / Real Payment Data:** No personally identifiable information (PII) or proprietary payment datasets.

---

## 5. Architecture

The system pipeline is strictly unidirectional and decoupled across modular boundaries:

```
[Failed Payment Event]
          │
          ▼
┌───────────────────────────────────────┐
│          Context Ingestion            │
│  (Metadata, Error Payload, History)   │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│          AI Context Layer             │
│  - Parses unstructured error messages │
│  - Extracts structured failure signals│
│  - NO execution authority             │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│      Recovery Probability Model       │
│  - Interpretable Statistical Model    │
│  - Estimates P(recovery|context, act) │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│           Economic Engine             │
│  - Computes Expected Value (EV)       │
│  - EV = P * Amount - Cost - Penalty   │
│  - Proposes Argmax(EV) candidate      │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│       Deterministic Policy Gate       │
│  - State Machine validation           │
│  - Retry budget check                 │
│  - Cooldown window check              │
│  - Deduplication & Idempotency        │
│  - AUTHORIZES or REJECTS action       │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│               Executor                │
│  - Dispatches authorized actions      │
│  - Enforces execution idempotency     │
│  - Handles timeouts & failures safely │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│       Outcome & Audit/Evaluation      │
│  - State transition recording         │
│  - Immutable append-only audit log    │
│  - Metric tracking (Net Value, etc.)  │
└───────────────────────────────────────┘
```

### Module Responsibilities & Interfaces

1. **Context Ingestion**: Ingests raw failure event payload (Payment ID, Attempt Count, Amount, Currency, Customer Tier, Gateway Raw Error Code, Error Message, Timestamp).
2. **AI Context Layer**: Converts messy text and codes into standard categorical failure taxonomy (e.g., `INSUFFICIENT_FUNDS`, `NETWORK_TIMEOUT`, `CARD_EXPIRED`, `AUTHENTICATION_FAILED`, `AUTHENTICATION_REJECTED`, `INVALID_DETAILS`).
3. **Recovery Probability Model**: Given structured context vector $x$ and candidate action $a \in \mathcal{A}$, outputs probability scalar $P(\text{recovery} \mid x, a) \in [0, 1]$.
4. **Economic Engine**: Computes EV for each available action in $\mathcal{A}$, ranks actions by EV, and nominates the top valid candidate action.
5. **Deterministic Policy Gate**: Validates the proposed action against hard regulatory, business, and state rules. May override or clamp proposed actions to `STOP` or `ESCALATE` if safety invariants would be violated.
6. **Executor**: Executes only policy-authorized actions via synthetic payment handlers with idempotency tracking.
7. **Audit & Evaluation Engine**: Emits structured log entries per transaction lifecycle step; computes global performance metrics.

---

## 6. AI Responsibilities & Guardrails

### Core Rule
**The Large Language Model (LLM) must NEVER possess direct execution authority.**

### Permitted AI Functions
1. **Context Extraction & Taxonomy Mapping**: Parsing unstructured decline strings (e.g., `"bank_declined_do_not_honor"`, `"504 Gateway Timeout while contacting issuer switch"`) into structured failure categorizations.
2. **Rationale Explanation**: Generating a human-readable diagnostic explanation of the failure context for audit logs and escalation summaries.

### Explicit AI Guardrails & Restrictions
* The LLM **shall not** execute or dispatch payment actions.
* The LLM **shall not** perform probability arithmetic or calibrate probability values.
* The LLM **shall not** perform economic value calculations.
* The LLM **shall not** bypass the deterministic policy gate.
* Fallback behavior: If the LLM extraction fails, times out, or returns invalid JSON/schema, the system must deterministically fall back to rule-based context extraction without halting execution.

---

## 7. Recovery Model

The probability estimation component must be an interpretable statistical machine learning model (e.g., Logistic Regression / Regularized GLM / Calibrated Classifier).

### Formulation
For a payment failure with extracted structured context feature vector $x \in \mathbb{R}^d$ and candidate action $a \in \mathcal{A}$:

$$\hat{P}(\text{recovery} \mid x, a) = \sigma\left( w_a^T x + b_a \right)$$

where:
* $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function.
* $w_a, b_a$ are learned parameter weights for action $a$.
* Context vector $x$ includes: failure category, failure severity, customer history score, payment amount bracket, failure hour/day, attempt number, and time elapsed.

### Requirements
* Model outputs must be calibrated probabilities in $[0, 1]$.
* Model parameters must be inspectable (e.g., feature weights, odds ratios).
* The ML model must not have access to unpartitioned test-set data.

---

## 8. Economic Objective

The Economic Engine scores each candidate action $a \in \mathcal{A}$ using an explicit Expected Value (EV) formulation:

$$\text{EV}(a) = \hat{P}(\text{recovery} \mid x, a) \cdot V - C(a) - \Omega(a, x)$$

where:
* $V \in \mathbb{R}^+$ is the transaction payment amount.
* $\hat{P}(\text{recovery} \mid x, a)$ is the estimated recovery probability for action $a$.
* $C(a) \ge 0$ is the direct operational/intervention cost of executing action $a$ (e.g., gateway retry API fee, SMS/WhatsApp charge, escalation agent labor cost).
* $\Omega(a, x) \ge 0$ is the friction/penalty cost of executing action $a$ under context $x$ (e.g., customer churn penalty for excessive nudges, merchant risk penalty for high-frequency retries on hard declines).

### Selection & Gating Rule
$$\hat{a}_{\text{proposed}} = \arg\max_{a \in \mathcal{A}} \text{EV}(a)$$

* If $\max_{a} \text{EV}(a) \le 0$, the default recommendation is `STOP`.
* **Gating Invariant:** High EV alone does not guarantee execution. The candidate action $\hat{a}_{\text{proposed}}$ is sent to the Deterministic Policy Gate. Only upon policy clearance is the action authorized.

---

## 9. Action Space

The engine operates on a discrete, strictly bounded action space $\mathcal{A}$:

| Action | Description | Typical Use Context | Direct Cost $C(a)$ | Penalty Risk $\Omega(a, x)$ |
| :--- | :--- | :--- | :--- | :--- |
| `RETRY_NOW` | Immediate gateway retry within $0 - 5$ seconds | Transient network/gateway timeout, switch reset | Low (gateway fee) | Medium if repeated |
| `RETRY_LATER` | Scheduled gateway retry after a cooldown window | Daily limit reset, balance replenishment window | Low (gateway fee) | Low |
| `SEND_LINK` | Generate and dispatch alternate payment link via email/SMS | Card expired, 3DS authentication failure | Medium (SMS + gateway link fee) | Low/Medium |
| `NUDGE` | Send non-intrusive push/in-app prompt to update details | Customer hesitation, app crash during 2FA | Medium (messaging fee) | High if spammed |
| `ESCALATE` | Route transaction to high-priority customer support queue | High-value VIP customer, repeated ambiguous failure | High (human agent time) | Low (VIP service) |
| `STOP` | Cease further recovery attempts; mark permanently failed | Hard decline, stolen card, exhausted retry budget | $0.00$ | $0.00$ |

---

## 10. Payment State Machine

The lifecycle of each failed payment is strictly governed by a deterministic Finite State Machine (FSM):

```
                     ┌──────────────────┐
                     │      FAILED      │
                     │  (Initial State) │
                     └─────────┬────────┘
                               │
                               │ Start Recovery
                               ▼
                     ┌──────────────────┐
                     │    RECOVERING    │◄───────────────┐
                     └──┬──────┬──────┬─┘               │
                        │      │      │                 │ Retry /
           Success      │      │      │ Permanent Stop  │ Re-attempt
      ┌─────────────────┘      │      └────────────────┐│ (within budget)
      ▼                        ▼                       ▼▼
┌───────────┐            ┌───────────┐           ┌───────────┐
│ RECOVERED │            │ ESCALATED │           │  FAILED   │
│ (Terminal)│            │ (Terminal)│           │ (Terminal)│
└───────────┘            └───────────┘           └───────────┘
```

### State Definitions
* `FAILED` (Initial): Raw failure event ingested; ready for initial evaluation.
* `RECOVERING`: Active intervention initiated or scheduled (waiting on cooldown or customer response).
* `RECOVERED` (Terminal): Payment successfully authorized and captured. No further actions permitted.
* `ESCALATED` (Terminal): Case transferred to human operations queue. Automated recovery halts.
* `FAILED` (Terminal): All retry attempts exhausted, hard stop triggered, or recovery aborted. No further actions permitted.

### Invariants
* Any event or action attempting an illegal transition (e.g., `RECOVERED` $\to$ `RECOVERING` or `FAILED (Terminal)` $\to$ `RECOVERED`) must be immediately rejected with an invariant violation recorded.

---

## 11. Safety Requirements & Policy Engine

The Deterministic Policy Gate enforces hard safety invariants that can never be overridden by AI, ML probability, or EV optimization:

1. **Event Deduplication**: Every incoming event must carry a unique `event_id`. Duplicate `event_id` arrivals within a sliding window are dropped idempotently.
2. **Retry Budget Limits**: Maximum number of recovery attempts per payment $N_{\max}$ (default: $N_{\max} = 3$). Once $N \ge N_{\max}$, the policy forces `STOP` or `ESCALATE`.
3. **Mandatory Cooldowns**: Enforce minimum duration between consecutive interventions on the same payment (e.g., minimum 15 minutes between `RETRY_LATER` or `NUDGE` attempts).
4. **Hard Decline Exclusion**: Inviolable rules banning retries for deterministic hard declines (e.g., stolen card, fraudulent card, account closed).
5. **State Transition Validation**: Every action must validate current state before execution.
6. **Executor Timeout Protection**: Synthetic execution calls must be wrapped with timeouts; timed-out actions must enter a safe pending/reconciliation state rather than blind re-execution.
7. **Out-of-Order Event Handling**: Sequence numbers / timestamps are checked; stale events arriving after newer events or terminal transitions are discarded safely.
8. **Immutable Audit Trail**: Every policy decision (Approved, Rejected, Overridden) and the active rule responsible must be logged to an append-only audit record.

---

## 12. Idempotency

* **Idempotency Key Construction**: Each action execution creates an idempotent key:
  $$\text{IdempotencyKey} = \text{SHA256}(\text{payment\_id} + \text{action\_type} + \text{attempt\_number} + \text{event\_id})$$
* **Execution Guarantee**: Before any executor call, the system checks whether $\text{IdempotencyKey}$ has already executed or is currently in flight.
* **Duplicate Protection**: Repeated dispatches of the same action request return the original cached response without triggering downstream actions.

---

## 13. Synthetic Data Engine

To ensure rigorous, unbiased evaluation without data leakage:

### Ground Truth Generation
* Ground-truth payment recoverability is governed by an **independent stochastic process** (e.g., latent failure recovery distributions with ground-truth transition matrices).
* **Strict Independence Rule:** Synthetic labels must NOT be generated using the heuristic rules, logistic regression model, or decision logic of RecoverIQ.

### Features & Signals
* Synthetic payments contain:
  * `payment_id`, `customer_id`, `amount`, `currency`, `timestamp`
  * `raw_error_code`, `raw_error_message`
  * `customer_tier` (e.g., VIP, Standard, New)
  * `payment_method` (e.g., Credit Card, Debit Card, UPI/Instant Transfer, Net Banking)
  * `latent_recoverability_profile` (Hidden from model, used exclusively by the environment simulator to resolve action outcomes).

### Data Partitioning & Leakage Prevention
* Data must be partitioned by `payment_id` / `customer_id` into distinct **Train** and **Test** sets.
* Statistical models and contextual embeddings are trained strictly on the Train set; evaluation occurs strictly on the unseen Test set.

---

## 14. Baselines

RecoverIQ will be compared against two canonical baseline strategies:

1. **Fixed-Retry Baseline**:
   * Blind, static retry schedule (e.g., retry immediately at $t=0$, retry again at $t=+1\text{h}$, retry again at $t=+24\text{h}$, then stop).
   * Context-agnostic and cost-agnostic.
2. **Rule-Based Baseline**:
   * Deterministic if-else heuristics mirroring standard payment ops (e.g., if `error == INSUFFICIENT_FUNDS` $\to$ `RETRY_LATER`; if `error == 504_TIMEOUT` $\to$ `RETRY_NOW`; if `error == CARD_EXPIRED` $\to$ `SEND_LINK`; else `STOP`).
   * No probabilistic estimation or economic expected-value optimization.
3. **RecoverIQ (Full System)**:
   * AI context parsing + Statistical $P(\text{recovery})$ + Economic EV Optimization + Deterministic Policy Gate.

---

## 15. Evaluation Metrics

### Primary Metric
* **Net Recovered Value (NRV)**:
  $$\text{NRV} = \sum_{i \in \text{Recovered}} V_i - \sum_{j \in \text{All Actions}} C(a_j) - \sum_{k \in \text{Penalties}} \Omega(a_k, x_k)$$

### Secondary Metrics
1. **Gross Recovered Revenue**: Total raw amount of recovered payments ($\sum V_{\text{recovered}}$).
2. **Recovery Rate**: Percentage of failed payments successfully recovered ($\frac{N_{\text{recovered}}}{N_{\text{failed}}} \times 100\%$).
3. **Average Attempts Per Payment**: Total recovery actions executed divided by total failed payments.
4. **Unnecessary Interventions Count**: Actions dispatched on unrecoverable payments or redundant actions on already solvable payments.
5. **Escalation Rate**: Percentage of transactions escalated to manual support.
6. **Policy Violation Rate**: Number of policy gate breaches (Must be strictly **0.00%**).
7. **Average Recovery Latency**: Mean time elapsed from initial failure to terminal state.
8. **Failure-Handling Success Rate**: Percentage of injected faults (timeouts, duplicates, out-of-order events) handled without state corruption or double execution.

---

## 16. Multi-Seed Evaluation

To eliminate stochastic anomalies and ensure statistical validity:
* **Minimum Target**: 3 independent random seeds ($S \in \{42, 100, 2024\}$).
* **Preferred Target**: 5 independent random seeds ($S \in \{42, 100, 2024, 777, 999\}$).
* **Reporting Standard**: All performance tables and plots must report the **Mean $\pm$ Standard Deviation** across the seeds. No cherry-picking individual runs.

---

## 17. Ablation Experiments

To isolate the contribution of each system component:

1. **Ablation 1 (RecoverIQ w/o AI Context Layer)**:
   * Replaces LLM structured context extraction with direct raw string/regex mapping into the statistical model.
   * Isolates the value added by LLM contextual interpretation.
2. **Ablation 2 (RecoverIQ w/o Economic Engine)**:
   * Replaces EV maximization with greedy probability selection ($\arg\max_a \hat{P}(\text{recovery} \mid x, a)$ without considering costs $C(a)$ or penalties $\Omega$).
   * Isolates the value added by the economic cost-benefit optimization.

---

## 18. Failure & Resilience Testing

A dedicated failure injection test suite will validate system robustness against real-world payment edge cases:

1. **Duplicate Webhook Delivery**: Dispatching the identical failure webhook multiple times concurrently and sequentially.
2. **Out-of-Order Webhook Delivery**: Sending a later status update (e.g., `RECOVERED` or `FAILED`) prior to the initial failure or retry acknowledgement.
3. **Executor Timeout**: Simulating network timeouts during executor action dispatch; verifying proper idempotent recovery/retry without duplicate charges.
4. **Duplicate Action Request**: Emitting identical action recommendations concurrently to the Policy Gate and Executor.
5. **Exhausted Retry Budget**: Attempting to force additional recovery actions after reaching $N_{\max}$.
6. **Policy Rejection**: Submitting high-EV candidate actions that violate safety invariants (e.g., retrying a stolen card) and verifying 100% rejection rate.

---

## 19. Claim Limitations & Boundaries

The following scientific and operational limitations apply to all reporting and documentation:
* **No Production Razorpay Integration**: RecoverIQ is evaluated using a simulated environment; no live gateway connection is claimed.
* **No Production-Level Performance Claims**: Metrics reflect controlled synthetic scenarios and do not represent claims of production enterprise throughput.
* **No Claim of Inventing Retries**: Payment retries are standard industry practice; RecoverIQ's contribution is the specific decoupled architecture combining context interpretation, statistical estimation, economic gating, and deterministic safety enforcement.
* **Controlled Evaluation**: System results are valid within the bounds of the synthetic benchmark environment and must be presented as such.

---

## 20. Definition of Done (DoD)

The RecoverIQ project milestone will be considered complete when:
1. **Specification Verified**: `SPEC.md` is fully defined and free of internal contradictions.
2. **Clean Modular Architecture**: Context Ingestion, AI Layer, Statistical Model, Economic Engine, Policy Gate, Executor, and Audit Log are cleanly decoupled into distinct modules.
3. **Zero Safety Invariant Violations**: Policy Engine guarantees 0 invariant breaches across all test runs.
4. **Deterministic Failure Resilience**: All 6 failure injection tests pass with 100% reliability.
5. **Clean Data Generator**: Independent synthetic generator produces reproducible train/test sets without target leakage.
6. **Multi-Seed Benchmark Executed**: Baselines (Fixed-Retry, Rule-Based), RecoverIQ, and Ablations are evaluated over $\ge 3$ (preferred 5) seeds with Mean $\pm$ Std reported for all primary and secondary metrics.
