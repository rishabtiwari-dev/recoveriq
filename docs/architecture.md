# RecoverIQ Architecture & Module Boundaries (Sprint 1)

This document describes the software architecture and module interfaces established in Sprint 1 of the RecoverIQ project, strictly aligned with `SPEC.md`.

---

## 1. System Dataflow & Safety Boundary

The architecture enforces a strict unidirectional pipeline where each layer is decoupled:

```
[PaymentFailedEvent]
        │
        ▼
[Context Ingestion / RuleBasedContextExtractor]
        │
        ▼
[AI Context Layer (StubAIContextLayer)]  ◄─── CRITICAL: Interprets ONLY. NO execution authority.
        │
        ▼
[Recovery Probability Model (StubProbabilityModel)]
        │
        ▼
[Economic Engine (DefaultEconomicEngine)]  ◄─── Computes EV = P * V - Cost - Penalty
        │
        ▼
[Deterministic Policy Gate (InvariantPolicyGate)] ◄─── MANDATORY: State, Budget, Hard Declines, Cooldowns
        │
        ▼
[Action Executor (InMemoryActionExecutor)]  ◄─── Strictly executes authorized actions with idempotency
        │
        ▼
[Structured Audit Logger (InMemoryAuditLogger)] & State Machine
```

---

## 2. Module Responsibilities

| Module | Package | Responsibility |
| :--- | :--- | :--- |
| **Domain** | `recoveriq.domain` | Strongly typed entities (`Payment`, `PaymentContext`, `Action`, `PaymentState`, `IdempotencyRecord`, `RecoveryDecision`, `PolicyDecision`). |
| **Context** | `recoveriq.context` | Ingests raw failure event payloads and performs fallback/rule-based categorization. |
| **AI** | `recoveriq.ai` | Performs unstructured text/signal interpretation; produces structured context and diagnostic rationale. Zero execution capabilities. |
| **Model** | `recoveriq.model` | Statistical recovery probability model contract estimating $P(\text{recovery} \mid \text{context}, a)$. |
| **Economics** | `recoveriq.economics` | Computes Net Expected Value ($EV(a) = P \cdot V - C - \Omega$) using Decimal numerical safety. |
| **Policy** | `recoveriq.policy` | Deterministic invariant gating (state machine checks, max retry budget, hard decline retry exclusions, cooldowns). |
| **Executor** | `recoveriq.executor` | Dispatches authorized actions under SHA-256 idempotency protection. |
| **Audit** | `recoveriq.audit` | Immutable structured logging across all lifecycle events. |
| **Config** | `recoveriq.config` | Strongly typed configuration system for policies, costs, penalties, and thresholds. |
