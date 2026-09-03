"""RecoverIQ Sprint 8 — Economic Engine Verification (SPEC §8 & §9).

This script performs verification of the Economic Engine across seven distinct sections:
- SECTION A: SPEC §9 Action Costs (Direct operational costs C(a))
- SECTION B: Penalty Matrix (Friction penalty Omega(a, x) across Customer Tiers)
- SECTION C: Expected Value (EV) Formula Demonstration (EV = P * V - Cost - Penalty)
- SECTION D: STOP Fallback Mechanism (Boundary condition when max EV <= 0)
- SECTION E: Custom EV Threshold Behavior (Configurable min_ev_threshold)
- SECTION F: EconomicEngine Protocol Conformance (Runtime structural check)
- SECTION G: Research Limitation Statement (Sprint 6 A2 parameter regime findings)
"""

import sys
from decimal import Decimal
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repo root is on sys.path when script is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recoveriq.config.settings import (
    ActionCostConfig,
    EconomicConfig,
    PenaltyConfig,
)
from recoveriq.domain.actions import Action
from recoveriq.domain.models import (
    CustomerTier,
    FailureCategory,
    FailureSeverity,
    PaymentContext,
    PaymentMethod,
)
from recoveriq.economics.engine import DefaultEconomicEngine, EconomicEngine
from recoveriq.model.probability import ProbabilityEstimate


def verify_section_a_action_costs() -> bool:
    """SECTION A — SPEC §9 Action Costs."""
    print("\n" + "=" * 80)
    print("SECTION A — SPEC §9 ACTION COSTS C(a)")
    print("=" * 80)

    cost_cfg = ActionCostConfig()
    expected_costs = {
        Action.RETRY_NOW: Decimal("0.15"),
        Action.RETRY_LATER: Decimal("0.15"),
        Action.SEND_LINK: Decimal("0.35"),
        Action.NUDGE: Decimal("0.10"),
        Action.ESCALATE: Decimal("3.50"),
        Action.STOP: Decimal("0.00"),
    }

    print(f"{'Action':<15} | {'Configured Cost':<15} | {'Expected Cost':<15} | {'Status'}")
    print("-" * 65)

    all_passed = True
    for action in Action:
        actual = cost_cfg.get_cost(action)
        expected = expected_costs[action]
        match = (actual == expected)
        all_passed = all_passed and match
        status_str = "PASS" if match else "FAIL"
        print(f"{action.value:<15} | INR {actual:<10.2f} | INR {expected:<10.2f} | {status_str}")

    assert all_passed, "Action costs do not match SPEC §9 configured values!"
    print("\n[VERDICT: SECTION A PASS]")
    return True


def verify_section_b_penalty_matrix() -> bool:
    """SECTION B — Penalty Matrix across Action and CustomerTier."""
    print("\n" + "=" * 80)
    print("SECTION B — PENALTY MATRIX Omega(a, CustomerTier)")
    print("=" * 80)

    penalty_cfg = PenaltyConfig()
    tiers = [CustomerTier.STANDARD, CustomerTier.PREMIUM, CustomerTier.VIP, CustomerTier.NEW]

    # Header
    header = f"{'Action':<15} | " + " | ".join(f"{t.value:<12}" for t in tiers)
    print(header)
    print("-" * len(header))

    for action in Action:
        row = [f"{action.value:<15}"]
        for tier in tiers:
            pen = penalty_cfg.get_penalty(action, tier)
            row.append(f"INR {pen:<7.3f}")
        print(" | ".join(row))

    # Assertions
    vip_nudge = penalty_cfg.get_penalty(Action.NUDGE, CustomerTier.VIP)
    std_escalate = penalty_cfg.get_penalty(Action.ESCALATE, CustomerTier.STANDARD)
    new_retry_now = penalty_cfg.get_penalty(Action.RETRY_NOW, CustomerTier.NEW)

    assert vip_nudge == Decimal("0.75"), f"Expected VIP Nudge = 0.75, got {vip_nudge}"
    assert std_escalate == Decimal("0.00"), f"Expected Standard Escalate = 0.00, got {std_escalate}"
    assert new_retry_now == Decimal("0.06"), f"Expected New Retry Now = 0.06, got {new_retry_now}"

    print("\n[VERDICT: SECTION B PASS] Spot-checks verified (VIPxNUDGE=0.75, STDxESCALATE=0.00, NEWxRETRY_NOW=0.06)")
    return True


def verify_section_c_ev_demonstration() -> bool:
    """SECTION C — Expected Value Arithmetic Demonstration."""
    print("\n" + "=" * 80)
    print("SECTION C — EXPECTED VALUE ARITHMETIC DEMONSTRATION")
    print("=" * 80)

    payment_amount = Decimal("1000.00")
    tier = CustomerTier.STANDARD

    context = PaymentContext(
        payment_id="pay_demo_1000",
        customer_id="cust_demo_1000",
        customer_tier=tier,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )

    probabilities = {
        Action.RETRY_NOW: ProbabilityEstimate(Action.RETRY_NOW, Decimal("0.60")),
        Action.RETRY_LATER: ProbabilityEstimate(Action.RETRY_LATER, Decimal("0.50")),
        Action.SEND_LINK: ProbabilityEstimate(Action.SEND_LINK, Decimal("0.40")),
        Action.NUDGE: ProbabilityEstimate(Action.NUDGE, Decimal("0.30")),
        Action.ESCALATE: ProbabilityEstimate(Action.ESCALATE, Decimal("0.80")),
        Action.STOP: ProbabilityEstimate(Action.STOP, Decimal("0.00")),
    }

    engine = DefaultEconomicEngine()
    decision = engine.evaluate_actions(context, payment_amount, probabilities)

    print(f"Scenario: Amount = INR {payment_amount:.2f}, Customer Tier = {tier.value}\n")
    print(f"{'Action':<15} | {'Prob P(a)':<10} | {'Gross EV':<12} | {'Cost C(a)':<10} | {'Penalty':<10} | {'Net EV':<12}")
    print("-" * 80)

    for cand in decision.candidate_evaluations:
        print(
            f"{cand.action.value:<15} | "
            f"{cand.estimated_probability:<10.2%} | "
            f"INR {cand.gross_expected_value:<8.2f} | "
            f"INR {cand.intervention_cost:<6.2f} | "
            f"INR {cand.friction_penalty:<6.2f} | "
            f"INR {cand.net_expected_value:<8.2f}"
        )

    print(f"\nProposed Action: {decision.proposed_action.value}")
    print(f"Rationale: {decision.rationale}")

    # Specific assertions
    assert decision.proposed_action == Action.ESCALATE, f"Expected ESCALATE, got {decision.proposed_action}"
    escalate_eval = decision.best_candidate
    assert escalate_eval is not None
    assert escalate_eval.net_expected_value == Decimal("796.50"), (
        f"Expected net EV 796.50 (0.80*1000 - 3.50 - 0.00), got {escalate_eval.net_expected_value}"
    )

    print("\n[VERDICT: SECTION C PASS]")
    return True


def verify_section_d_stop_fallback() -> bool:
    """SECTION D — STOP Fallback when EV <= 0."""
    print("\n" + "=" * 80)
    print("SECTION D — STOP FALLBACK MECHANISM")
    print("=" * 80)

    context = PaymentContext(
        payment_id="pay_stop_demo",
        customer_id="cust_stop_demo",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="HARD_DECLINE",
        raw_error_message="Card stolen / lost",
        failure_category=FailureCategory.HARD_DECLINE,
        failure_severity=FailureSeverity.FATAL,
    )

    probabilities = {action: ProbabilityEstimate(action, Decimal("0.00")) for action in Action}

    engine = DefaultEconomicEngine()
    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)

    print(f"All candidate probabilities = 0.00")
    print(f"Max candidate net EV: INR {decision.candidate_evaluations[0].net_expected_value:.2f}")
    print(f"Proposed Action: {decision.proposed_action.value}")
    print(f"Rationale: {decision.rationale}")

    assert decision.proposed_action == Action.STOP, f"Expected STOP, got {decision.proposed_action}"
    print("\n[VERDICT: SECTION D PASS]")
    return True


def verify_section_e_custom_threshold() -> bool:
    """SECTION E — Custom min_ev_threshold Behavior."""
    print("\n" + "=" * 80)
    print("SECTION E — CUSTOM EV THRESHOLD BEHAVIOR")
    print("=" * 80)

    threshold = Decimal("100.00")
    custom_cfg = EconomicConfig(min_ev_threshold=threshold)
    engine = DefaultEconomicEngine(config=custom_cfg)

    context = PaymentContext(
        payment_id="pay_thresh_demo",
        customer_id="cust_thresh_demo",
        customer_tier=CustomerTier.STANDARD,
        payment_method=PaymentMethod.CREDIT_CARD,
        raw_error_code="TIMEOUT",
        raw_error_message="Gateway timeout",
        failure_category=FailureCategory.NETWORK_TIMEOUT,
        failure_severity=FailureSeverity.TRANSIENT,
    )

    probabilities = {action: ProbabilityEstimate(action, Decimal("0.50")) for action in Action}
    probabilities[Action.STOP] = ProbabilityEstimate(Action.STOP, Decimal("0.00"))

    decision = engine.evaluate_actions(context, Decimal("100.00"), probabilities)
    best_ev = decision.candidate_evaluations[0].net_expected_value

    print(f"Configured min_ev_threshold: INR {threshold:.2f}")
    print(f"Highest candidate net EV: INR {best_ev:.2f}")
    print(f"Proposed Action: {decision.proposed_action.value}")

    assert best_ev < threshold, f"Candidate EV {best_ev} should be below threshold {threshold}"
    assert decision.proposed_action == Action.STOP, f"Expected STOP fallback, got {decision.proposed_action}"

    print("\n[VERDICT: SECTION E PASS]")
    return True


def verify_section_f_protocol_conformance() -> bool:
    """SECTION F — Protocol Conformance."""
    print("\n" + "=" * 80)
    print("SECTION F — ECONOMIC ENGINE PROTOCOL CONFORMANCE")
    print("=" * 80)

    engine = DefaultEconomicEngine()
    is_conforming = isinstance(engine, EconomicEngine)
    print(f"isinstance(DefaultEconomicEngine(), EconomicEngine): {is_conforming}")

    assert is_conforming, "DefaultEconomicEngine does not satisfy EconomicEngine protocol!"
    print("\n[VERDICT: SECTION F PASS]")
    return True


def verify_section_g_research_limitation() -> None:
    """SECTION G — Explicit Research Limitation Statement."""
    print("\n" + "=" * 80)
    print("SECTION G — SCIENTIFIC RESEARCH LIMITATIONS (Sprint 6 A2 Attribution Finding)")
    print("=" * 80)
    limitation_text = """
SCIENTIFIC FINDING & PARAMETER BOUNDARY CAVEAT:
1. In RecoverIQ's current benchmark regime, ticket values (mean ~INR 3,800) heavily
   dominate intervention costs (max cost = INR 3.50 for ESCALATE, i.e., ~0.09% of payment).
2. As a consequence, Sprint 6 Ablation 2 (NoEcon — greedy probability maximization)
   achieved +298.02 INR higher Net Recovered Value (NRV) than full RecoverIQ across
   5 seeds. Because ticket values are so high, greedy-P over-selects ESCALATE on
   recoverable transactions where EV remains strongly positive despite human costs.
3. The Economic Engine's cost-subtraction term therefore exerts a relatively small
   quantitative selection pressure under these specific synthetic parameter distributions.
4. Per RecoverIQ scientific integrity rules:
   - We DO NOT artificially alter cost or penalty parameters to inflate the economic engine's contribution.
   - We DO NOT alter benchmark seeds or ticket distributions to engineer favorable conclusions.
   - We report this limitation honestly as an empirical property of the synthetic parameter regime.
"""
    print(limitation_text.strip())
    print("\n[VERDICT: SECTION G REPORTED]")


def main() -> int:
    print("=" * 80)
    print("RECOVERIQ SPRINT 8 — ECONOMIC ENGINE COMPREHENSIVE VERIFICATION")
    print("SPEC §8 (Economic Objective), §9 (Action Space), §15 (NRV Framework)")
    print("=" * 80)

    all_ok = True
    all_ok = all_ok and verify_section_a_action_costs()
    all_ok = all_ok and verify_section_b_penalty_matrix()
    all_ok = all_ok and verify_section_c_ev_demonstration()
    all_ok = all_ok and verify_section_d_stop_fallback()
    all_ok = all_ok and verify_section_e_custom_threshold()
    all_ok = all_ok and verify_section_f_protocol_conformance()
    verify_section_g_research_limitation()

    print("\n" + "=" * 80)
    print("ALL 7 SPRINT 8 ECONOMIC VERIFICATION SECTIONS COMPLETED SUCCESSFULLY (EXIT 0)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
