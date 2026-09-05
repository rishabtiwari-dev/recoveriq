"""RecoverIQ — Interactive Research Demonstration & Presentation UI.

Run with:
    streamlit run app/demo.py
"""

from decimal import Decimal
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

# Configure wide layout and academic theme
st.set_page_config(
    page_title="RecoverIQ — Sequential Payment Recovery Research Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from recoveriq.domain.actions import Action
from recoveriq.domain.models import CustomerTier, FailureCategory
from recoveriq.domain.state import PaymentState
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


@st.cache_resource(show_spinner="Initializing RecoverIQ Research Engine & Training Models...")
def get_demo_engine() -> DemoEngine:
    engine = DemoEngine(seed=42, max_attempts=3)
    engine.initialize()
    return engine


engine = get_demo_engine()


# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
st.sidebar.title("⚡ RecoverIQ")
st.sidebar.caption("Sequential Decision Research Platform")

demo_mode = st.sidebar.radio(
    "Select Mode:",
    [
        "Presentation Mode",
        "Single Payment Walkthrough",
        "Strategy Comparison",
        "Research Dashboards (M0–M3 & D0–D3)",
        "System Architecture & Methodology",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Research Highlights:**
- **Sprint 13:** Bellman Finite-Horizon DP
- **Sprint 14:** Model-Free Fitted Q-Iteration
- **Sprint 15:** Uncertainty-Aware Hybrid Policy
- **Benchmark:** 20 Seeds under CRN ($N=5,000$)
"""
)
st.sidebar.info("Demo Environment: Local Synthetic Simulation. Zero external API calls or real financial data accessed.")


# =========================================================================
# SECTION 1: HERO / PROJECT OVERVIEW (Displayed on All Modes)
# =========================================================================
st.title("RecoverIQ: Sequential Payment Recovery via Dynamic Programming & Model-Free Learning")
st.markdown(
    """
*A research-oriented platform demonstrating optimal sequential intervention decisions under model misspecification and distribution shift.*
"""
)

badge_cols = st.columns(7)
badge_cols[0].metric("Decision Horizon", "3 Attempts", "Cooldown 900s")
badge_cols[1].metric("Model-Based", "Bellman DP", "Finite-Horizon")
badge_cols[2].metric("Model-Free", "Fitted Q", "Tabular MC")
badge_cols[3].metric("Hybrid", "Q-Arbitration", "Equal vs Unc")
badge_cols[4].metric("Model Error", "M0 → M3", "±30pp + ESC bias")
badge_cols[5].metric("Shift Scenarios", "D0 → D3", "2x Value / +1 Tier")
badge_cols[6].metric("Validation", "20 Seeds", "CRN Synchronized")

st.markdown("---")


# =========================================================================
# MODE 1: PRESENTATION MODE (Clean, Structured Narrative Flow)
# =========================================================================
if demo_mode == "Presentation Mode":
    st.header("Executive Presentation Flow")
    st.caption("A structured walkthrough for live project presentations, evaluations, and video recording.")

    # 1. Problem Definition
    with st.expander("1. What Problem Are We Solving?", expanded=True):
        st.markdown(
            """
            When an online transaction fails, standard industry gateways either retry immediately (causing merchant friction and cardholder fatigue) or stop altogether.
            A recovery system faces a **sequential decision problem**:
            - **Candidate Actions:** `RETRY_NOW`, `RETRY_LATER`, `SEND_LINK`, `NUDGE`, `ESCALATE`, `STOP`.
            - **The Dilemma:** An action that looks best *now* (e.g., escalating immediately for high single-step probability) terminates the automated trajectory and destroys the **future option value** of cheap retries.
            """
        )
        st.code(
            """
Failed Payment Event ──> Observable State (Category, Tier, Attempt)
                              │
                              ▼
                 [ Sequential Decision Engine ]
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           RETRY_NOW     RETRY_LATER     SEND_LINK ...
               │              │              │
               └──────────────┴──────────────┘
                              ▼
            Simulated Outcome ──> Next Attempt or Terminal (Recovered / Escalated / Stopped)
            """,
            language="text",
        )

    # 2. Research Discoveries Summary
    with st.expander("2. The Research Journey & Key Discoveries", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Sprint 13: Bellman Option Value")
            st.markdown(
                """
                - Proved that finite-horizon dynamic programming naturally suppresses premature early escalation.
                - **Finding:** Endogenously reduced Attempt 1 escalation to **0.0%** without hardcoded rules.
                """
            )
        with c2:
            st.markdown("#### Sprint 14: Model-Free Robustness")
            st.markdown(
                """
                - Tested vulnerability to probability model misspecification.
                - **Finding:** ModelFree Fitted Q achieved **67.80%** recovery vs. Bellman's **60.44%**, with **0.0%** degradation under severe distortion (M3).
                """
            )
        with c3:
            st.markdown("#### Sprint 15: Hybrid Arbitration")
            st.markdown(
                """
                - Combined Bellman lookahead + ModelFree empirical returns.
                - **Key Discovery:** Self-reported uncertainty based on probability sharpness can become **confidently wrong** under model distortion.
                """
            )

    # 3. Interactive Executive Comparison
    with st.expander("3. Executive Benchmark Summary (20 Seeds, CRN)", expanded=True):
        df_base = pd.DataFrame(BASELINE_BENCHMARK_M0_D0)
        df_display = df_base[["strategy", "recovery_rate", "automated_nrv", "escalation_rate", "avg_attempts"]].copy()
        df_display.columns = ["Strategy", "Recovery Rate (%)", "Automated NRV (INR)", "Escalation Rate (%)", "Avg Attempts"]
        st.dataframe(
            df_display.style.format(
                {
                    "Recovery Rate (%)": "{:.2f}%",
                    "Automated NRV (INR)": "₹{:,.2f}",
                    "Escalation Rate (%)": "{:.2f}%",
                    "Avg Attempts": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    # 4. Terminal Escalation Chart
    with st.expander("4. Terminal Escalation Collapse & Mitigation", expanded=True):
        st.markdown(
            "At Attempt 3 (terminal horizon), pure Bellman DP loses future option value and aggressively escalates. RecoverIQ-Hybrid mitigates this collapse:"
        )
        df_esc = pd.DataFrame(ATTEMPT_3_ACTION_DISTRIBUTION)
        st.bar_chart(df_esc.set_index("strategy")[["escalate", "retry_later", "send_link", "retry_now"]])

    # 5. Scientific Positioning & Limitations
    with st.expander("5. Scientific Limitations & Future Research", expanded=True):
        st.markdown(
            """
            - **No Artificial Claims:** We do not claim universal dominance for any single strategy. ModelFree won on automated NRV because it avoided escalation; under high human-ops capacity, Bellman/Tiered full-system valuation remains competitive.
            - **Why This is Research:** Every conclusion is backed by 20-seed Common Random Numbers paired bootstrap confidence intervals.
            - **Sprint 16 Next Step:** Investigate conformal prediction and epistemic ensembling to provide provably valid uncertainty estimates for hybrid arbitration.
            """
        )


# =========================================================================
# MODE 2: SINGLE PAYMENT WALKTHROUGH
# =========================================================================
elif demo_mode == "Single Payment Walkthrough":
    st.header("Interactive Single-Payment Walkthrough")
    st.markdown("Step through a payment recovery episode and observe decision state transitions under the selected policy.")

    col_ctrl, col_display = st.columns([1, 2])

    with col_ctrl:
        st.subheader("1. Payment Configuration")
        payment_source = st.radio("Payment Source:", ["Held-Out Test Sample", "Custom Observable Context"])

        if payment_source == "Held-Out Test Sample":
            sample_idx = st.number_input("Sample Index (0 to 149):", min_value=0, max_value=149, value=0, step=1)
            active_record = engine.get_sample_payment(sample_idx)
        else:
            cust_amount = st.number_input("Payment Amount (INR):", min_value=100.0, max_value=100000.0, value=4500.0, step=500.0)
            cust_fail = st.selectbox("Failure Category:", list(FailureCategory), index=0)
            cust_tier = st.selectbox("Customer Tier:", list(CustomerTier), index=1)
            active_record = engine.create_custom_payment(
                payment_id="custom-demo-001",
                amount=cust_amount,
                failure_category=cust_fail,
                customer_tier=cust_tier,
            )

        st.markdown("**Observable Features at Decision Time:**")
        st.json(
            {
                "payment_id": active_record.payment_id,
                "amount": float(active_record.amount),
                "failure_category": active_record.failure_category.value,
                "customer_tier": active_record.customer_tier.value,
                "raw_error_code": active_record.raw_error_code,
                "payment_method": active_record.payment_method.value if active_record.payment_method else "UPI",
            }
        )

        st.subheader("2. Select Decision Policy")
        chosen_strategy = st.selectbox("Policy:", list(engine.strategies.keys()), index=4)

        run_btn = st.button("▶ Run Multi-Step Trajectory", type="primary")

    with col_display:
        st.subheader("2. Decision Evaluation & Multi-Attempt Timeline")

        # Step 1 Immediate Decision Inspection
        step1_decisions = engine.compare_decisions_for_payment(active_record, attempt=1)
        selected_decision = step1_decisions[chosen_strategy]

        st.info(f"**Attempt 1 Proposed Action:** `{selected_decision['action']}` (Selected by {chosen_strategy})")

        if run_btn:
            episode = engine.run_full_trajectory(chosen_strategy, active_record)

            st.markdown(f"### Trajectory Outcome: `{episode.terminal_state.value}`")
            metric_cols = st.columns(4)
            metric_cols[0].metric("Final Status", "RECOVERED" if episode.final_recovered else "FAILED/ESCALATED")
            metric_cols[1].metric("Attempts Executed", f"{episode.attempt_count} of 3")
            metric_cols[2].metric("Total Cost & Penalty", f"₹{float(episode.total_cost + episode.total_penalty):,.2f}")
            metric_cols[3].metric("Net Recovered Value (NRV)", f"₹{float(episode.net_recovered_value):,.2f}")

            st.markdown("#### Sequential Attempt Log:")
            for s in episode.steps:
                with st.container():
                    st.markdown(f"**Attempt {s.step_number}:**")
                    st.write(
                        f"- Proposed: `{s.proposed_action.value}` | Authorized: `{s.authorized_action.value}` "
                        f"| Step Outcome: `{'SUCCESS' if s.recovered else 'FAILED'}` "
                        f"| Step Cost: ₹{float(s.step_cost):.2f} | Resulting State: `{s.resulting_state.value}`"
                    )
        else:
            st.caption("Click 'Run Multi-Step Trajectory' above to simulate sequential outcomes through the environment.")


# =========================================================================
# MODE 3: STRATEGY COMPARISON (Side-by-Side on Same Context)
# =========================================================================
elif demo_mode == "Strategy Comparison":
    st.header("Synchronous Decision Comparison on Identical State")
    st.markdown("Inspect how different decision paradigms evaluate the exact same payment context across attempts.")

    c1, c2, c3 = st.columns(3)
    p_amount = c1.number_input("Amount (INR):", min_value=100.0, max_value=100000.0, value=3500.0, step=250.0)
    p_fail = c2.selectbox("Failure Category:", list(FailureCategory), index=0)
    p_tier = c3.selectbox("Customer Tier:", list(CustomerTier), index=1)

    custom_rec = engine.create_custom_payment("comp-001", p_amount, p_fail, p_tier)

    st.markdown("### Decision Breakdown across All 3 Attempts:")
    tabs = st.tabs(["Attempt 1", "Attempt 2", "Attempt 3"])

    for att_idx, tab in enumerate(tabs, start=1):
        with tab:
            decisions = engine.compare_decisions_for_payment(custom_rec, attempt=att_idx)
            df_dec = pd.DataFrame(
                [
                    {
                        "Strategy": name,
                        "Selected Action": info["action"],
                        "Q / Score": f"{info.get('q_value', info.get('q_hybrid', 'N/A'))}",
                    }
                    for name, info in decisions.items()
                ]
            )
            st.dataframe(df_dec, use_container_width=True)


# =========================================================================
# MODE 4: RESEARCH DASHBOARDS (Verified Experimental Matrices)
# =========================================================================
elif demo_mode == "Research Dashboards (M0–M3 & D0–D3)":
    st.header("Verified Empirical Research Dashboards")
    st.caption("Authoritative data compiled across 20 canonical seeds with Common Random Numbers (5,000 evaluations per cell).")

    dash_tabs = st.tabs(
        [
            "Model Error (M0–M3)",
            "Distribution Shift (D0–D3)",
            "Statistical CIs (Paired CRN)",
            "Hypotheses Scientific Matrix",
        ]
    )

    with dash_tabs[0]:
        st.subheader("Model Misspecification Sensitivity (M0–M3)")
        st.markdown(
            """
            - **M0:** Correct Trained Model
            - **M1:** Mild (±10pp calibration squeeze toward 0.50)
            - **M2:** Moderate (±20pp calibration squeeze)
            - **M3:** Severe (±30pp squeeze + 0.20 additive ESCALATE bias)
            """
        )
        df_me = pd.DataFrame(MODEL_ERROR_RESULTS)
        st.dataframe(
            df_me.style.format(
                {
                    "bellman_nrv": "₹{:,.2f}",
                    "modelfree_nrv": "₹{:,.2f}",
                    "hybrid_unc_nrv": "₹{:,.2f}",
                    "hybrid_eq_nrv": "₹{:,.2f}",
                    "bellman_recovery": "{:.2f}%",
                    "modelfree_recovery": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )
        st.line_chart(df_me.set_index("condition")[["bellman_nrv", "modelfree_nrv", "hybrid_unc_nrv", "hybrid_eq_nrv"]])

    with dash_tabs[1]:
        st.subheader("Distribution Shift Sensitivity (D0–D3)")
        st.markdown(
            """
            - **D0:** In-Distribution Evaluation Set
            - **D1:** Value Shift (Amounts multiplied by 2.0x)
            - **D2:** Profile Shift (Customer tiers elevated by +1 level)
            - **D3:** Combined Shift (D1 + D2 simultaneously applied)
            """
        )
        df_ds = pd.DataFrame(DISTRIBUTION_SHIFT_RESULTS)
        st.dataframe(
            df_ds.style.format(
                {
                    "bellman_nrv": "₹{:,.2f}",
                    "modelfree_nrv": "₹{:,.2f}",
                    "hybrid_unc_nrv": "₹{:,.2f}",
                    "bellman_recovery": "{:.2f}%",
                    "modelfree_recovery": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )
        st.bar_chart(df_ds.set_index("shift")[["bellman_nrv", "modelfree_nrv", "hybrid_unc_nrv"]])

    with dash_tabs[2]:
        st.subheader("Paired Counterfactual CRN Differences & Bootstrap 95% CIs")
        df_paired = pd.DataFrame(PAIRED_CRN_STATISTICS)
        st.dataframe(
            df_paired.style.format(
                {
                    "mean_diff_per_payment": "₹{:,.2f}",
                    "ci_lower": "₹{:,.2f}",
                    "ci_upper": "₹{:,.2f}",
                    "recovery_lift_pts": "{:+.2f}% pts",
                }
            ),
            use_container_width=True,
        )

    with dash_tabs[3]:
        st.subheader("Scientific Hypotheses Audit (Sprints 14 & 15)")
        df_hyp = pd.DataFrame(RESEARCH_HYPOTHESES_VERDICTS)
        st.dataframe(df_hyp, use_container_width=True)


# =========================================================================
# MODE 5: SYSTEM ARCHITECTURE & METHODOLOGY
# =========================================================================
elif demo_mode == "System Architecture & Methodology":
    st.header("RecoverIQ Research Architecture & Module Contracts")

    st.markdown(
        """
        ### Multi-Step Trajectory Decision Loop:
        """
    )
    st.code(
        """
        Observable Context  ──> Policy Proposes Candidate Action (Argmax Q)
                                           │
                                           ▼
                            [ Invariant Policy Gate ]  <── Clamps invalid attempts or cooldown violations
                                           │
                                           ▼
                             Authorized Recovery Action
                                           │
                                           ▼
                         [ Simulation Environment (CRN) ]
                                           │
                                           ▼
                          Outcome, Direct Cost & Customer Penalty
                                           │
                                           ▼
                       Terminal State Check (RECOVERED / ESCALATED / FAILED)
        """,
        language="text",
    )

    st.markdown(
        """
        ### Formal Scientific Formulations:

        **1. Bellman Finite-Horizon Dynamic Programming:**
        $$Q_t(s, a) = \\text{Immediate\\_EV}(s, a) + (1 - \\hat{P}(a \\mid s)) \\cdot \\max_{a'} Q_{t+1}(s', a')$$

        **2. Model-Free Fitted Q-Iteration:**
        $$\\hat{Q}(s, a) = \\frac{1}{N(s, a)} \\sum_{i=1}^{N(s, a)} G_t^{(i)}$$

        **3. Uncertainty-Aware Hybrid Policy:**
        $$Q_{\\text{hybrid}}(s, a) = w(s, a) Q_{\\text{bellman}}(s, a) + (1 - w(s, a)) Q_{\\text{modelfree}}(s, a)$$
        """
    )

    st.info("Safety Invariant: All policy executions must submit candidate actions through InvariantPolicyGate.")
