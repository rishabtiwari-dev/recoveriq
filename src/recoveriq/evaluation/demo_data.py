"""Sprint 16 — Verified Research Benchmark Results Cache.

Stores verified empirical experimental metrics and paired statistical outputs
from the 20-seed Common Random Numbers experiments completed in Sprints 13, 14, and 15.
Used by app/demo.py to render authoritative research dashboards without requiring
re-computation of the 5,000-trajectory 20-seed benchmark on every page render.
"""

from typing import Any, Dict, List

# =========================================================================
# TABLE 1: Clean Baseline Benchmark (M0, D0 across 20 Seeds)
# =========================================================================
BASELINE_BENCHMARK_M0_D0: List[Dict[str, Any]] = [
    {
        "strategy": "Always-Stop",
        "recovery_rate": 0.00,
        "recovery_rate_std": 0.00,
        "avg_attempts": 1.00,
        "automated_nrv": 0.00,
        "automated_nrv_std": 0.00,
        "escalation_rate": 0.00,
        "escalation_rate_std": 0.00,
    },
    {
        "strategy": "Fixed-Retry",
        "recovery_rate": 55.68,
        "recovery_rate_std": 2.52,
        "avg_attempts": 1.97,
        "automated_nrv": 1262492.53,
        "automated_nrv_std": 245134.66,
        "escalation_rate": 0.00,
        "escalation_rate_std": 0.00,
    },
    {
        "strategy": "Rule-Based",
        "recovery_rate": 61.90,
        "recovery_rate_std": 3.38,
        "avg_attempts": 1.83,
        "automated_nrv": 1350901.24,
        "automated_nrv_std": 220850.11,
        "escalation_rate": 0.00,
        "escalation_rate_std": 0.00,
    },
    {
        "strategy": "RecoverIQ-Unconstrained",
        "recovery_rate": 19.99,
        "recovery_rate_std": 11.60,
        "avg_attempts": 1.26,
        "automated_nrv": 601799.82,
        "automated_nrv_std": 305775.93,
        "escalation_rate": 71.88,
        "escalation_rate_std": 13.34,
    },
    {
        "strategy": "RecoverIQ-Tiered",
        "recovery_rate": 60.44,
        "recovery_rate_std": 2.93,
        "avg_attempts": 1.87,
        "automated_nrv": 1366873.58,
        "automated_nrv_std": 235935.44,
        "escalation_rate": 22.50,
        "escalation_rate_std": 5.64,
    },
    {
        "strategy": "RecoverIQ-Bellman",
        "recovery_rate": 60.44,
        "recovery_rate_std": 2.93,
        "avg_attempts": 1.87,
        "automated_nrv": 1367896.21,
        "automated_nrv_std": 235279.97,
        "escalation_rate": 22.48,
        "escalation_rate_std": 5.64,
    },
    {
        "strategy": "RecoverIQ-ModelFree",
        "recovery_rate": 67.80,
        "recovery_rate_std": 2.25,
        "avg_attempts": 2.00,
        "automated_nrv": 1481173.68,
        "automated_nrv_std": 218892.84,
        "escalation_rate": 0.00,
        "escalation_rate_std": 0.00,
    },
    {
        "strategy": "RecoverIQ-Hybrid-Uncertainty",
        "recovery_rate": 62.81,
        "recovery_rate_std": 2.25,
        "avg_attempts": 1.94,
        "automated_nrv": 1345807.93,
        "automated_nrv_std": 237403.22,
        "escalation_rate": 10.34,
        "escalation_rate_std": 2.94,
    },
    {
        "strategy": "RecoverIQ-Hybrid-Equal",
        "recovery_rate": 68.87,
        "recovery_rate_std": 2.57,
        "avg_attempts": 1.98,
        "automated_nrv": 1476069.85,
        "automated_nrv_std": 229886.80,
        "escalation_rate": 3.99,
        "escalation_rate_std": 1.97,
    },
    {
        "strategy": "RecoverIQ-Hybrid-Fixed",
        "recovery_rate": 68.55,
        "recovery_rate_std": 1.81,
        "avg_attempts": 1.96,
        "automated_nrv": 1457863.04,
        "automated_nrv_std": 198895.19,
        "escalation_rate": 8.02,
        "escalation_rate_std": 2.72,
    },
]

# =========================================================================
# TABLE 2: Model Error Sensitivity (M0–M3 across 20 Seeds)
# =========================================================================
MODEL_ERROR_RESULTS: List[Dict[str, Any]] = [
    {
        "condition": "M0 (Correct)",
        "bellman_nrv": 1367896.21,
        "modelfree_nrv": 1481173.68,
        "hybrid_unc_nrv": 1345807.93,
        "hybrid_eq_nrv": 1476069.85,
        "bellman_recovery": 60.44,
        "modelfree_recovery": 67.80,
    },
    {
        "condition": "M1 (Mild ±10pp)",
        "bellman_nrv": 1367946.97,
        "modelfree_nrv": 1481173.68,
        "hybrid_unc_nrv": 1330287.74,
        "hybrid_eq_nrv": 1478005.14,
        "bellman_recovery": 60.49,
        "modelfree_recovery": 67.80,
    },
    {
        "condition": "M2 (Moderate ±20pp)",
        "bellman_nrv": 1367954.84,
        "modelfree_nrv": 1481173.68,
        "hybrid_unc_nrv": 1325242.93,
        "hybrid_eq_nrv": 1471552.57,
        "bellman_recovery": 60.51,
        "modelfree_recovery": 67.80,
    },
    {
        "condition": "M3 (Severe ±30pp + ESCALATE bias)",
        "bellman_nrv": 1292907.58,
        "modelfree_nrv": 1481173.68,
        "hybrid_unc_nrv": 1206480.09,
        "hybrid_eq_nrv": 1420931.84,
        "bellman_recovery": 56.79,
        "modelfree_recovery": 67.80,
    },
]

# Relative Degradation from M0 to M3
DEGRADATION_M0_TO_M3: Dict[str, Dict[str, float]] = {
    "RecoverIQ-Bellman": {"delta_nrv": -74988.63, "rel_pct": -5.48},
    "RecoverIQ-ModelFree": {"delta_nrv": 0.00, "rel_pct": 0.00},
    "RecoverIQ-Hybrid-Uncertainty": {"delta_nrv": -139327.84, "rel_pct": -10.35},
    "RecoverIQ-Hybrid-Equal": {"delta_nrv": -55138.01, "rel_pct": -3.74},
}

# =========================================================================
# TABLE 3: Distribution Shift Sensitivity (D0–D3 across 20 Seeds)
# =========================================================================
DISTRIBUTION_SHIFT_RESULTS: List[Dict[str, Any]] = [
    {
        "shift": "D0 (In-Distribution)",
        "bellman_nrv": 1367896.21,
        "modelfree_nrv": 1481173.68,
        "hybrid_unc_nrv": 1345807.93,
        "bellman_recovery": 60.44,
        "modelfree_recovery": 67.80,
    },
    {
        "shift": "D1 (Value Shift 2x)",
        "bellman_nrv": 2701062.23,
        "modelfree_nrv": 2962497.22,
        "hybrid_unc_nrv": 2379416.53,
        "bellman_recovery": 60.68,
        "modelfree_recovery": 67.80,
    },
    {
        "shift": "D2 (Profile Shift +1 Tier)",
        "bellman_nrv": 1379724.03,
        "modelfree_nrv": 1522585.80,
        "hybrid_unc_nrv": 1421396.07,
        "bellman_recovery": 61.38,
        "modelfree_recovery": 69.46,
    },
    {
        "shift": "D3 (Combined D1+D2)",
        "bellman_nrv": 2757878.68,
        "modelfree_nrv": 3045333.84,
        "hybrid_unc_nrv": 2677715.68,
        "bellman_recovery": 61.07,
        "modelfree_recovery": 69.46,
    },
]

# =========================================================================
# TABLE 4: Paired CRN Comparisons & 95% Bootstrap Confidence Intervals
# =========================================================================
PAIRED_CRN_STATISTICS: List[Dict[str, Any]] = [
    {
        "comparison": "ModelFree vs Bellman",
        "mean_diff_per_payment": 454.11,
        "ci_lower": 216.42,
        "ci_upper": 707.12,
        "recovery_lift_pts": 7.38,
        "significant": True,
    },
    {
        "comparison": "ModelFree vs Tiered",
        "mean_diff_per_payment": 458.21,
        "ci_lower": 224.37,
        "ci_upper": 710.47,
        "recovery_lift_pts": 7.38,
        "significant": True,
    },
    {
        "comparison": "ModelFree vs Rule-Based",
        "mean_diff_per_payment": 522.24,
        "ci_lower": 256.56,
        "ci_upper": 785.79,
        "recovery_lift_pts": 5.91,
        "significant": True,
    },
    {
        "comparison": "Bellman vs Unconstrained",
        "mean_diff_per_payment": 3071.14,
        "ci_lower": 2754.17,
        "ci_upper": 3407.45,
        "recovery_lift_pts": 40.77,
        "significant": True,
    },
    {
        "comparison": "Bellman vs Rule-Based",
        "mean_diff_per_payment": 68.13,
        "ci_lower": -153.03,
        "ci_upper": 268.11,
        "recovery_lift_pts": -1.46,
        "significant": False,
    },
    {
        "comparison": "Hybrid-Uncertainty vs Bellman",
        "mean_diff_per_payment": -88.55,
        "ci_lower": -303.68,
        "ci_upper": 158.81,
        "recovery_lift_pts": 2.39,
        "significant": False,
    },
    {
        "comparison": "Hybrid-Uncertainty vs ModelFree",
        "mean_diff_per_payment": -542.66,
        "ci_lower": -768.92,
        "ci_upper": -305.60,
        "recovery_lift_pts": -4.99,
        "significant": True,
    },
    {
        "comparison": "Hybrid-Equal vs Hybrid-Uncertainty",
        "mean_diff_per_payment": 522.20,
        "ci_lower": 292.19,
        "ci_upper": 743.41,
        "recovery_lift_pts": 6.07,
        "significant": True,
    },
]

# =========================================================================
# TABLE 5: Attempt 3 Action Distributions (Terminal Escalation Analysis)
# =========================================================================
ATTEMPT_3_ACTION_DISTRIBUTION: List[Dict[str, Any]] = [
    {
        "strategy": "RecoverIQ-Bellman",
        "retry_now": 0.7,
        "retry_later": 21.1,
        "send_link": 11.5,
        "nudge": 1.6,
        "escalate": 64.8,
        "stop": 0.4,
    },
    {
        "strategy": "RecoverIQ-ModelFree",
        "retry_now": 14.6,
        "retry_later": 32.4,
        "send_link": 41.8,
        "nudge": 0.0,
        "escalate": 0.0,
        "stop": 11.2,
    },
    {
        "strategy": "RecoverIQ-Hybrid-Uncertainty",
        "retry_now": 4.9,
        "retry_later": 40.1,
        "send_link": 24.8,
        "nudge": 1.5,
        "escalate": 27.4,
        "stop": 1.2,
    },
    {
        "strategy": "RecoverIQ-Tiered",
        "retry_now": 0.7,
        "retry_later": 21.0,
        "send_link": 11.5,
        "nudge": 1.6,
        "escalate": 64.8,
        "stop": 0.4,
    },
]

# =========================================================================
# TABLE 6: Scientific Hypotheses Summary (Sprint 14 & 15)
# =========================================================================
RESEARCH_HYPOTHESES_VERDICTS: List[Dict[str, str]] = [
    {
        "sprint": "Sprint 14",
        "id": "H1",
        "statement": "Model-Free policy degrades less than Bellman DP under probability misspecification.",
        "verdict": "SUPPORTED",
        "evidence": "ModelFree had 0.0% degradation at M3 vs Bellman -5.48% (-INR 74.9K).",
    },
    {
        "sprint": "Sprint 14",
        "id": "H2",
        "statement": "Model-Free achieves higher automated NRV than Bellman DP under misspecification & clean baseline.",
        "verdict": "SUPPORTED",
        "evidence": "+INR 454.11/payment lift at M0 (95% CI: [+216.42, +707.12]); leads at M3 by +INR 188K/seed.",
    },
    {
        "sprint": "Sprint 14",
        "id": "H3",
        "statement": "Model-Free is less sensitive to probability model error under evaluation distribution shifts (D1–D3).",
        "verdict": "SUPPORTED",
        "evidence": "Under D3, ModelFree scaled by +105.6% vs Bellman +101.6%.",
    },
    {
        "sprint": "Sprint 14",
        "id": "H4",
        "statement": "Multi-step sequential option value preservation dominates single-step stops even under misspecification.",
        "verdict": "SUPPORTED",
        "evidence": "Both sequential policies generated >INR 1.29M NRV at M3 vs Always-Stop INR 0.00.",
    },
    {
        "sprint": "Sprint 15",
        "id": "H1",
        "statement": "Uncertainty-aware Hybrid achieves higher automated NRV than pure Bellman under clean conditions.",
        "verdict": "NOT SUPPORTED",
        "evidence": "Paired difference -INR 88.55/pay (95% CI: [-303.68, +158.81]).",
    },
    {
        "sprint": "Sprint 15",
        "id": "H2",
        "statement": "Uncertainty-aware Hybrid degrades less than Bellman under severe misspecification M3.",
        "verdict": "NOT SUPPORTED",
        "evidence": "Hybrid degraded -10.35% vs Bellman -5.48% due to the 'confidently wrong' model failure mode.",
    },
    {
        "sprint": "Sprint 15",
        "id": "H3",
        "statement": "Uncertainty-aware weighting outperforms fixed and equal-weight hybrid regimes.",
        "verdict": "NOT SUPPORTED",
        "evidence": "Hybrid-Equal (+INR 1.48M) and Hybrid-Fixed (+INR 1.46M) both surpassed Hybrid-Uncertainty (+INR 1.35M).",
    },
    {
        "sprint": "Sprint 15",
        "id": "H4",
        "statement": "Hybrid significantly suppresses Bellman's Attempt 3 terminal escalation collapse.",
        "verdict": "SUPPORTED",
        "evidence": "Attempt 3 ESCALATE dropped from 64.78% (Bellman) to 27.41% (Hybrid-Unc) and 3.99% overall (Hybrid-Eq).",
    },
    {
        "sprint": "Sprint 15",
        "id": "H5",
        "statement": "Hybrid scales more resiliently under combined value/profile shift (D3) than pure Bellman.",
        "verdict": "NOT SUPPORTED",
        "evidence": "Bellman scaled +101.6% (INR 1.37M -> 2.76M); Hybrid scaled +99.0% (INR 1.35M -> 2.68M).",
    },
]
