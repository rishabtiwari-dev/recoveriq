# RecoverIQ Sprint 16 — Interactive Demo Guide

## 1. Installation & Environment Setup

Ensure your local virtual environment has the required dependencies installed:

```bash
# Verify Python and dependencies
python --version
pip install streamlit pandas numpy
```

---

## 2. Launching the Demo Application

Start the Streamlit demonstration server directly from the repository root:

```bash
streamlit run app/demo.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 3. Available Demonstration Modes

Use the sidebar radio selector to switch between the 5 distinct operational modes:

### Mode 1: Presentation Mode
- Designed specifically for live presentations, reviewer walk-throughs, and recorded video demos.
- Guides the viewer through:
  1. Problem formulation and action space
  2. The research journey from Sprint 13 to Sprint 15
  3. Executive comparison across all 10 paradigms
  4. Terminal escalation collapse and how Hybrid mitigates it
  5. Scientific limitations and next steps

### Mode 2: Single Payment Walkthrough
- Select an authentic held-out test payment or configure custom observable attributes (Amount, Failure Category, Customer Tier).
- Choose any policy (`Bellman`, `ModelFree`, `Hybrid-Equal`, etc.).
- Click **Run Multi-Step Trajectory** to watch the simulated environment step through each attempt, logging costs, authorized actions, and terminal recovery status.

### Mode 3: Strategy Comparison
- Evaluates the exact same observable payment context across all policies synchronously.
- Displays selected candidate actions and value scores across Attempts 1, 2, and 3 side-by-side.

### Mode 4: Research Dashboards
- **Model Error (M0–M3):** Explores policy degradation when the probability model is misspecified.
- **Distribution Shift (D0–D3):** Analyzes performance scaling when payment amounts double or customer tiers escalate.
- **Statistical CIs (Paired CRN):** Displays bootstrap 95% confidence intervals across 5,000 counterfactual payment episodes.
- **Scientific Hypotheses Audit:** Shows the exact empirical verdicts for Sprints 14 and 15.

### Mode 5: System Architecture & Methodology
- Displays the mathematical formulations of Bellman DP, Model-Free Fitted Q, and Hybrid Arbitration.
- Explains the strict policy gating and anti-leakage boundaries.

---

## 4. What to Say During Presentations

- **"RecoverIQ is not a single-step classifier."** Emphasize that payment recovery is a sequential decision process where early actions must preserve future option value.
- **"We test against model misspecification."** Highlight that model-based Bellman DP is vulnerable when its probability model is wrong, which motivated Model-Free Fitted Q-Iteration.
- **"We report useful negative findings."** Sprint 15 discovered that self-reported model certainty can become "confidently wrong" under model error, meaning equal-weight hybrid ($w=0.50$) outperforms uncertainty-weighted heuristics.

---

## 5. What NOT to Claim

- **Do NOT claim universal dominance:** ModelFree won on automated NRV because it avoided escalation; under high human-ops capacity, Bellman/Tiered full-system valuation remains competitive.
- **Do NOT claim live payment gateway integration:** All demonstrations execute within the local synthetic simulation environment with strict Common Random Numbers.
- **Do NOT claim conformal prediction is implemented:** Conformal bounds and epistemic ensembling are the recommended future research questions for Sprint 17.
