# RecoverIQ Sprint 16 — Live Presentation & Video Recording Flow

A structured 5–8 minute presentation flow designed for screen sharing, video recordings, or live technical committee evaluations.

---

## Presentation Sequence & Script

### 0:00 – 1:00 | Problem & Motivation
1. Open the demo at `http://localhost:8501`.
2. Select **"Presentation Mode"** from the sidebar.
3. Expand **"1. What Problem Are We Solving?"**:
   - *Script:* "RecoverIQ addresses online payment recovery as a sequential decision problem. When a transaction fails, gateways either retry blindly or abandon the payment. RecoverIQ selects among 6 candidate actions across up to 3 sequential attempts, balancing direct recovery chance against customer friction and direct gateway costs."

### 1:00 – 2:30 | The Research Journey (Sprints 13 $\to$ 15)
1. Expand **"2. The Research Journey & Key Discoveries"**:
   - *Script:* "In Sprint 13, we established that finite-horizon Bellman Dynamic Programming captures multi-step option value and endogenously eliminates premature early escalation without hardcoded rules. However, Bellman directly trusts the probability model $\hat{P}(a|x)$."
   - *Script:* "In Sprint 14, we evaluated Model-Free Fitted Q-Iteration under deliberate probability model misspecification (M0–M3). ModelFree achieved 67.80% recovery with zero degradation under severe model error, outperforming Bellman DP."
   - *Script:* "In Sprint 15, we combined both into RecoverIQ-Hybrid. Crucially, we discovered a key scientific finding: heuristic uncertainty estimation based on probability decisiveness becomes 'confidently wrong' under severe misspecification, making Equal-Weight Hybrid ($w=0.50$) the superior robust arbitration strategy."

### 2:30 – 4:00 | Live Payment Walkthrough
1. Switch to **"Single Payment Walkthrough"** from the sidebar.
2. Select a sample payment (e.g., Sample Index 0) or customize Amount = ₹4,500, Failure Category = `INSUFFICIENT_FUNDS`, Customer Tier = `STANDARD`.
3. Point out:
   - "Notice that only observable decision-time features are displayed. No hidden ground-truth outcomes are leaked."
4. Select `RecoverIQ-Hybrid-Equal` and click **"Run Multi-Step Trajectory"**:
   - Walk the audience through Attempt 1 (`RETRY_LATER`), Attempt 2 outcome, and the final `RECOVERED` terminal state.
   - Show the final Net Recovered Value (NRV) calculation.

### 4:00 – 5:30 | Strategy Comparison & Terminal Escalation
1. Switch to **"Strategy Comparison"**:
   - Show how different paradigms select different actions for the same payment.
   - Note Attempt 3: pure Bellman collapses toward `ESCALATE` (64.8%), while Hybrid and ModelFree preserve automated recovery retries (`RETRY_LATER` / `SEND_LINK`).
2. Switch to **"Research Dashboards"**:
   - Display the **Model Error (M0–M3)** line chart: show Bellman degrading -5.48% at M3 while ModelFree remains completely flat (0.00% degradation).
   - Display the **Paired CRN Statistics**: highlight that ModelFree vs Bellman shows a statistically significant lift of +₹454.11/payment with a 95% bootstrap CI of [+₹216.42, +₹707.12].

### 5:30 – 6:30 | Scientific Integrity & Next Steps
1. Return to **"Presentation Mode"** and expand **"5. Scientific Limitations & Future Research"**:
   - *Script:* "RecoverIQ follows strict scientific rigor. We report our negative findings: Sprint 15's uncertainty-aware heuristic did not beat equal weighting. This directly motivates our next research direction: implementing conformal prediction and epistemic ensembling for provably valid uncertainty bounds."
2. Conclude the demonstration.
