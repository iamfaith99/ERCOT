# Proposal: Simulating Battery Bidding Strategies in ERCOT Under Real-Time Co-Optimization


### Project Title

*Simulating Battery Bidding Strategies in ERCOT: Anticipating Real-Time Co-Optimization (RTC+B)*

---

### Background & Motivation

On December 5, 2025, ERCOT will implement Real-Time Co-Optimization plus Batteries (RTC+B), fundamentally changing how energy and ancillary services are priced and dispatched. Energy storage resources (ESRs) will be treated as single models capable of both charging and discharging, and ancillary service demand curves (ASDCs) will be co-optimized with energy every five minutes. Batteries are expected to play a critical role in reliability and price formation, yet little analysis exists on how trading strategies might shift once RTC+B goes live.

---

### Research Question

*How would ERCOT battery trading strategies differ under today’s market rules versus the upcoming RTC+B framework, and what quantitative methods best capture this decision-making?*

---

### Objectives

1. Build a prototype model of a utility-scale battery operating in ERCOT.
2. Compare baseline arbitrage and ancillary service strategies under current market design with an RTC+B approximation.
3. Incorporate Bayesian forecasting and reinforcement learning to test advanced strategy design.
4. Evaluate profitability, risk, and system impacts of different approaches.

---

### Data & Tools

* **ERCOT MIS reports** for validation of prices and settlement rules.
* **Coding languages:** Julia (primary) and Python (secondary).

---

### Methodology

1. Model a 100 MW / 400 MWh battery with round-trip efficiency and state-of-charge tracking.
2. Simulate strategies:

   * Energy arbitrage (buy low, sell high).
   * Energy + ancillary service co-optimization using historical MCPCs.
   * Approximate RTC+B co-optimization with ASDCs included.
3. Apply Bayesian forecasting for short-term LMP and MCPC prediction.
4. Train a reinforcement learning agent to maximize PnL given uncertainty and operational constraints.
5. Compare outcomes across case study weeks (normal vs scarcity events).

---

### Deliverables

* Academic paper (8–12 pages) with introduction, literature review, methods, results, and implications.
* Code prototype in Julia/Python (open-source notebook).
* Presentation with charts: SOC trajectories, price forecasts, strategy PnLs.

---

### Timeline

* Weeks 1–2: Data pipeline & battery model.
* Weeks 3–5: Baseline arbitrage and ancillary simulations.
* Weeks 6–7: RTC+B approximation and forecasting methods.
* Weeks 8–9: RL strategy training and case study analysis.
* Week 10: Draft paper and prepare presentation.

---
