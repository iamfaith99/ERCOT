# Decentralized Bayesian Market-of-Models Blueprint

Chasing a very juicy synthesis. Here’s a concrete blueprint that fuses Hayek’s “prices-as-knowledge,” de Finetti’s coherence, Bayesian/plausible reasoning, DAGs, data assimilation, RL, and option theory—then lands it straight in ERCOT’s RTC + B world as a practical, decentralized virtual power-trading system.

---

## One-line picture

**A decentralized Bayesian market-of-models**: an ensemble of forecasting “agents” publishes coherent probabilities (as prices) for Boolean events that factor along a DAG of grid/market states. A **data-assimilation engine** keeps the latent grid state fresh every 5 minutes. A **market scoring rule** aggregates beliefs into arbitrage-resistant prices (coherent by design). A **risk-aware RL policy** consumes those prices to place bids/offers/hedges, treating every action’s P\&L as an **option payoff** valued under an internally consistent, Arrow-Debreu-style **risk-neutral measure** implied by the prices.

---

## Why these pieces fit (the philosophical spine)

- **Hayek → prices as compressed knowledge.** Prices are signals that coordinate local knowledge without central command. We’ll literalize this: every model/agent publishes a price for events, and the price vector becomes the actionable shared state. ([American Economic Association][1])
- **de Finetti → coherence or Dutch book.** If your posted odds violate probability axioms on the Boolean algebra of events, an opponent can dutch-book you. We make **coherence** enforceable by design: the market maker (log scoring rule) ensures consistent prices over Boolean combinations and conditionals. ([Wikipedia][2])
- **Jaynes/Cox → plausible reasoning = probability on a Boolean algebra.** This legitimizes running fully Bayesian updates over a logical structure (your Boolean functions/constraints). ([Bayes at Washington University][3])
- **Hanson → LMSR / combinatorial markets** preserve conditional independence relations in DAG-factorized event spaces—exactly the structure we need to scale ERCOT’s combinatorial state. ([Hanson University][4])
- **Option theory → consistent valuation.** Given a coherent set of state prices, we price actions as contingent claims using risk-neutral expectations of payoffs—making the RL policy and the valuation layer speak the same language. ([Wikipedia][5])

---

## ERCOT RTC + B: the constraints your system must “breathe”

- **Co-optimization is inside SCED.** Scarcity lives inside the optimization via **Ancillary Service Demand Curves (ASDCs)**; MCPCs reflect opportunity costs and feed into LMPs. (The simulator deck even walks through MCPC as a shadow price and how energy/AS trade off.) ([ERCOT][6])
- **Batteries are single-model ESRs with SoC in RT optimization.** SCED incorporates SoC, ramp/duration, and will not violate telemetered SoC bounds. Set-Point Deviation (SPD) replaces BPD and tolerances tighten. FRRS is eliminated under RTC+B. ([ERCOT][7])
- **Telemetry expands materially.** New ramp-rate telemetry by AS product, frequency-responsive capability limits, and ESR SoC telemetry all gate awards and must be in your observation model. ([ERCOT][8])
- **DAM changes to align with RT.** DAM gets ASDCs, **AS-only offers**, and ESR **Energy Bid/Offer Curves**; DAM ignores SoC (watch the oversell risk), while RT does not. ([ERCOT][9])
- **Timeline reality.** ERCOT’s public plan targets **Dec 5, 2025** for RTC + B go-live; market trials, telemetry tests, and scorecards run through 2025. ([ERCOT][10])

These facts shape what you can predict, trade, and simulate; the architecture below bakes them in.

---

## Architecture (modular, decentralized, and ERCOT-aware)

### 0) Event algebra & DAG (your “language of uncertainty”)

- **Boolean events** express discrete market structure: *constraint c binds in interval t*; *AS product k clears above x*; *node n LMP in bucket j*; *ESR e hits SoC floor*; *RUC insufficiency flag*, etc.
- Factor the joint over a **Bayesian network**: physics/ops nodes (load, outages, renewables, SoC, shift factors, commitment/availability flags) → market-mechanism nodes (binding constraints, AS awards, MCPC) → price nodes (LMPs, AS prices). Inference rides the DAG. ([Wikipedia][11])

### 1) Data assimilation layer (keeps the latent state honest)

- State model (5-min cadence):
  $x_{t+1} = f(x_t, u_t) + w_t$ with components for load, wind/solar, outages, **SoC dynamics**, and DC-OPF sensitivities.
  Observations $y_t$: telemetry (ESR SoC/max/min, net/gross MW, AS ramp rates), UDSP, awards, LMP/MCPC prints, weather ingest. **Assimilate** with EnKF/L-EnKF or Rao–Blackwellized PF (discrete binding-set + continuous variables). ([arXiv][12])
- Rationale: EnKF scales to huge state vectors, copes well with mildly non-Gaussian forecasting in power systems. ([People @ EECS][13])
- ERCOT alignment: include **AS-specific ramp telemetry** and ESR SoC bounds as hard/soft constraints in the observation operator $h(x)$. ([ERCOT][8])

### 2) Market-of-models (coherence-enforcing aggregator)

- Each model/agent posts a **price** (= probability) for its slice of the event DAG.
- Aggregate with **Logarithmic Market Scoring Rule (LMSR)** over the combinatorial space; use the DAG factorization to keep complexity controlled and maintain conditional independence. The LMSR’s convex cost ensures **no Dutch book**, so beliefs are coherent. ([Hanson University][4])

### 3) Option-valuation / state-price layer (turn prices into risk-neutral measures)

- Treat the LMSR price vector as **approximate Arrow–Debreu state prices** after appropriate normalization; compute implied **risk-neutral probabilities** $Q$ and use them to value any action as a contingent claim:

  $$
  \text{Value}(a) = \mathbb{E}_Q[\text{P\&L}(a,\omega)].
  $$
- This unifies forecasting and valuation; arbitrage checks reduce to state-price consistency. ([Wikipedia][5])

### 4) RL policy (decisions, but Bayesian and safe)

- Use **RL-as-inference / max-entropy RL** so control is a variational inference problem; it plays beautifully with the DAG and the assimilation posterior. ([arXiv][14])
- **Safety & market realism** as constraints: CVaR-constrained policy optimization, budget/credit constraints, telemetry/SoC/duration constraints, SPD penalties, and ERCOT compliance rules as *hard* constraints or Lagrangian penalties. ([stanfordasl.github.io][15])
- Action space by participant type:
  - **Asset-owning QSE**: ESR **Energy Bid/Offer Curves**, resource-specific **AS offers** (RegUp/Down, RRS, ECRS, Non-Spin), **self-provision** where allowed (e.g., NCLRs with UFR), and DAM vs RT shaping. ([ERCOT][9])
  - **Financial trader** (no resources): signals for DAM position-taking, CRR/hedges, bilateral structures; align with RTC market trials data streams. ([ERCOT][16])

---

## Mathematical core (minimal but sharp)

1. **DAG factorization**

   $$
   p(x)=\prod_{v} p(x_v \mid x_{\text{pa}(v)}).
   $$

   Binary binding-set variables make price formation piecewise-linear:
   $ \text{LMP}_b = \lambda - \sum_c \text{SF}_{b,c}\,\text{SP}_c$, where $\text{SP}_c$ is a shadow price active iff constraint $c$ binds.

2. **Coherent pricing** (de Finetti)
   Assign prices $p(E)\in[0,1]$ on the Boolean algebra of events with $p(\neg E)=1-p(E)$, $p(E\vee F)=p(E)+p(F)-p(E\wedge F)$. LMSR’s convex cost $C(q)$ prevents Dutch books while letting agents move the book to their belief. ([Wikipedia][2])

3. **Assimilation update** (EnKF sketch)
   Forecast: $x^-_{t}=f(x_{t-1},u_{t-1})+\epsilon$.
   Analysis: $x^{+}_{t}=x^{-}_{t}+K_t\big(y_t-h(x^{-}_{t})\big)$, with gain $K_t$ from ensemble covariances; encode SoC/duration/ramp/telemetry as $h(\cdot)$ and constraints. ([People @ EECS][13])

4. **Risk-neutral valuation**
   Given normalized state prices $\{\pi(\omega)\}$ with $\sum_\omega \pi(\omega)=\beta$ (discount), $Q(\omega)=\pi(\omega)/\beta$. Price any payoff $H$: $H_0=\beta\,\mathbb{E}_Q[H]$. ([Wikipedia][5])

5. **Safe RL objective**

   $$
   \max_\pi \ \mathbb{E}[\text{P\&L}] \quad \text{s.t.}\quad \text{CVaR}_\alpha(\text{loss})\le \kappa,\ \ \text{feasibility}(x_t,a_t)\in\mathcal{C}_{\text{ERCOT}}.
   $$

   Solve via Lagrangian actor–critic or trust-region CVaR methods. ([stanfordasl.github.io][15])

---

## What “decentralized” means here (and why it helps)

- **Agent per sub-domain**: load, wind/solar, outages, topology, batteries/SoC, transmission constraints, price formation, and even human heuristics—each publishes and trades in the relevant conditional events of the DAG.
- **Mechanism = prediction market**: LMSR makes it cheap for any agent to *move the price* where it has genuine information; others can take the other side. This is Hayek’s knowledge aggregation made precise, and de Finetti’s coherence prevents self-contradiction. ([Hanson University][4])
- **Fault tolerance**: if one agent is wrong, its stake (not the whole system) suffers; prices drift back as others arbitrage the error.

---

## ERCOT-specific modeling notes (pragmatic details)

- **Model ASDCs explicitly.** In both DAM and RT, the ASDCs are now the scarcity engine; RT co-optimization folds AS costs into LMPs and yields MCPCs as real shadow prices. Your simulator must reproduce this relationship; ERCOT’s training deck provides step-by-step examples. ([ERCOT][6])
- **Respect telemetry limits** when sampling feasible awards in the forecast: RegUp/Down/ECRS/Non-Spin ramp rates, frequency-responsive capacity limits, and ESR SoC bounds. These are gating variables for award feasibility. ([ERCOT][8])
- **Single-model ESR in RT, SoC ignored in DAM.** Exploit the **DAM/RT wedge**: DAM clears ESRs without SoC, but SCED enforces SoC. Your policy should monetize the wedge carefully without creating SPD/settlement risk. ([ERCOT][7])
- **Product surface to trade/forecast**: 5-min LMPs by node/hub, MCPCs for RegUp/Down, RRS (with UFR/FFR composition), ECRS, Non-Spin; plus constraint-binding probabilities and reserve sufficiency. Use ERCOT’s **RTC market trials** postings for backtests. ([ERCOT][16])

---

## Implementation outline (Julia-first, reproducible, test-heavy)

Types (sketch):

```julia
struct EventID; name::Symbol; parents::Vector{EventID}; end
struct LMSR{T}; b::T; q::Dict{EventID,Float64}; end                 # market maker
struct DAGModel; nodes::Vector{EventID}; end
struct Assimilator{M<:AbstractModel}; model::M; rng::StableRNG; end # EnKF/PF
struct State; x::NamedTuple; t::DateTime; end
struct Action; bids::NamedTuple; offers::NamedTuple; end
struct Risk; cvar_alpha::Float64; budgets::NamedTuple; end
```

Pipelines:

- **Ingest → Assimilate** (`advance!(assimilator, y_t)`) → **Price** (`post!(lmsr, beliefs)`) → **Value** (`state_prices(lmsr) → V(a)`) → **Policy** (`a_t = π(state_prices, posterior)`) → **Execute/Simulate** → **Score & Learn** (proper scoring + P&L attribution).

Guardrails:

- Unitful quantities (`Unitful.jl`), UTC timestamps, deterministic RNG, pinned toolchains; pre-trade **pure-function risk checks** (SoC/duration/SPD/credit) as CI gates; ≥95% tests including doctests and **JET** type-stability checks.

---

## Minimal experiments to get traction fast

1. **Five-bus sandbox**: Reproduce ERCOT’s RTC+B simulator examples to validate MCPC/LMP interactions under ASDCs; your EnKF should learn to detect binding sets. ([ERCOT][6])
2. **ESR policy wedge**: Train a safe RL policy to shape ESR **Energy Bid/Offer Curves** in DAM (SoC-blind) vs RT (SoC-aware), penalized by expected SPD risk under the assimilation posterior. ([ERCOT][7])
3. **Constraint-aware pricing**: Run a combinatorial market over Boolean *constraint-binds* events; verify DAG-preserving LMSR gives sharper LMP distribution tails than independent bets. ([Hanson University][4])
4. **Backtests with market-trial data**: Calibrate against **RTC Market Trials** DAM/RT postings to estimate liquidity/impact parameters for LMSR (the *b* parameter) and to benchmark Brier scores vs P&L. ([ERCOT][16])

---

## Risks, gotchas, and how this design handles them

- **Dimensionality** (too many events): the **DAG** lets you factor questions and use LMSR modularity; only open markets on binding subsets (e.g., top-K constraints by PTDF sensitivity). ([Hanson University][4])
- **Model error** (renewables, topology): the **assimilation** loop corrects drift every 5 minutes with fresh telemetry (including ESR SoC and AS-ramp capability). ([ERCOT][8])
- **Overfitting / tail risk**: the **CVaR-constrained RL** objective keeps downside contained while still optimizing expected returns. ([stanfordasl.github.io][15])
- **Protocol drift**: anchoring the mechanism to **ERCOT’s official RTC + B documentation** keeps your interface stable as rules finalize toward go-live. ([ERCOT][10])

---

### Why this is more than a neat mash-up

- It **makes Hayek operational**: information enters as **local trades on beliefs**, not as monolithic models.
- It **makes de Finetti enforceable**: the mechanism refuses incoherent odds.
- It **makes option theory actionable**: every decision is priced as a derivative on your event DAG.
- It **makes ERCOT RTC+B tractable**: ASDC-driven scarcity and ESR single-model SoC are first-class state variables, not afterthoughts.

---

## Your Plan

1. **Digest the blueprint**
   - Merge the RTC+B architecture and PTDF learning strategy to enumerate concrete deliverables: data prep extensions, modeling scripts, assimilation hooks, pricing/market layer, RL scaffolding, validation dashboards.

2. **Map deliverables to repo structure**
   - Align outputs with current directories: DuckDB `mart` views, `features` schema tables, Julia modules under `src/`, `scripts/` for jobs.    
   - Flag new components required: `features.sced_mu`, PTDF fitting script (`scripts/fit_effective_ptdfs.jl`), LMSR/coherence module, assimilation engine skeleton, RL/policy module, monitoring notebooks.

3. **Execution order**
   - Extend mart views → add constraint feature tables/views → implement PTDF learner + export → stand up LMSR/coherence market maker service → build assimilation and RL scaffolds → integrate option-style valuation → testing/validation harness.

4. **Dependencies & scheduling**
   - Document table/view prerequisites, telemetry feeds, and modeling inputs.    
   - Define nightly/5-min job cadence: `fetch_and_ingest` → assimilation update → PTDF refresh (daily) → LMSR price reset → risk policy evaluation.    
   - List safety/validation checks (unit tests, regression suites, CVaR and feasibility guards) and monitoring required before production trades.

---

[1]: https://www.aeaweb.org/articles?id=10.1257/0002828041464570 "Hayek, 1945: The Use of Knowledge in Society"
[2]: https://en.wikipedia.org/wiki/Dutch_book "Dutch book"
[3]: https://bayes.wustl.edu/etj/prob/book.pdf "Jaynes: Probability Theory The Logic of Science"
[4]: https://mason.gmu.edu/~rhanson/mktscore.pdf "Hanson: Combinatorial Information Market Design"
[5]: https://en.wikipedia.org/wiki/Risk-neutral_measure "Risk-neutral measure"
[6]: https://www.ercot.com/files/docs/2024/02/20/RTC-B%20Simulator%20Overview.pdf "ERCOT RTC+B Simulator Overview"
[7]: https://www.ercot.com/services/rq/rtcb "ERCOT RTC+B ESR Design"
[8]: https://www.ercot.com/mp/data-products/data-product-details?id=NP6-323-CD "ORDC Telemetry Requirements"
[9]: https://www.ercot.com/files/docs/2024/06/12/RTC-B-DAM-Updates.pdf "RTC+B DAM Updates"
[10]: https://www.ercot.com/files/docs/2024/01/30/RTC-B-Program-Timeline.pdf "RTC+B Program Timeline"
[11]: https://en.wikipedia.org/wiki/Bayesian_network "Bayesian network"
[12]: https://arxiv.org/abs/1906.05433 "EnKF for Power Systems"
[13]: https://people.eecs.berkeley.edu/~sinclair/cs174/EnKF.pdf "Ensemble Kalman Filters in Power Systems"
[14]: https://arxiv.org/abs/1702.07479 "RL as Inference"
[15]: https://stanfordasl.github.io/2019/07/22/cvar-rl/ "CVaR-Constrained RL"
[16]: https://www.ercot.com/files/docs/2024/05/10/RTC-B-Market-Trials-Data.pdf "RTC Market Trials Data"
