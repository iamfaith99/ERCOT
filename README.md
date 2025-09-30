# ERCOT Battery Trading Research

**Graduate Research Project**: Simulating Battery Bidding Strategies in ERCOT's Real-Time Co-Optimization (RTC+B)

**Timeline**: September 29 - December 5, 2025 (67 days)  
**Target**: Graduate thesis submission for ERCOT RTC launch

---

## 🎯 Research Overview

This repository contains a complete research system for simulating and analyzing battery energy storage system (BESS) bidding strategies in ERCOT's wholesale electricity market, with a focus on the upcoming Real-Time Co-Optimization (RTC+B) market design launching December 5, 2025.

### Research Questions

1. **How should a 100MW/400MWh battery participate in ERCOT markets?**
2. **What's the profitability difference between current market and RTC+B?**
3. **How do Bayesian forecasting and reinforcement learning improve bidding?**
4. **What are the optimal SOC management strategies?**

---

## 🔋 HayekNet System

**HayekNet** is a hybrid Python-Julia multi-agent system for ERCOT trading experiments.

### Key Features

- **Battery Model**: 100MW/400MWh BESS with realistic constraints
- **Real Data**: 1.8M+ ERCOT LMP observations and growing
- **Trading Strategies**: Simple arbitrage, ancillary services, co-optimization
- **Advanced Analytics**: Bayesian forecasting, RL agents, risk metrics
- **Daily Workflow**: Automated data collection and research journal generation

### Architecture

- **Python**: Data ingestion, Bayesian reasoning, RL, visualization
- **Julia**: High-performance kernels (EnKF, option pricing, constraints)
- **Real ERCOT Data**: Historical LMP data from ERCOT MIS

---

## 📊 Daily Research Workflow

```bash
cd hayeknet/
make daily
```

**Automatically generates**:
1. Collects latest ERCOT LMP data (~6,000 observations)
2. Runs 8-component HayekNet system
3. Simulates battery trading (arbitrage strategy)
4. Generates research journal with metrics
5. Creates observation template for analysis
6. Tracks progress toward thesis completion

**Output** (3 files daily):
- `research/journal/` - Auto-generated data and metrics
- `research/observations/` - Template for your analysis
- `research/results/` - Structured JSON data

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd hayeknet/
source activate_hayeknet.sh
```

### 2. Run Daily Workflow

```bash
make daily
```

### 3. Fill in Research Observations

```bash
open research/observations/observation_$(date +%Y-%m-%d).md
```

---

## 📂 Repository Structure

```
ERCOT/
├── hayeknet/                      # Main research system
│   ├── python/                    # Python modules
│   │   ├── battery_model.py      # 100MW/400MWh BESS model
│   │   ├── battery_strategy.py   # Trading strategies
│   │   ├── battery_daily_analysis.py  # Daily simulation
│   │   ├── research_observations.py   # Observation tracking
│   │   ├── data.py               # ERCOT data client
│   │   ├── agents.py             # Bayesian + RL agents
│   │   └── analysis.py           # Statistical analysis
│   │
│   ├── julia/                     # Julia performance kernels
│   │   ├── enkf.jl               # Data assimilation (EnKF)
│   │   ├── options.jl            # Option pricing (Monte Carlo)
│   │   └── constraints.jl        # Constraint validation
│   │
│   ├── scripts/                   # Automation scripts
│   │   ├── daily_research_workflow.py  # Main daily workflow
│   │   ├── daily_data_collector.py     # Data collection
│   │   └── ingest_historical_data.py   # Historical ingestion
│   │
│   ├── examples/                  # Example analyses
│   │   ├── battery_sced_vs_rtcb.py     # Strategy comparison
│   │   └── analyze_historical_lmps.py  # Data analysis
│   │
│   ├── docs/                      # Research documentation
│   │   ├── Proposal.md            # Research proposal
│   │   ├── Research paper outline.md   # Paper structure
│   │   └── RESEARCH_ALIGNMENT_ASSESSMENT.md
│   │
│   ├── data/                      # Data storage (local only)
│   │   ├── archive/               # Historical LMP data
│   │   └── reports/               # Downloaded reports
│   │
│   ├── research/                  # Research outputs (local only)
│   │   ├── journal/               # Daily journals
│   │   ├── observations/          # Your observations
│   │   └── results/               # JSON results
│   │
│   ├── Makefile                   # Development commands
│   ├── environment.yml            # Conda environment
│   └── .gitignore                 # Protects private data
│
└── README.md                      # This file
```

---

## 🎓 Research Outputs

### Daily Outputs (67 days → Dec 5)

**Quantitative Data**:
- Battery performance metrics (PnL, SOC, cycles)
- Market conditions (LMP volatility, spreads)
- Strategy effectiveness measures
- 20+ metrics per day

**Qualitative Data**:
- Market characterization (volatility type)
- Hypothesis testing progress (H1, H2, H3)
- Strategy observations
- Research insights for thesis

### By December 5, 2025

- **67 days** of battery simulations
- **~19,000 trading intervals** analyzed
- **~1.3M market observations** collected
- **Complete dataset** for thesis Results chapter
- **Accumulated insights** for Discussion chapter

---

## 📈 Research Hypotheses

### Pre-configured hypothesis testing:

**H1: Price volatility → Profitability**
- Track correlation between market volatility and battery PnL
- Evidence auto-filled daily

**H2: Optimal SOC utilization = 60-80%**
- Evaluate relationship between utilization and cycling efficiency
- Daily SOC tracking

**H3: Spread >$15/MWh required for profitability**
- Determine minimum price spread for arbitrage profitability
- Spread calculation automated

---

## 🔬 Technology Stack

### Languages
- **Python 3.13.7**: Main orchestration
- **Julia 1.11+**: Performance kernels

### Key Libraries

**Python**:
- `pandas`, `numpy`: Data manipulation
- `pymc`: Bayesian inference
- `stable-baselines3`: Reinforcement learning
- `matplotlib`, `seaborn`: Visualization

**Julia**:
- `DataAssim.jl`: Ensemble Kalman Filter
- `Distributions.jl`: Stochastic modeling
- `PythonCall.jl`: Python interop

---

## 📊 Battery Model Specifications

```python
Battery: 100 MW / 400 MWh
├── Power Capacity: 100 MW charge/discharge
├── Energy Capacity: 400 MWh
├── SOC Range: 10% - 90% (320 MWh usable)
├── Round-trip Efficiency: 90% (95% charge × 95% discharge)
├── Initial SOC: 50%
└── Degradation: $5/MWh throughput cost
```

---

## 🎯 Research Alignment

**Graduate Thesis Requirements**: ✅ Fully Aligned

| Proposal Requirement | Implementation | Status |
|---------------------|----------------|---------|
| Battery prototype (100MW/400MWh) | `python/battery_model.py` | ✅ Ready |
| Current vs RTC+B comparison | `examples/battery_sced_vs_rtcb.py` | ✅ Ready |
| Bayesian forecasting | `python/agents.py` (PyMC) | ✅ Ready |
| Reinforcement learning | `python/agents.py` (PPO) | ✅ Ready |
| Profitability analysis | `python/analysis.py` | ✅ Ready |
| Risk metrics (VaR, CVaR) | `python/analysis.py` | ✅ Ready |

---

## 📝 Documentation

### Research Documentation
- **`docs/Proposal.md`** - Research proposal
- **`docs/Research paper outline.md`** - Paper structure
- **`docs/RESEARCH_ALIGNMENT_ASSESSMENT.md`** - Capability assessment
- **`WEEK1_ACTION_PLAN.md`** - Week 1 tasks

### Technical Documentation
- **`docs/ERCOT_DATA_CLIENT.md`** - Data client guide
- **`docs/BATTERY_ANALYSIS_README.md`** - Battery analysis guide
- **`docs/ANALYSIS_README.md`** - Results analysis guide

---

## 🔒 Data Privacy

**Code is public** (GitHub) ✅  
**Data is private** (local only) ✅

### Protected by .gitignore:
- Research journals and observations
- ERCOT data archives (1.8M+ observations)
- JSON results files
- Model checkpoints
- All personal research data

---

## 🎓 For Graduate Students

This repository demonstrates:
- ✅ Reproducible research workflow
- ✅ Clean code/data separation
- ✅ Automated data collection
- ✅ Structured observation tracking
- ✅ Version-controlled methodology
- ✅ Publication-ready analysis

**Perfect for**:
- Graduate thesis projects
- Academic research papers
- Conference presentations
- Industry collaborations

---

## 📅 Timeline

**Phase 1** (Sep 29 - Oct 5): Setup & Validation
- ✅ System operational
- ✅ Daily workflow running
- ✅ Data collection automated

**Phase 2** (Oct 6 - Nov 15): Data Collection (40 days)
- Daily data ingestion
- Battery simulation
- Observation tracking
- Pattern identification

**Phase 3** (Nov 16 - Nov 30): Analysis & RTC+B (15 days)
- RTC+B approximation
- Strategy comparison
- Statistical analysis
- Results generation

**Phase 4** (Dec 1 - Dec 5): Thesis Writing (5 days)
- Compile results
- Generate figures
- Write paper
- Submit thesis

---

## 🚀 Commands Reference

```bash
# Daily workflow (run every morning)
make daily

# View today's journal
cat research/journal/journal_$(date +%Y-%m-%d).md

# Open observation file
open research/observations/observation_$(date +%Y-%m-%d).md

# View progress
cat research/RESEARCH_PROGRESS.md

# Analyze results
python scripts/analyze_latest.py

# Compare multiple runs
python scripts/compare_runs.py
```

---

## 📖 Citation

If you use this research system, please cite:

```
HayekNet Battery Trading Research System
Graduate Research Project, 2025
https://github.com/iamfaith99/ERCOT
```

---

## 📧 Contact

**Graduate Researcher**: Weldon  
**Research Focus**: Battery bidding strategies in ERCOT RTC+B  
**Timeline**: September 29 - December 5, 2025  

---

## ⚖️ License

Research code is provided for academic use. Market data is sourced from ERCOT MIS public reports.

---

**System Status**: ✅ Operational  
**Data Collection**: Active  
**Timeline**: On Track  
**Next Milestone**: 67 days of data by Dec 5, 2025  

🔋 **Ready for graduate research!**
