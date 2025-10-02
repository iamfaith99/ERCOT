# HayekNet Research Project Configuration

**Graduate Thesis Project**: Battery Energy Storage Optimization in ERCOT RTC Market  
**Student**: Weldon T. Antoine III  
**Institution**: Utah State University  
**Timeline**: October 2025 - December 2025

---

## 📅 Study Period

**Official Start Date**: **October 1, 2025** (Day 1)  
**End Date**: December 6, 2025 (Day 67)  
**Total Duration**: 67 consecutive days

### Key Dates

| Date | Event | Status |
|------|-------|--------|
| Oct 1, 2025 | Day 1 - Study begins | ✅ In Progress |
| Nov 15, 2025 | Day 45 - Mid-point analysis | Scheduled |
| Dec 5, 2025 | Day 66 - ERCOT RTC+B Launch | Scheduled |
| Dec 6, 2025 | Day 67 - Study ends | Scheduled |

---

## 🎯 Research Objectives

1. **Baseline Performance**: Quantify battery arbitrage profitability in current SCED-only market
2. **RL Agent Training**: Train multi-agent system on 67 days of real ERCOT data
3. **Market Evolution**: Document pre-launch market patterns leading up to RTC+B
4. **Strategy Comparison**: Compare rule-based vs. RL-based bidding strategies
5. **Thesis Deliverable**: Complete graduate thesis with empirical results

---

## 🤖 Multi-Agent RL System

### Agents
- **Battery Agent**: 100 MW / 400 MWh BESS
- **Solar Agent**: 200 MW capacity
- **Wind Agent**: 150 MW capacity

### Training Schedule
- **Frequency**: Nightly at 8:00 PM MT
- **Method**: Incremental PPO training on ALL historical data
- **Expected Evolution**:
  - **Week 1**: Exploration phase (~3M observations)
  - **Week 5**: Pattern recognition (~15M observations)  
  - **Week 10**: Mature strategies (~33M observations)

---

## 📊 Data Collection

### Source
- **Primary**: ERCOT MIS Real-Time LMP Reports
- **Report Type**: Settlement Point Prices (Node-level)
- **Frequency**: 5-minute intervals (288 per day)
- **Settlement Points**: 1,053 nodes

### Collection Schedule
- **Time**: 1:00 AM MT daily (automatic)
- **Wake**: Mac wakes at 12:55 AM MT
- **Duration**: ~15 minutes per run
- **Storage**: Parquet format with snappy compression
- **Rationale**: Collects complete previous day (00:00-23:55)

### Expected Volume
- **Per Day**: ~303,000 observations (1,053 nodes × 288 intervals)
- **Total Study**: ~20.3M observations (67 days)

---

## 🔬 Experimental Design

### Hypotheses

**H1: RL Outperforms Heuristics**
- Null: RL agents achieve similar PnL as simple arbitrage rules
- Alternative: RL agents achieve statistically significant higher returns

**H2: Agent Learning Improves Over Time**  
- Null: Agent performance is constant throughout study
- Alternative: Agent performance improves as training data accumulates

**H3: Multi-Agent Coordination Benefits**
- Null: Independent agents perform as well as coordinated system
- Alternative: Coordinated multi-agent system achieves higher portfolio returns

### Metrics
- **Primary**: Daily PnL ($/day)
- **Secondary**: Sharpe ratio, max drawdown, SOC utilization
- **Risk**: Value at Risk (VaR), Conditional VaR (CVaR)
- **Efficiency**: Round-trip efficiency, cycling rate

---

## 📁 Data Management

### Archive Structure
```
data/archive/ercot_lmp/
├── ercot_lmp_2025_10.parquet  # Oct 2025 data
├── ercot_lmp_2025_11.parquet  # Nov 2025 data
└── ercot_lmp_2025_12.parquet  # Dec 2025 data (partial)
```

### Model Checkpoints
```
models/qse_agents/
├── battery_latest.zip  # Current battery agent
├── solar_latest.zip    # Current solar agent
├── wind_latest.zip     # Current wind agent
└── marl_state.json     # Agent beliefs and history
```

### Research Outputs
```
research/
├── journal/              # Daily auto-generated journals
├── observations/         # Manual research observations
├── results/              # Experiment results (JSON)
└── RESEARCH_PROGRESS.md  # Cumulative progress tracker
```

---

## 🔄 Nightly Automation

### Workflow (1:00 AM MT)

1. **Wake Mac** (12:55 AM MT via pmset)
2. **Activate Environment** (hayeknet conda env)
3. **Collect Data** (~5 min)
   - Fetch COMPLETE previous day (00:00-23:55)
   - Append to archive
   - Deduplicate
4. **Train Agents** (~10 min)
   - Load ALL historical data
   - Train battery, solar, wind agents
   - Save updated models
5. **Battery Analysis** (~1 min)
   - Run arbitrage simulation on full day
   - Calculate daily PnL
   - Track SOC utilization
6. **Generate Reports** (~1 min)
   - Create daily journal
   - Update progress tracker
   - Save results JSON
7. **Sleep** - Mac can sleep until next day

**Total Runtime**: ~15 minutes  
**User Intervention**: None (fully automated)  
**Key Advantage**: Each run captures a complete 24-hour period

---

## 📈 Expected Timeline

### Week 1 (Oct 1-7): System Validation
- ✅ Confirm data collection working
- ✅ Agents training successfully
- ✅ No missed days
- **Goal**: Stable baseline established

### Weeks 2-4 (Oct 8-28): Pattern Learning
- Agents identify daily cycles
- Weather-load correlations emerge
- Volatility patterns recognized
- **Goal**: Basic strategies learned

### Weeks 5-8 (Oct 29-Nov 25): Strategy Refinement
- Optimal timing learned
- Risk management improves
- Multi-agent coordination
- **Goal**: Competitive with heuristics

### Weeks 9-10 (Nov 26-Dec 6): Mature Performance
- Stable profitable strategies
- Adapts to regime changes
- Ready for RTC+B launch
- **Goal**: Thesis-quality results

---

## 🎓 Thesis Milestones

| Milestone | Date | Status |
|-----------|------|--------|
| Proposal Defense | Sept 15, 2025 | ✅ Complete |
| Data Collection Start | Oct 1, 2025 | ✅ In Progress |
| Mid-point Analysis | Nov 15, 2025 | Pending |
| RTC+B Launch | Dec 5, 2025 | Pending |
| Data Collection End | Dec 6, 2025 | Pending |
| Draft Chapters | Jan 15, 2026 | Pending |
| Committee Review | Feb 1, 2026 | Pending |
| Final Defense | Mar 1, 2026 | Pending |

---

## 🛠️ Technical Stack

### Languages
- **Python 3.13.7**: Orchestration, ML, data processing
- **Julia 1.11**: High-performance numerical kernels
- **Bash**: Automation scripts

### ML/Stats
- **PyMC 5.25**: Bayesian inference
- **Stable-Baselines3 2.7**: Reinforcement learning (PPO)
- **PyTorch 2.7**: Neural network backend
- **Gymnasium 1.2**: RL environment interface

### Data
- **Pandas 2.2**: Data manipulation
- **NumPy 2.2**: Numerical computing
- **Parquet**: Compressed columnar storage

### Environment
- **Conda**: Package/environment management
- **macOS**: Development platform
- **LaunchD**: Task scheduling

---

## 📝 Notes

**Data Availability**: ERCOT real-time reports have ~18-24 hour retention. Historical archives take 7-30 days to appear. This necessitated waiting until Oct 1 for clean study start.

**Sept 30 Data**: Partial data from Sept 30 afternoon (13:40-23:55) was collected but excluded from study to maintain dataset consistency. Study officially begins Oct 1, 2025 00:00:00.

**Reproducibility**: All code, data, and model checkpoints are version-controlled via Git. Daily logs track exact data collection times, training parameters, and system state.

---

**Last Updated**: October 1, 2025, 8:43 PM MT  
**Status**: Day 1 in progress ✅
