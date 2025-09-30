# HayekNet

HayekNet is a hybrid Python-Julia prototype for multi-agent trading experiments targeting ERCOT's Real-Time Co-Optimization (RTC). Python orchestrates data ingestion, Bayesian reasoning, reinforcement learning, and visualization; Julia supplies high-performance kernels for data assimilation, stochastic option pricing, and constraint validation.

## Repository Layout

```
hayeknet/
├── python/              # Python orchestration layer
│   ├── main.py         # Entry point
│   ├── main_with_analysis.py  # ✨ Enhanced runner with analysis
│   ├── results.py      # ✨ Results persistence & provenance
│   ├── analysis.py     # ✨ Statistical analysis & plotting
│   ├── data.py         # Enhanced ERCOT data client
│   ├── agents.py       # Bayesian + RL agents
│   ├── storage.py      # Dashboard data storage
│   └── utils.py        # Julia interop utilities
├── julia/              # Julia performance kernels
│   ├── enkf.jl         # Ensemble Kalman Filter
│   ├── options.jl      # Monte Carlo option pricing
│   └── constraints.jl  # DAG constraint validation
├── envs/
│   └── rtc_env.py      # Gymnasium trading environment
├── scripts/            # ✨ Analysis scripts
│   ├── analyze_latest.py     # Analyze most recent run
│   └── compare_runs.py       # Multi-run comparison
├── examples/           # ✨ Example workflows
│   └── run_experiment.py     # Systematic experiments
├── docs/               # ✨ Documentation & notebooks
│   ├── analysis_template.qmd      # Single-run analysis
│   ├── comparative_analysis.qmd   # Multi-run comparison
│   ├── ANALYSIS_README.md         # Analysis guide
│   └── ERCOT_DATA_CLIENT.md       # Data client docs
├── tests/              # ✨ Test suite
│   ├── test_analysis.py    # Analysis tests
│   └── test_storage.py     # Storage tests
├── runs/               # ✨ Simulation results (auto-created)
├── config/
│   └── ercot_config.json # ERCOT API configuration
├── environment.yml      # Conda environment specification
├── activate_hayeknet.sh # Environment activation script
├── validate_ercot.py   # System validation script
├── Makefile            # Development commands
├── ANALYSIS_SETUP_SUMMARY.md  # ✨ Analysis infrastructure summary
├── SETUP_GUIDE.md      # Complete setup guide
└── README.md           # This file
```

## Getting Started

### Quick Setup

1. **Create conda environment**
   ```bash
   # Create environment from specification
   conda env create -f environment.yml
   
   # Activate environment
   source activate_hayeknet.sh
   
   # Test installation
   python validate_ercot.py
   ```

2. **Run the system**
   ```bash
   python -m python.main
   ```

### Using Makefile Commands

```bash
make help          # Show available commands
make test          # Run validation
make run           # Run basic simulation
make run-analysis  # ✨ Run simulation with full analysis
make analyze       # ✨ Analyze latest run
make compare-runs  # ✨ Compare multiple runs
make docs          # ✨ Render Quarto analysis documents
make jupyter       # Start Jupyter Lab
make clean         # Remove environment
make clean-runs    # ✨ Remove all simulation results
```

### Manual Setup (Advanced)

1. **Conda environment**
   ```bash
   # Create environment from YAML
   conda env create -f environment.yml
   conda activate hayeknet
   ```

2. **Julia environment**
   ```bash
   # Install Julia 1.11+ from https://julialang.org
   julia --project=julia -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
   ```

3. **Verify installation**
   ```bash
   python validate_ercot.py
   python -c "from juliacall import Main as jl; print('✅ Julia integration ready')"
   ```

### Environment Management

```bash
# Activate environment
conda activate hayeknet
# or
source activate_hayeknet.sh

# Deactivate
conda deactivate

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n hayeknet
```

## ✨ Data Analysis Infrastructure

HayekNet now includes a comprehensive analysis framework for simulation results:

### Quick Start

```bash
# Run simulation with automatic analysis
make run-analysis

# Analyze latest run
make analyze

# Compare multiple runs
make compare-runs

# Generate Quarto reports
cd docs && quarto render analysis_template.qmd
```

### Features

- **Results Persistence**: Structured storage with Parquet/CSV, full provenance tracking
- **Statistical Analysis**: 18+ metrics covering market conditions, DA performance, Bayesian inference, RL strategy, and economic outcomes
- **Visualization Suite**: 5 publication-quality plot types (300 DPI)
- **Reproducibility**: Git SHA, Julia/Python versions, config capture, deterministic RNG
- **Comparison Tools**: Multi-run analysis with statistical tests
- **Quarto Notebooks**: Rich HTML/PDF reports with embedded analysis

See `docs/ANALYSIS_README.md` for complete documentation and `ANALYSIS_SETUP_SUMMARY.md` for implementation details.

## Key Modules

### Core System
- `python/data.py`: ERCOT data ingestion with real dashboard API integration
- `julia/enkf.jl`: Ensemble Kalman Filter for state estimation
- `python/agents.py`: Bayesian reasoning and RL agents (PPO)
- `envs/rtc_env.py`: Gymnasium trading environment
- `julia/options.jl`: Monte Carlo option pricing
- `julia/constraints.jl`: DAG constraint validation

### Analysis Infrastructure ✨
- `python/results.py`: Results persistence with provenance tracking
- `python/analysis.py`: Statistical analysis and plotting engine
- `python/main_with_analysis.py`: Enhanced simulation runner
- `scripts/analyze_latest.py`: Analyze most recent run
- `scripts/compare_runs.py`: Multi-run comparison tool
- `docs/*.qmd`: Quarto notebooks for reproducible reports

Adapt and extend as the research project matures.
