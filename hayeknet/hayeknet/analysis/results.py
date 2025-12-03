"""Results persistence and provenance tracking for HayekNet simulations."""
from __future__ import annotations

import json
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


@dataclass
class SimulationMetadata:
    """Provenance tracking for simulation runs."""
    
    run_id: str
    timestamp: datetime
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    julia_version: Optional[str] = None
    python_version: Optional[str] = None
    hostname: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: Optional[float] = None
    
    @classmethod
    def capture(cls, config: Dict[str, Any] | None = None) -> SimulationMetadata:
        """Capture current environment metadata."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        timestamp = datetime.now(timezone.utc)
        
        # Capture git state
        git_sha = cls._get_git_sha()
        git_branch = cls._get_git_branch()
        git_dirty = cls._check_git_dirty()
        
        # Capture versions
        python_version = cls._get_python_version()
        julia_version = cls._get_julia_version()
        
        # Capture hostname
        hostname = socket.gethostname()
        
        return cls(
            run_id=run_id,
            timestamp=timestamp,
            git_sha=git_sha,
            git_branch=git_branch,
            git_dirty=git_dirty,
            julia_version=julia_version,
            python_version=python_version,
            hostname=hostname,
            config=config or {},
        )
    
    @staticmethod
    def _get_git_sha() -> Optional[str]:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def _get_git_branch() -> Optional[str]:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @staticmethod
    def _check_git_dirty() -> bool:
        """Check if git working directory is dirty."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return bool(result.stdout.strip())
        except Exception:
            pass
        return False
    
    @staticmethod
    def _get_python_version() -> Optional[str]:
        """Get Python version."""
        import sys
        return sys.version.split()[0]
    
    @staticmethod
    def _get_julia_version() -> Optional[str]:
        """Get Julia version if available."""
        try:
            from juliacall import Main as jl
            return str(jl.VERSION)
        except Exception:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    def to_json(self, path: Path) -> None:
        """Write metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class SimulationResults:
    """Complete results from a HayekNet simulation run."""
    
    # Core outputs
    mean_state: np.ndarray
    analysis_ensemble: np.ndarray
    belief_trace: np.ndarray
    rl_actions: List[float]
    option_prices: List[float]
    constraints_satisfied: List[bool]
    
    # Environment data
    market_data: pd.DataFrame
    beliefs: np.ndarray
    
    # RL training metrics
    rl_metadata: Dict[str, Any]
    
    # Data assimilation metrics
    enkf_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Bayesian inference metrics
    bayesian_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Economic metrics
    economic_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance
    metadata: Optional[SimulationMetadata] = None


@dataclass
class ResultsWriter:
    """Persist simulation results with provenance tracking."""
    
    results_dir: Path
    format: str = "parquet"  # "parquet" or "csv"
    
    def __post_init__(self) -> None:
        """Ensure results directory exists."""
        self.results_dir = self.results_dir.expanduser().resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.format == "parquet" and not HAS_PARQUET:
            print("Warning: pyarrow not available, falling back to CSV format")
            self.format = "csv"
    
    def write(self, results: SimulationResults) -> Path:
        """Write complete simulation results to disk."""
        if results.metadata is None:
            results.metadata = SimulationMetadata.capture()
        
        run_id = results.metadata.run_id
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Write metadata
        results.metadata.to_json(run_dir / "metadata.json")
        
        # Write market data
        self._write_dataframe(
            results.market_data,
            run_dir / f"market_data.{self.format}",
        )
        
        # Write state evolution
        state_df = pd.DataFrame({
            "timestep": np.arange(len(results.mean_state)),
            "mean_state_0": results.mean_state[:, 0] if results.mean_state.ndim > 1 else results.mean_state,
            "mean_state_1": results.mean_state[:, 1] if results.mean_state.ndim > 1 and results.mean_state.shape[1] > 1 else np.nan,
        })
        self._write_dataframe(state_df, run_dir / f"state_evolution.{self.format}")
        
        # Write beliefs
        belief_df = pd.DataFrame({
            "timestep": np.arange(len(results.beliefs)),
            "belief": results.beliefs,
        })
        self._write_dataframe(belief_df, run_dir / f"beliefs.{self.format}")
        
        # Write RL actions if available
        if results.rl_actions:
            actions_df = pd.DataFrame({
                "timestep": np.arange(len(results.rl_actions)),
                "action_mw": results.rl_actions,
            })
            self._write_dataframe(actions_df, run_dir / f"rl_actions.{self.format}")
        
        # Write option prices if available
        if results.option_prices:
            options_df = pd.DataFrame({
                "timestep": np.arange(len(results.option_prices)),
                "option_price": results.option_prices,
            })
            self._write_dataframe(options_df, run_dir / f"option_prices.{self.format}")
        
        # Write constraints
        if results.constraints_satisfied:
            constraints_df = pd.DataFrame({
                "timestep": np.arange(len(results.constraints_satisfied)),
                "satisfied": results.constraints_satisfied,
            })
            self._write_dataframe(constraints_df, run_dir / f"constraints.{self.format}")
        
        # Write metrics
        metrics = {
            "enkf": results.enkf_metrics,
            "bayesian": results.bayesian_metrics,
            "rl": results.rl_metadata,
            "economic": results.economic_metrics,
        }
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Write analysis ensemble as numpy array
        np.save(run_dir / "analysis_ensemble.npy", results.analysis_ensemble)
        
        print(f"✅ Results written to {run_dir}")
        return run_dir
    
    def _write_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        """Write dataframe in the configured format."""
        if self.format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)


@dataclass
class ResultsReader:
    """Read simulation results from disk."""
    
    results_dir: Path
    
    def __post_init__(self) -> None:
        """Resolve results directory path."""
        self.results_dir = self.results_dir.expanduser().resolve()
    
    def list_runs(self) -> List[str]:
        """List all available simulation runs."""
        if not self.results_dir.exists():
            return []
        
        runs = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                runs.append(item.name)
        
        return sorted(runs)
    
    def read(self, run_id: str) -> SimulationResults:
        """Read complete simulation results for a given run."""
        run_dir = self.results_dir / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found in {self.results_dir}")
        
        # Read metadata
        with open(run_dir / "metadata.json") as f:
            meta_dict = json.load(f)
            meta_dict["timestamp"] = datetime.fromisoformat(meta_dict["timestamp"])
            metadata = SimulationMetadata(**meta_dict)
        
        # Read dataframes
        market_data = self._read_dataframe(run_dir, "market_data")
        state_df = self._read_dataframe(run_dir, "state_evolution")
        belief_df = self._read_dataframe(run_dir, "beliefs")
        
        # Read optional outputs
        rl_actions = []
        if (run_dir / "rl_actions.parquet").exists() or (run_dir / "rl_actions.csv").exists():
            actions_df = self._read_dataframe(run_dir, "rl_actions")
            rl_actions = actions_df["action_mw"].tolist()
        
        option_prices = []
        if (run_dir / "option_prices.parquet").exists() or (run_dir / "option_prices.csv").exists():
            options_df = self._read_dataframe(run_dir, "option_prices")
            option_prices = options_df["option_price"].tolist()
        
        constraints_satisfied = []
        if (run_dir / "constraints.parquet").exists() or (run_dir / "constraints.csv").exists():
            constraints_df = self._read_dataframe(run_dir, "constraints")
            constraints_satisfied = constraints_df["satisfied"].tolist()
        
        # Read metrics
        with open(run_dir / "metrics.json") as f:
            metrics = json.load(f)
        
        # Read ensemble
        analysis_ensemble = np.load(run_dir / "analysis_ensemble.npy")
        
        # Reconstruct mean state
        mean_state = state_df[["mean_state_0", "mean_state_1"]].to_numpy()
        
        return SimulationResults(
            mean_state=mean_state,
            analysis_ensemble=analysis_ensemble,
            belief_trace=belief_df["belief"].to_numpy(),
            rl_actions=rl_actions,
            option_prices=option_prices,
            constraints_satisfied=constraints_satisfied,
            market_data=market_data,
            beliefs=belief_df["belief"].to_numpy(),
            rl_metadata=metrics.get("rl", {}),
            enkf_metrics=metrics.get("enkf", {}),
            bayesian_metrics=metrics.get("bayesian", {}),
            economic_metrics=metrics.get("economic", {}),
            metadata=metadata,
        )
    
    def _read_dataframe(self, run_dir: Path, name: str) -> pd.DataFrame:
        """Read dataframe from either parquet or CSV."""
        parquet_path = run_dir / f"{name}.parquet"
        csv_path = run_dir / f"{name}.csv"
        
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found")
