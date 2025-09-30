"""Interop helpers bridging Python and Julia components."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from juliacall import Main as jl
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError("juliacall is required for Julia interoperability") from exc


@lru_cache(maxsize=1)
def init_julia(project_root: Path | None = None):
    """Configure Julia once per process and return the global `Main` handle."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    julia_root = project_root / "julia"
    jl.seval(f'using Pkg; Pkg.activate("{julia_root}")')

    include_julia_sources(
        "julia/enkf.jl",
        "julia/options.jl",
        "julia/constraints.jl",
    )
    return jl


def include_julia_sources(*relative_paths: str) -> None:
    """Include Julia source files relative to the repository root."""
    repo_root = Path(__file__).resolve().parents[1]
    for rel in relative_paths:
        jl.include(str(repo_root / rel))


def run_enkf(
    prior: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    inflation: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Call the Julia EnKF update and map the results back to NumPy arrays."""
    init_julia()

    # Convert numpy arrays to Julia arrays
    julia_prior = prior
    julia_obs = observations  
    julia_H = H
    julia_R = R

    mean_state, ensemble = jl.HayekNetEnKF.update_enkf(
        julia_prior,
        julia_obs,
        H=julia_H,
        R=julia_R,
        inflation=inflation,
    )

    return np.array(mean_state, dtype=float).squeeze(), np.array(ensemble, dtype=float)


def price_option(
    S0: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    *,
    steps: int = 64,
    trajectories: int = 1024,
    seed: int | None = None,
) -> float:
    """Wrapper around the Julia Monte Carlo call option pricer."""
    init_julia()
    mean_value, _ = jl.HayekNetOptions.price_call_option(
        S0,
        strike,
        rate,
        volatility,
        maturity,
        steps=steps,
        trajectories=trajectories,
        seed=seed,
    )
    return float(mean_value)


def validate_constraints(edge_list, source: int, sink: int, capacity: float, bid: float) -> bool:
    """Expose the Julia DAG constraint checker to Python code."""
    init_julia()
    return bool(
        jl.HayekNetConstraints.validate_dag_constraints(
            edge_list,
            source,
            sink,
            capacity,
            bid,
        )
    )

