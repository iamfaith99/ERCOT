"""Bayesian reasoning components."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pymc as pm
except ImportError:  # pragma: no cover - optional heavy dependency
    pm = None  # type: ignore


@dataclass
class BayesianReasoner:
    """Perform Bayesian plausible reasoning with coherence safeguards."""

    prior_high: float = 0.3
    evidence_noise: float = 0.1
    samples: int = 1000
    tune: int = 500

    def update(self, da_estimate: float) -> float:
        if pm is None:
            raise RuntimeError("pymc is required for Bayesian updates")

        with pm.Model():
            high = pm.Bernoulli("high", self.prior_high)
            mu = pm.math.switch(high, 1.0, -1.0) * da_estimate
            pm.Normal("obs", mu=mu, sigma=self.evidence_noise, observed=da_estimate)
            trace = pm.sample(
                self.samples,
                tune=self.tune,
                progressbar=False,
                chains=1,
                cores=1,
                return_inferencedata=True,
            )

        post = float(trace.posterior["high"].mean().values)
        if not np.isclose(post + (1 - post), 1.0, atol=1e-6):
            raise ValueError("Bayesian update produced incoherent beliefs")
        if post < 0 or post > 1:
            raise ValueError("Posterior probability is outside [0, 1]")
        return post

