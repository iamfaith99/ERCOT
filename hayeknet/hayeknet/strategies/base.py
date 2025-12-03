"""Base classes for battery bidding strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from hayeknet.core.battery import BatterySimulator
from hayeknet.core.market import BidDecision, MarketDesign


class BiddingStrategy(ABC):
    """Abstract base class for battery bidding strategies."""
    
    @abstractmethod
    def generate_bid(
        self,
        battery: BatterySimulator,
        market_data: pd.Series,
        market_design: MarketDesign,
    ) -> BidDecision:
        """
        Generate bid decision for current interval.
        
        Parameters
        ----------
        battery : BatterySimulator
            Current battery state
        market_data : pd.Series
            Market data for this interval
        market_design : MarketDesign
            Which market design (SCED or RTC+B)
            
        Returns
        -------
        BidDecision
            Bid to submit
        """
        pass
