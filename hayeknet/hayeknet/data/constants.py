"""Constants for data column names and standardizations."""
from __future__ import annotations

# Standard column names
COL_TIMESTAMP = "timestamp"
COL_LMP_USD = "lmp_usd"
COL_NET_LOAD_MW = "net_load_mw"
COL_SETTLEMENT_POINT = "settlement_point"

# Alternative column names that may appear in raw data
ALT_LMP_COLUMNS = ["LMP", "lmp", "Settlement Point Price", "Price"]
ALT_TIMESTAMP_COLUMNS = ["SCEDTimestamp", "DeliveryDate", "timestamp", "OperDay"]

