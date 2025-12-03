"""ERCOT API endpoints and schema mappings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ERCOTEndpoints:
    """Real ERCOT API endpoints and data product identifiers."""
    
    # Base URLs
    data_portal_base: str = "https://data.ercot.com/api/1/"
    api_explorer_base: str = "https://apiexplorer.ercot.com/api/"
    mis_reports_base: str = "https://mis.ercot.com/misapp/GetReports.do"
    ice_doc_list_base: str = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
    ice_doc_download_base: str = "https://www.ercot.com/misdownload/servlets/mirDownload"
    
    # Report Type IDs for historical data
    report_type_ids: Dict[str, str] = field(default_factory=lambda: {
        "real_time_lmp": "12300",  # LMPs by Resource Nodes, Load Zones and Trading Hubs
        "real_time_load": "13071",  # Real-Time Load Data
        "system_lambda": "12301",  # System Lambda
        "ancillary_prices": "13060",  # AS Prices
        "dam_lmp": "12301",  # Day-Ahead Market LMPs
    })
    
    # Key data products (EMIL IDs)
    real_time_lmp: str = "np6-788-cd"  # LMPs by Resource Nodes, Load Zones and Trading Hubs
    real_time_load: str = "np3-910-er"  # 2-Day Real Time Gen and Load Data Reports
    system_lambda: str = "np6-905-cd"   # Real-Time System Lambda
    ancillary_prices: str = "np6-787-cd"  # Real-Time Ancillary Service Prices
    sced_gen_dispatch: str = "np6-323-cd"  # SCED Resource Data
    rtc_lmp: str = "np6-788-rtc"  # RTC Market Trials LMPs (when available)
    
    # Data formats
    formats: List[str] = field(default_factory=lambda: ["csv", "xml", "zip"])


@dataclass 
class ERCOTSchemaMapping:
    """Schema mappings for ERCOT data products to standardized HayekNet format."""
    
    # NP6-788-CD: Real-Time LMPs schema
    lmp_schema: Dict[str, str] = field(default_factory=lambda: {
        "DeliveryDate": "delivery_date",
        "DeliveryHour": "delivery_hour", 
        "DeliveryInterval": "delivery_interval",
        "SettlementPoint": "settlement_point",
        "SettlementPointName": "settlement_point_name",
        "SettlementPointType": "settlement_point_type",
        "LMP": "lmp_usd",
        "EnergyComponent": "energy_component",
        "CongestionComponent": "congestion_component", 
        "LossComponent": "loss_component"
    })
    
    # NP3-910-ER: Real-Time Load schema
    load_schema: Dict[str, str] = field(default_factory=lambda: {
        "OperDay": "oper_day",
        "HourEnding": "hour_ending",
        "COAST": "coast_load_mw",
        "EAST": "east_load_mw", 
        "FWEST": "far_west_load_mw",
        "NORTH": "north_load_mw",
        "NCENT": "north_central_load_mw",
        "SOUTH": "south_load_mw",
        "SCENT": "south_central_load_mw",
        "WEST": "west_load_mw",
        "ERCOT": "total_load_mw"
    })
    
    # NP6-787-CD: Ancillary Services schema  
    ancillary_schema: Dict[str, str] = field(default_factory=lambda: {
        "DeliveryDate": "delivery_date",
        "HourEnding": "hour_ending",
        "AncillaryType": "ancillary_type",
        "MCPC": "market_clearing_price"
    })

