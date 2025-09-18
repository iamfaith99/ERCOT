CREATE SCHEMA IF NOT EXISTS mart;

CREATE OR REPLACE VIEW mart.fact_rt_5min AS
WITH
    lmp AS (
        SELECT
            strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AS sced_ts_ct,
            strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago' AS sced_ts_utc,
            date_trunc('minute', strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago') AS sced_ts_utc_minute,
            RepeatedHourFlag AS sced_repeated_hour_flag,
            SettlementPoint AS settlement_point,
            LMP AS rt_lmp
        FROM market.rt_lmp_5min
    ),
    ordc_raw AS (
        SELECT
            strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago' AS sced_ts_utc,
            SystemLambda,
            PRC,
            RTORPA,
            RTOFFPA,
            RTORDPA,
            RTOLCAP,
            RTOFFCAP,
            RTOLNSRS
        FROM ancillary.rt_ordc_adders_sced
    ),
    ordc AS (
        SELECT
            date_trunc('minute', sced_ts_utc) AS sced_ts_utc_minute,
            max(sced_ts_utc) AS sced_ts_utc,
            max(SystemLambda) AS system_lambda,
            max(PRC) AS prc,
            max(RTORPA) AS ordc_online_adder,
            max(RTOFFPA) AS ordc_offline_adder,
            max(RTORDPA) AS reliability_adder,
            max(RTOLCAP) AS ordc_online_capacity_mw,
            max(RTOFFCAP) AS ordc_offline_capacity_mw,
            max(RTOLNSRS) AS ordc_online_nonspin_mw
        FROM ordc_raw
        GROUP BY 1
    ),
    mcpc_raw AS (
        SELECT
            strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago' AS sced_ts_utc,
            ASType,
            MCPC
        FROM ancillary.rt_mcpc_sced
    ),
    mcpc AS (
        SELECT *
        FROM (
            SELECT
                date_trunc('minute', sced_ts_utc) AS sced_ts_utc_minute,
                ASType,
                MCPC
            FROM mcpc_raw
        )
        PIVOT (
            max(MCPC) FOR ASType IN ('REGUP', 'REGDN', 'RRS', 'ECRS', 'NSPIN')
        )
    ),
    capability_raw AS (
        SELECT
            strptime(SCEDTimestamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago' AS sced_ts_utc,
            CapREGUPTotal,
            CapREGDNTotal,
            CapRRSTotal,
            CapECRSTotal,
            CapNSPINTotal
        FROM ancillary.rt_total_capability
    ),
    capability AS (
        SELECT
            date_trunc('minute', sced_ts_utc) AS sced_ts_utc_minute,
            max(sced_ts_utc) AS sced_ts_utc,
            max(CapREGUPTotal) AS cap_regup_total,
            max(CapREGDNTotal) AS cap_regdn_total,
            max(CapRRSTotal) AS cap_rrs_total,
            max(CapECRSTotal) AS cap_ecrs_total,
            max(CapNSPINTotal) AS cap_nspin_total
        FROM capability_raw
        GROUP BY 1
    ),
    wind AS (
        SELECT
            date_trunc('minute', strptime(INTERVAL_ENDING, '%m/%d/%Y %H:%M') AT TIME ZONE 'America/Chicago') AS interval_ending_utc,
            avg(SYSTEM_WIDE_GEN) AS wind_system_mw,
            avg(PANHANDLE) AS wind_panhandle_mw,
            avg(COASTAL) AS wind_coastal_mw,
            avg(SOUTH) AS wind_south_mw,
            avg(WEST) AS wind_west_mw,
            avg(NORTH) AS wind_north_mw,
            avg(SYSTEM_WIDE_HSL) AS wind_system_hsl_mw
        FROM context.wind_5min_region
        GROUP BY 1
    ),
    solar AS (
        SELECT
            date_trunc('minute', INTERVAL_ENDING AT TIME ZONE 'America/Chicago') AS interval_ending_utc,
            avg(SYSTEM_WIDE_GEN) AS solar_system_mw,
            avg(SYSTEM_WIDE_HSL) AS solar_system_hsl_mw,
            avg(CenterWest_GEN) AS solar_center_west_mw,
            avg(NorthWest_GEN) AS solar_north_west_mw,
            avg(FarWest_GEN) AS solar_far_west_mw,
            avg(FarEast_GEN) AS solar_far_east_mw,
            avg(SouthEast_GEN) AS solar_south_east_mw,
            avg(CenterEast_GEN) AS solar_center_east_mw
        FROM context.solar_5min_region
        GROUP BY 1
    ),
    load_forecast AS (
        SELECT
            (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
            avg(SystemTotal) AS system_load_forecast_mw,
            avg(Coast) AS load_coast_mw,
            avg(East) AS load_east_mw,
            avg(FarWest) AS load_far_west_mw,
            avg(North) AS load_north_mw,
            avg(NorthCentral) AS load_north_central_mw,
            avg(SouthCentral) AS load_south_central_mw,
            avg(Southern) AS load_southern_mw,
            avg(West) AS load_west_mw
        FROM context.load_forecast_weather_zone
        GROUP BY 1
    ),
    outage AS (
        SELECT
            (
                CAST(Date AS TIMESTAMP)
                + CASE WHEN CAST(HourEnding AS INTEGER) = 24 THEN INTERVAL 1 DAY ELSE INTERVAL 0 DAY END
                + (CASE WHEN CAST(HourEnding AS INTEGER) = 24 THEN 0 ELSE CAST(HourEnding AS INTEGER) END) * INTERVAL '1 hour'
            ) AT TIME ZONE 'America/Chicago' AS he_utc,
            avg(TotalResourceMWZoneSouth) AS outage_resource_south_mw,
            avg(TotalResourceMWZoneNorth) AS outage_resource_north_mw,
            avg(TotalResourceMWZoneWest) AS outage_resource_west_mw,
            avg(TotalResourceMWZoneHouston) AS outage_resource_houston_mw,
            avg(TotalIRRMWZoneSouth) AS outage_irr_south_mw,
            avg(TotalIRRMWZoneNorth) AS outage_irr_north_mw,
            avg(TotalIRRMWZoneWest) AS outage_irr_west_mw,
            avg(TotalIRRMWZoneHouston) AS outage_irr_houston_mw
        FROM context.resource_outage_capacity
        GROUP BY 1
    )
SELECT
    l.sced_ts_ct,
    l.sced_ts_utc,
    l.sced_ts_utc_minute,
    l.sced_repeated_hour_flag,
    l.settlement_point,
    l.rt_lmp,
    ordc.system_lambda,
    ordc.prc,
    ordc.ordc_online_adder,
    ordc.ordc_offline_adder,
    ordc.reliability_adder,
    (ordc.ordc_online_adder + ordc.ordc_offline_adder + ordc.reliability_adder) AS total_price_adder,
    ordc.ordc_online_capacity_mw,
    ordc.ordc_offline_capacity_mw,
    ordc.ordc_online_nonspin_mw,
    mcpc."REGUP" AS mcpc_regup,
    mcpc."REGDN" AS mcpc_regdn,
    mcpc."RRS" AS mcpc_rrs,
    mcpc."ECRS" AS mcpc_ecrs,
    mcpc."NSPIN" AS mcpc_nspin,
    capability.cap_regup_total,
    capability.cap_regdn_total,
    capability.cap_rrs_total,
    capability.cap_ecrs_total,
    capability.cap_nspin_total,
    wind.wind_system_mw,
    wind.wind_panhandle_mw,
    wind.wind_coastal_mw,
    wind.wind_south_mw,
    wind.wind_west_mw,
    wind.wind_north_mw,
    wind.wind_system_hsl_mw,
    solar.solar_system_mw,
    solar.solar_system_hsl_mw,
    solar.solar_center_west_mw,
    solar.solar_north_west_mw,
    solar.solar_far_west_mw,
    solar.solar_far_east_mw,
    solar.solar_south_east_mw,
    solar.solar_center_east_mw,
    load_forecast.system_load_forecast_mw,
    load_forecast.load_coast_mw,
    load_forecast.load_east_mw,
    load_forecast.load_far_west_mw,
    load_forecast.load_north_mw,
    load_forecast.load_north_central_mw,
    load_forecast.load_south_central_mw,
    load_forecast.load_southern_mw,
    load_forecast.load_west_mw,
    outage.outage_resource_south_mw,
    outage.outage_resource_north_mw,
    outage.outage_resource_west_mw,
    outage.outage_resource_houston_mw,
    outage.outage_irr_south_mw,
    outage.outage_irr_north_mw,
    outage.outage_irr_west_mw,
    outage.outage_irr_houston_mw
FROM lmp AS l
LEFT JOIN ordc ON ordc.sced_ts_utc_minute = l.sced_ts_utc_minute
LEFT JOIN mcpc ON mcpc.sced_ts_utc_minute = l.sced_ts_utc_minute
LEFT JOIN capability ON capability.sced_ts_utc_minute = l.sced_ts_utc_minute
LEFT JOIN wind ON wind.interval_ending_utc = l.sced_ts_utc_minute
LEFT JOIN solar ON solar.interval_ending_utc = l.sced_ts_utc_minute
LEFT JOIN load_forecast ON load_forecast.he_utc = date_trunc('hour', l.sced_ts_utc)
LEFT JOIN outage ON outage.he_utc = date_trunc('hour', l.sced_ts_utc);

CREATE OR REPLACE VIEW mart.fact_dam_hourly AS
WITH
    spp AS (
        SELECT
            (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
            DeliveryDate AS delivery_date,
            HourEnding AS hour_ending,
            SettlementPoint AS settlement_point,
            SettlementPointPrice AS dam_settlement_point_price
        FROM market.dam_settlement_point_prices
    ),
    mcpc AS (
        SELECT *
        FROM (
            SELECT
                (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
                AncillaryType,
                MCPC
            FROM ancillary.dam_mcpc
        )
        PIVOT (
            max(MCPC) FOR AncillaryType IN ('REGUP', 'REGDN', 'RRS', 'ECRS', 'NSPIN')
        )
    ),
    purchased AS (
        SELECT
            (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
            Settlement_Point AS settlement_point,
            Total_DAM_Energy_Bought AS dam_energy_bought_mwh
        FROM reference.dam_total_energy_purchased
    ),
    sold AS (
        SELECT
            (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
            Settlement_Point AS settlement_point,
            TotalDAMEnergySold AS dam_energy_sold_mwh
        FROM reference.dam_total_energy_sold
    )
SELECT
    spp.he_utc,
    spp.delivery_date,
    spp.hour_ending,
    spp.settlement_point,
    spp.dam_settlement_point_price,
    mcpc."REGUP" AS dam_mcpc_regup,
    mcpc."REGDN" AS dam_mcpc_regdn,
    mcpc."RRS" AS dam_mcpc_rrs,
    mcpc."ECRS" AS dam_mcpc_ecrs,
    mcpc."NSPIN" AS dam_mcpc_nspin,
    purchased.dam_energy_bought_mwh,
    sold.dam_energy_sold_mwh
FROM spp
LEFT JOIN mcpc USING (he_utc)
LEFT JOIN purchased USING (he_utc, settlement_point)
LEFT JOIN sold USING (he_utc, settlement_point);

CREATE OR REPLACE VIEW mart.dam_hourly_lmp_bus AS
SELECT
    (DeliveryDate + HourEnding) AT TIME ZONE 'America/Chicago' AS he_utc,
    DeliveryDate AS delivery_date,
    HourEnding AS hour_ending,
    BusName AS bus_name,
    LMP AS dam_lmp
FROM market.dam_hourly_lmps;

CREATE SCHEMA IF NOT EXISTS dashboard;

CREATE OR REPLACE VIEW dashboard.node_driver_live AS
WITH latest AS (
    SELECT *
    FROM features.sced_mu
    ORDER BY sced_ts_utc DESC
    LIMIT 1
),
meta AS (
    SELECT sced_ts_utc, sced_ts_utc_minute
    FROM latest
),
mu_values AS (
    SELECT
        key AS constraint_name,
        TRY_CAST(value AS DOUBLE) AS mu_value
    FROM latest, LATERAL json_each(to_json(latest))
    WHERE key NOT IN ('sced_ts_utc', 'sced_ts_utc_minute')
),
base AS (
    SELECT
        ntc.node,
        ntc.constraint_name,
        ntc.rank_for_node,
        ntc.beta,
        ntc.avg_abs_mu,
        ntc.avg_abs_contrib,
        COALESCE(mu.mu_value, 0.0) AS current_mu,
        ntc.beta * COALESCE(mu.mu_value, 0.0) AS live_contribution,
        abs(ntc.beta * COALESCE(mu.mu_value, 0.0)) AS abs_live_contribution,
        meta.sced_ts_utc,
        meta.sced_ts_utc_minute
    FROM ref.node_top_constraints ntc
    LEFT JOIN mu_values mu USING (constraint_name)
    CROSS JOIN meta
),
ranked AS (
    SELECT
        base.*,
        ROW_NUMBER() OVER (PARTITION BY node ORDER BY abs_live_contribution DESC, constraint_name) AS live_rank,
        SUM(abs_live_contribution) OVER (PARTITION BY node) AS total_abs_live_contribution
    FROM base
)
SELECT
    node,
    constraint_name,
    rank_for_node,
    live_rank,
    beta,
    current_mu,
    live_contribution,
    abs_live_contribution,
    total_abs_live_contribution,
    CASE
        WHEN total_abs_live_contribution > 0 THEN abs_live_contribution / total_abs_live_contribution
        ELSE NULL
    END AS live_share,
    CASE
        WHEN current_mu > 0 THEN 'positive'
        WHEN current_mu < 0 THEN 'negative'
        WHEN current_mu = 0 THEN 'flat'
        ELSE 'missing'
    END AS mu_direction,
    avg_abs_mu,
    avg_abs_contrib,
    sced_ts_utc,
    sced_ts_utc_minute
FROM ranked
WHERE live_rank <= 10
ORDER BY node, live_rank;
