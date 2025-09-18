-- Top binding constraints over the trailing 120 days
CREATE OR REPLACE TABLE ref.top_constraints AS
SELECT
    ConstraintName AS constraint_name,
    COUNT(*) AS bind_ct,
    SUM(ABS(ShadowPrice)) AS abs_mu
FROM market.sced_shadow_prices
WHERE STRPTIME(SCEDTimeStamp, '%m/%d/%Y %H:%M:%S') >= NOW() - INTERVAL '120 days'
GROUP BY 1
ORDER BY abs_mu DESC
LIMIT 300;

-- Congestion-only LMP targets with optional scarcity regressors
CREATE OR REPLACE VIEW features.lmp_target AS
SELECT
    sced_ts_utc,
    settlement_point,
    rt_lmp - COALESCE(system_lambda, 0) AS y_congestion,
    COALESCE(ordc_online_adder + ordc_offline_adder + reliability_adder, 0) AS scarcity_adder,
    COALESCE(mcpc_regup, 0) AS mcpc_regup,
    COALESCE(mcpc_rrs, 0) AS mcpc_rrs
FROM mart.fact_rt_5min;
