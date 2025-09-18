-- Top binding constraints over the trailing 120 days
CREATE OR REPLACE TABLE ref.top_constraints AS
SELECT
    ConstraintName AS constraint_name,
    COUNT(*) AS bind_ct,
    SUM(ABS(ShadowPrice)) AS abs_mu
FROM market.sced_shadow_prices
WHERE (STRPTIME(SCEDTimeStamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago') >= NOW() - INTERVAL '120 days'
GROUP BY 1
ORDER BY abs_mu DESC
LIMIT 300;

-- Congestion-only LMP targets with optional scarcity regressors
CREATE OR REPLACE VIEW features.lmp_target AS
SELECT
    DATE_TRUNC('minute', sced_ts_utc) AS sced_ts_utc_minute,
    sced_ts_utc,
    settlement_point,
    rt_lmp - COALESCE(system_lambda, 0) AS y_congestion,
    COALESCE(ordc_online_adder + ordc_offline_adder + reliability_adder, 0) AS scarcity_adder,
    COALESCE(mcpc_regup, 0) AS mcpc_regup,
    COALESCE(mcpc_rrs, 0) AS mcpc_rrs,
    COALESCE(mcpc_ecrs, 0) AS mcpc_ecrs,
    COALESCE(mcpc_nspin, 0) AS mcpc_nspin
FROM mart.fact_rt_5min;

CREATE OR REPLACE VIEW ref.ptdf_window AS
SELECT window_start, window_end
FROM ref.estimated_ptdf_metadata
ORDER BY run_ts DESC
LIMIT 1;

CREATE OR REPLACE VIEW ref.constraint_activity AS
WITH win AS (SELECT * FROM ref.ptdf_window)
SELECT
    sp.ConstraintName AS constraint_name,
    AVG(ABS(sp.ShadowPrice)) AS avg_abs_mu
FROM market.sced_shadow_prices AS sp, win
WHERE (STRPTIME(sp.SCEDTimeStamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago')
      BETWEEN win.window_start AND win.window_end
GROUP BY 1;

CREATE OR REPLACE VIEW ref.node_top_constraints AS
SELECT
    e.node,
    e.constraint_name,
    e.beta,
    ca.avg_abs_mu,
    ABS(e.beta) * ca.avg_abs_mu AS avg_abs_contrib,
    ROW_NUMBER() OVER (PARTITION BY e.node ORDER BY ABS(e.beta) * ca.avg_abs_mu DESC) AS rank_for_node
FROM ref.estimated_ptdf AS e
JOIN ref.constraint_activity AS ca USING (constraint_name)
WHERE e.feature_type = 'constraint';
