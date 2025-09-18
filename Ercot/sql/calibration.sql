CREATE SCHEMA IF NOT EXISTS ref;

CREATE TABLE IF NOT EXISTS ref.logistic_map_params (
  event_kind TEXT,
  scope_key  TEXT,
  center     DOUBLE,
  scale      DOUBLE,
  sample_n   BIGINT,
  brier      DOUBLE,
  fitted_from TIMESTAMPTZ,
  fitted_to   TIMESTAMPTZ,
  updated_ts TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (event_kind, scope_key)
);
