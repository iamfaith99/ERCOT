#!/usr/bin/env julia
import DuckDB
using DataFrames

const DBPATH = joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb")
const SQL_FILES = [
    joinpath(@__DIR__, "..", "sql", "create_views.sql"),
    joinpath(@__DIR__, "..", "sql", "effective_ptdf.sql"),
]

function _quote_literal(s::AbstractString)
    return "'" * replace(s, "'" => "''") * "'"
end

function _create_sced_mu_pivot!(db::DuckDB.DB)
    constraints_df = DataFrame(DuckDB.execute(db, "SELECT constraint_name FROM ref.top_constraints ORDER BY abs_mu DESC"))
    if nrow(constraints_df) == 0
        @warn "ref.top_constraints is empty; skipping features.sced_mu pivot"
        return
    end

    literals = join(_quote_literal.(constraints_df.constraint_name), ",\n        ")

    pivot_sql = """
CREATE OR REPLACE TABLE features.sced_mu AS
SELECT *
FROM (
    SELECT
        (STRPTIME(SCEDTimeStamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago') AS sced_ts_utc,
        DATE_TRUNC('minute', STRPTIME(SCEDTimeStamp, '%m/%d/%Y %H:%M:%S') AT TIME ZONE 'America/Chicago') AS sced_ts_utc_minute,
        ConstraintName AS constraint_name,
        ShadowPrice AS shadow_price
    FROM market.sced_shadow_prices
    WHERE ConstraintName IN (SELECT constraint_name FROM ref.top_constraints)
)
PIVOT (
    SUM(shadow_price)
    FOR constraint_name IN (
        $literals
    )
);
"""

    DuckDB.execute(db, pivot_sql)
end

function main()
    db = DuckDB.DB(DBPATH)
    try
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS ref;")
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS features;")
        for sql_path in SQL_FILES
            isfile(sql_path) || continue
            source = read(sql_path, String)
            for raw_stmt in split(source, ';')
                stmt = strip(raw_stmt)
                isempty(stmt) && continue
                DuckDB.execute(db, stmt)
            end
        end

        _create_sced_mu_pivot!(db)
    finally
        close(db)
    end
end

main()
