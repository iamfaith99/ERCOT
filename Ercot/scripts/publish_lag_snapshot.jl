#!/usr/bin/env julia

using Pkg

const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

if get(ENV, "PUBLISH_LAG_INSTANTIATE", "true") != "false"
    Pkg.instantiate()
end

push!(LOAD_PATH, abspath(joinpath(ROOT, "src")))

using Dates
using ERCOTPipeline

const DB_PATH = abspath(joinpath(ROOT, "data", "duckdb", "ercot.duckdb"))
const SOURCE = get(ENV, "LAG_SNAPSHOT_SOURCE", "lagged_webapp")

snapshot = publish_lag_snapshot!(DB_PATH; source = SOURCE)

if snapshot === nothing
    @info "No snapshot published" db_path = DB_PATH
else
    snapshot_date, latest_minute, records = snapshot
    @info "Published lag snapshot" snapshot_date latest_minute records source = SOURCE
end
