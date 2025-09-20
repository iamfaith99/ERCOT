#!/usr/bin/env julia

using Pkg

const ROOT = abspath(joinpath(@__DIR__, ".."))
const WEBAPP_PATH = abspath(joinpath(ROOT, "webapp"))
const LOCAL_DEPOT = abspath(joinpath(ROOT, ".julia_depot"))

mkpath(LOCAL_DEPOT)
ENV["JULIA_DEPOT_PATH"] = LOCAL_DEPOT

Pkg.activate(WEBAPP_PATH)

if get(ENV, "WEBAPP_INSTANTIATE", "true") != "false"
    Pkg.instantiate()
end

push!(LOAD_PATH, abspath(joinpath(WEBAPP_PATH, "src")))

using LaggedWebApp

host = get(ENV, "WEBAPP_HOST", "127.0.0.1")
port = parse(Int, get(ENV, "WEBAPP_PORT", "9000"))
pool = parse(Int, get(ENV, "WEBAPP_DB_POOL", "4"))
db_path = get(ENV, "WEBAPP_DUCKDB_PATH", LaggedWebApp.DEFAULT_DB_PATH)

LaggedWebApp.start(; host = host, port = port, pool_size = pool, db_path = db_path)
