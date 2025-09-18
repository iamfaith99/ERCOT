#!/usr/bin/env julia
using Pkg
Pkg.activate(abspath(joinpath(@__DIR__, "..")))

using Dates
using HTTP
using JSON3
using Logging

push!(LOAD_PATH, abspath(joinpath(@__DIR__, "../src")))
include(joinpath(@__DIR__, "../src/ERCOTPipeline.jl"))
using .ERCOTPipeline

const DB_PATH = abspath(joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb"))

function parse_list(value::Union{Nothing,String})
    if value === nothing || isempty(value)
        return nothing
    end
    return split(value, ',')
end

function scenario_payload(params)
    nodes = parse_list(get(params, "nodes", nothing))
    nodes_limit = get(params, "limit", nothing)
    nodes_limit = nodes_limit === nothing ? nothing : tryparse(Int, nodes_limit)
    topk = get(params, "topk", "3")
    top_constraints = tryparse(Int, topk)
    top_constraints === nothing && (top_constraints = 3)

    summary = scenario_summary(DB_PATH; nodes_filter = nodes, nodes_limit = nodes_limit, top_constraints = top_constraints)
    return JSON3.write(summary)
end

function handle(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    params = Dict(HTTP.URIs.queryparams(uri))
    if uri.path == "/scenario"
        body = scenario_payload(params)
        return HTTP.Response(200, body, ["Content-Type" => "application/json"])
    elseif uri.path == "/healthz"
        metadata = load_metadata(DB_PATH)
        status = metadata === nothing ? "no-metadata" : string(metadata.run_ts)
        return HTTP.Response(200, status)
    else
        return HTTP.Response(404, "not found")
    end
end

function main()
    port = parse(Int, get(ENV, "PTDF_SERVICE_PORT", "8080"))
    host = get(ENV, "PTDF_SERVICE_HOST", "0.0.0.0")
    @info "Starting PTDF scenario service" host port
    HTTP.serve(handle, host, port)
end

main()
