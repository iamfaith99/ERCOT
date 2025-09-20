#!/usr/bin/env julia

using Dates
using JSON3
using Logging

const ROOT = abspath(joinpath(@__DIR__, ".."))
const MANIFEST_DIR = joinpath(ROOT, "data", "manifests")

struct PruneOptions
    keep::Union{Nothing,Int}
    datasets::Vector{String}
    dry_run::Bool
end

function parse_args()
    keep = nothing
    datasets = String[]
    dry_run = false
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--keep"
            i += 1
            i > length(ARGS) && error("--keep requires an integer value")
            keep = parse(Int, ARGS[i])
        elseif startswith(arg, "--keep=")
            keep = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("--dry-run", "-n")
            dry_run = true
        elseif startswith(arg, "--")
            error("Unknown flag $(arg)")
        else
            push!(datasets, String(arg))
        end
        i += 1
    end
    return PruneOptions(keep, datasets, dry_run)
end

function manifest_entry_paths(info::Dict{String,Any})
    paths = String[]
    raw = get(info, "raw", nothing)
    if raw !== nothing
        push!(paths, String(raw))
    end
    staged = get(info, "staged", nothing)
    if staged isa AbstractVector
        for item in staged
            push!(paths, String(item))
        end
    elseif staged !== nothing
        push!(paths, String(staged))
    end
    return paths
end

function manifest_entry_timestamp(info::Dict{String,Any})
    if haskey(info, "fetched_at")
        value = try
            DateTime(String(info["fetched_at"]))
        catch
            nothing
        end
        if value !== nothing
            return Dates.datetime2unix(value)
        end
    end
    best = -Inf
    for path in manifest_entry_paths(info)
        if isfile(path)
            best = max(best, stat(path).mtime)
        end
    end
    return best
end

function prune_manifest!(manifest::Dict{String,Any}; keep::Union{Nothing,Int}=nothing)
    files = manifest["files"]::Dict{String,Any}
    removed_missing = 0
    for (url, info) in collect(files)
        staged_paths = String[]
        if haskey(info, "staged")
            for item in info["staged"]
                path = String(item)
                if isfile(path)
                    push!(staged_paths, path)
                end
            end
            if isempty(staged_paths)
                delete!(info, "staged")
            else
                info["staged"] = staged_paths
            end
        end
        raw_path = get(info, "raw", nothing)
        raw_exists = raw_path !== nothing && isfile(String(raw_path))
        staged_exists = haskey(info, "staged") && !isempty(info["staged"])
        if !raw_exists && !staged_exists
            delete!(files, url)
            removed_missing += 1
        end
    end

    trimmed = 0
    keep === nothing && return (removed_missing=removed_missing, trimmed=trimmed)
    keep = max(keep, 0)
    current = length(files)
    if keep < current
        entries = collect(files)
        sort!(entries; by = entry -> manifest_entry_timestamp(entry[2]), rev=true)
        trimmed = current - keep
        manifest["files"] = Dict(entries[1:keep])
    end
    return (removed_missing=removed_missing, trimmed=trimmed)
end

function datasets_to_process(manifest_dir::String, opts::PruneOptions)
    entries = readdir(manifest_dir; join=true)
    filter!(path -> endswith(path, ".json"), entries)
    if isempty(opts.datasets)
        return sort(entries)
    end
    wanted = Set(opts.datasets)
    selected = String[]
    for path in entries
        name = splitext(basename(path))[1]
        if name in wanted
            push!(selected, path)
        end
    end
    if length(selected) < length(wanted)
        known = Set(splitext(basename(path))[1] for path in entries)
        missing = setdiff(wanted, known)
        !isempty(missing) && @warn "Requested datasets not found" missing=collect(missing)
    end
    return sort(selected)
end

function main()
    opts = parse_args()
    isdir(MANIFEST_DIR) || error("Manifest directory not found: $(MANIFEST_DIR)")
    targets = datasets_to_process(MANIFEST_DIR, opts)
    isempty(targets) && return
    for path in targets
        manifest = JSON3.read(read(path, String), Dict{String,Any})
        stats = prune_manifest!(manifest; keep=opts.keep)
        name = splitext(basename(path))[1]
        if stats.removed_missing == 0 && stats.trimmed == 0
            @info "No changes" name
            continue
        end
        if opts.dry_run
            @info "(dry-run) Would prune manifest" name removed_missing=stats.removed_missing trimmed=stats.trimmed
        else
            open(path, "w") do io
                JSON3.write(io, manifest; indent=2)
            end
            @info "Pruned manifest" name removed_missing=stats.removed_missing trimmed=stats.trimmed path
        end
    end
end

main()
