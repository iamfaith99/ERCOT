#!/usr/bin/env julia
using JSON3, Dates, URIs, Downloads, Gumbo, Cascadia
using ZipFile, CodecZlib, SHA, Glob, DuckDB, Logging

const ROOT   = @__DIR__
const DATA   = joinpath(ROOT, "data")
const RAW    = joinpath(DATA, "raw")
const STAGE  = joinpath(DATA, "staging")
const META   = joinpath(DATA, "manifests")
const DBPATH = joinpath(DATA, "duckdb", "ercot.duckdb")

mkpath.( [RAW, STAGE, META, dirname(DBPATH)] )

# ---------- utilities ----------

function sha256sum(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

manifest_path(dataset::String) = joinpath(META, "$(dataset).json")

function load_manifest(dataset::String)
    path = manifest_path(dataset)
    if isfile(path)
        manifest = JSON3.read(read(path, String), Dict{String,Any})
        get!(manifest, "files") do
            Dict{String,Any}()
        end
        return manifest
    else
        return Dict{String,Any}("files" => Dict{String,Any}())
    end
end

function save_manifest(dataset::String, manifest::Dict{String,Any})
    path = manifest_path(dataset)
    open(path, "w") do io
        JSON3.write(io, manifest; indent=2)
    end
end

function filename_from_url(url::String)
    uri = URI(url)
    path = uri.path
    fname = isempty(path) ? "download" : split(path, "/")[end]
    if isempty(fname)
        fname = "download"
    end
    if fname == "download" && !isempty(uri.query)
        sanitized = replace(uri.query, r"[^A-Za-z0-9._-]" => "_")
        fname = fname * "_" * sanitized
    end
    return fname
end

function unique_path(path::String)
    if !isfile(path)
        return path
    end
    base, ext = splitext(path)
    counter = 1
    while true
        candidate = string(base, "_", counter, ext)
        isfile(candidate) || return candidate
        counter += 1
    end
end

function download_file(url::String, rawdir::String)
    mkpath(rawdir)
    dest = unique_path(joinpath(rawdir, filename_from_url(url)))
    Downloads.download(url, dest)
    return dest
end

function normalize_csv_extension(path::String)
    root, ext = splitext(path)
    lower = lowercase(ext)
    if lower == ".csv" && ext != ".csv"
        newpath = root * ".csv"
        mv(path, newpath; force=true)
        return newpath
    end
    return path
end

function extract_to_stage(raw_path::String, stagedir::String)
    mkpath(stagedir)
    staged = String[]
    lower = lowercase(raw_path)
    if endswith(lower, ".zip")
        ZipFile.Reader(raw_path) do archive
            for file in archive.files
                name = file.name
                endswith(name, "/") && continue
                dest = joinpath(stagedir, basename(name))
                mkpath(dirname(dest))
                open(dest, "w") do io
                    write(io, read(file))
                end
                dest = normalize_csv_extension(dest)
                push!(staged, dest)
            end
        end
    elseif endswith(lower, ".gz") && !endswith(lower, ".tar.gz")
        dest = joinpath(stagedir, replace(basename(raw_path), r"\.gz$" => ""))
        open(raw_path, "r") do src
            stream = GzipDecompressorStream(src)
            open(dest, "w") do io
                write(io, read(stream))
            end
        end
        dest = normalize_csv_extension(dest)
        push!(staged, dest)
    else
        dest = joinpath(stagedir, basename(raw_path))
        cp(raw_path, dest; force=true)
        dest = normalize_csv_extension(dest)
        push!(staged, dest)
    end
    return staged
end

function staged_csvs(stagedir::String)
    csvs = Glob.glob("*.csv", stagedir)
    if isempty(csvs)
        for path in Glob.glob("*.CSV", stagedir)
            normalize_csv_extension(path)
        end
        csvs = Glob.glob("*.csv", stagedir)
    end
    sort!(csvs)
    return csvs
end

function quote_ident(name::AbstractString)
    return "\"" * replace(name, "\"" => "\"\"") * "\""
end

function split_table(dest_table::String)
    if occursin('.', dest_table)
        parts = split(dest_table, '.'; limit=2)
        return (String(parts[1]), String(parts[2]))
    else
        return (nothing, dest_table)
    end
end

function ensure_schema!(db::DuckDB.DB, schema::Union{Nothing,String})
    isnothing(schema) && return
    DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS $(quote_ident(schema));")
end

function ingest_csvs!(db::DuckDB.DB, table::String, stagedir::String)
    csvs = staged_csvs(stagedir)
    if isempty(csvs)
        @info "No staged CSVs found" stagedir
        return
    end
    schema, name = split_table(table)
    ensure_schema!(db, schema)
    pattern = joinpath(stagedir, "*.csv")
    full_table = isnothing(schema) ? quote_ident(name) : string(quote_ident(schema), ".", quote_ident(name))
    DuckDB.execute(db, "CREATE OR REPLACE TABLE $full_table AS SELECT * FROM read_csv_auto(?, union_by_name=TRUE);", (pattern,))
end

function absolute_href(base_url::String, href::String)
    isempty(href) && return nothing
    startswith(href, "#") && return nothing
    uri = try
        URI(href)
    catch err
        @warn "Skipping invalid href" href error=err
        return nothing
    end
    if isempty(uri.scheme)
        base_uri = URI(base_url)
        uri = URIs.resolvereference(base_uri, uri)
    end
    scheme = lowercase(uri.scheme)
    scheme in ("http", "https") || return nothing
    return string(uri)
end

function collect_index_urls(index_url::String, pattern::Regex)
    temp = Downloads.download(index_url)
    html = read(temp, String)
    rm(temp; force=true)
    doc = parsehtml(html)
    urls = String[]
    for node in eachmatch(sel"a", doc.root)
        href = get(node.attributes, "href", nothing)
        href === nothing && continue
        url = absolute_href(index_url, href)
        url === nothing && continue
        occursin(pattern, url) || continue
        push!(urls, url)
    end
    return unique(urls)
end

function dataset_urls(entry)
    mode = String(get(entry, "mode", "index"))
    pattern = haskey(entry, "file_regex") ? Regex(String(entry["file_regex"])) : r".*"
    if mode == "index"
        index_url = String(entry["index_url"])
        return collect_index_urls(index_url, pattern)
    elseif mode == "urls"
        return String.(collect(entry["urls"]))
    else
        error("Unsupported mode $(mode)")
    end
end

function run_pipeline(config_path::String)
    config = JSON3.read(read(config_path, String))
    db = DuckDB.DB(DBPATH)
    try
        for entry in config
            name = String(entry["name"])
            dest_table = String(entry["dest_table"])
            rawdir = joinpath(RAW, name)
            stagedir = joinpath(STAGE, name)
            mkpath(rawdir); mkpath(stagedir)

            urls = dataset_urls(entry)
            manifest = load_manifest(name)
            files = manifest["files"]::Dict{String,Any}

            for url in urls
                if haskey(files, url)
                    info = files[url]
                    raw_exists = haskey(info, "raw") && isfile(String(info["raw"]))
                    if raw_exists
                        @info "Skipping already downloaded" name url
                        continue
                    else
                        @info "Manifest entry missing file, re-downloading" name url
                        delete!(files, url)
                    end
                end

                @info "Downloading" name url
                rawfile = download_file(url, rawdir)
                hash = sha256sum(rawfile)
                staged = extract_to_stage(rawfile, stagedir)
                files[url] = Dict(
                    "raw" => rawfile,
                    "staged" => staged,
                    "sha256" => hash,
                    "fetched_at" => string(Dates.now()),
                )
            end

            save_manifest(name, manifest)
            ingest_csvs!(db, dest_table, stagedir)
        end
    finally
        close(db)
    end
    @info "Done."
end

config_file = isempty(ARGS) ? joinpath(ROOT, "config", "datasets.json") : ARGS[1]
run_pipeline(config_file)
