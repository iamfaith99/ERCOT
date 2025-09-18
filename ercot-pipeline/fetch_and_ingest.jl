#!/usr/bin/env julia
using JSON3, Dates, URIs, Downloads, Gumbo, Cascadia
using ZipFile, CodecZlib, SHA, Glob, DuckDB, Logging

const ROOT   = @__DIR__
const DATA   = joinpath(ROOT, "data")
const RAW    = joinpath(DATA, "raw")
const STAGE  = joinpath(DATA, "staging")
const META   = joinpath(DATA, "manifests")
const DBPATH = joinpath(DATA, "duckdb", "ercot.duckdb")

struct Options
    config_path::String
    datasets::Vector{String}
    max_new::Union{Nothing,Int}
    skip_db::Bool
end

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
    if isempty(splitext(fname)[2])
        params = URIs.queryparams(uri)
        if haskey(params, "filename")
            return String(params["filename"])
        end
    end
    if fname == "download" && !isempty(uri.query)
        sanitized = replace(uri.query, r"[^A-Za-z0-9._-]" => "_")
        fname = fname * "_" * sanitized
    end
    return fname
end

function download_string(url::String)
    temp = Downloads.download(url)
    try
        return read(temp, String)
    finally
        isfile(temp) && rm(temp; force=true)
    end
end

function download_file(url::String, rawdir::String; max_attempts::Int=3)
    mkpath(rawdir)
    dest = joinpath(rawdir, filename_from_url(url))
    temp = dest * ".tmp"
    for attempt in 1:max_attempts
        try
            Downloads.download(url, temp)
            mv(temp, dest; force=true)
            return dest
        catch err
            isfile(temp) && rm(temp; force=true)
            @warn "Download failed" url attempt error=err
            if attempt == max_attempts
                rethrow(err)
            else
                sleep(1.0)
            end
        end
    end
    error("unreachable")
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

function sanitize_csv!(path::String)
    lower = lowercase(path)
    endswith(lower, ".csv") || return
    tmp = path * ".tmp"
    open(path, "r") do src
        open(tmp, "w") do dest
            for line in eachline(src)
                isempty(strip(line)) && continue
                write(dest, line)
                write(dest, '\n')
            end
        end
    end
    mv(tmp, path; force=true)
end

function extract_to_stage(raw_path::String, stagedir::String)
    mkpath(stagedir)
    staged = String[]
    lower = lowercase(raw_path)
    if endswith(lower, ".zip")
        archive = ZipFile.Reader(raw_path)
        try
            for file in archive.files
                name = file.name
                endswith(name, "/") && continue
                dest = joinpath(stagedir, basename(name))
                mkpath(dirname(dest))
                open(dest, "w") do io
                    write(io, read(file))
                end
                dest = normalize_csv_extension(dest)
                sanitize_csv!(dest)
                push!(staged, dest)
            end
        finally
            close(archive)
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
        sanitize_csv!(dest)
        push!(staged, dest)
    else
        dest = joinpath(stagedir, basename(raw_path))
        cp(raw_path, dest; force=true)
        dest = normalize_csv_extension(dest)
        sanitize_csv!(dest)
        push!(staged, dest)
    end
    return staged
end

function download_and_stage(dataset::String, url::String, rawdir::String, stagedir::String; max_attempts::Int=3)
    for attempt in 1:max_attempts
        rawfile = download_file(url, rawdir)
        hash = sha256sum(rawfile)
        try
            staged = extract_to_stage(rawfile, stagedir)
            return rawfile, staged, hash
        catch err
            @warn "Failed to extract archive, retrying" dataset url attempt error=err
            isfile(rawfile) && rm(rawfile; force=true)
            attempt == max_attempts && rethrow(err)
        end
    end
    error("unreachable")
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
    DuckDB.execute(
        db,
        "CREATE OR REPLACE TABLE $full_table AS SELECT * FROM read_csv_auto(?, union_by_name=TRUE, sample_size=-1);",
        (pattern,),
    )
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

function extract_js_string(html::String, name::String)
    pattern = Regex("var\\s+" * name * "\\s*=\\s*['\" ]([^'\";]+)")
    m = match(pattern, html)
    return m === nothing ? nothing : String(m.captures[1])
end

function documents_from_list(doc_list::Any)
    docs = Dict{String,Any}[]
    if doc_list isa Vector
        for item in doc_list
            if item isa Dict
                if haskey(item, "Document") && item["Document"] isa Dict
                    push!(docs, item["Document"])
                elseif haskey(item, "ConstructedName")
                    push!(docs, item)
                end
            end
        end
    elseif doc_list isa Dict
        if haskey(doc_list, "Document")
            doc = doc_list["Document"]
            if doc isa Vector
                for d in doc
                    d isa Dict && push!(docs, d)
                end
            elseif doc isa Dict
                push!(docs, doc)
            end
        else
            push!(docs, doc_list)
        end
    end
    return docs
end

function collect_misdoc_urls(html::String, pattern::Regex)
    report_type = extract_js_string(html, "reportTypeID")
    report_list = extract_js_string(html, "reportListUrl")
    download_base = extract_js_string(html, "reportDownloadUrl")
    if any(x -> x === nothing, (report_type, report_list, download_base))
        return String[]
    end

    json_url = string(report_list, report_type)
    response = try
        download_string(json_url)
    catch err
        @warn "Failed to fetch report list" json_url error=err
        return String[]
    end

    data = JSON3.read(response, Dict{String,Any})
    docs_key = get(data, "ListDocsByRptTypeRes", Dict{String,Any}())
    doc_list = get(docs_key, "DocumentList", Any[])
    docs = documents_from_list(doc_list)

    pattern_str = lowercase(string(pattern))
    wants_xml = occursin("xml", pattern_str)

    urls = String[]
    for doc in docs
        constructed = String(get(doc, "ConstructedName", get(doc, "FriendlyName", "")))
        occursin(pattern, constructed) || continue
        if !wants_xml && occursin("xml", lowercase(constructed))
            continue
        end
        docid = get(doc, "DocID", nothing)
        docid === nothing && continue
        base_url = string(download_base, docid)
        escaped_name = URIs.escapeuri(constructed)
        sep = occursin('?', base_url) ? '&' : '?'
        full_url = string(base_url, sep, "filename=", escaped_name)
        push!(urls, full_url)
    end

    return unique(urls)
end

function collect_index_urls(index_url::String, pattern::Regex)
    html = download_string(index_url)
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
    isempty(urls) || return unique(urls)
    return collect_misdoc_urls(html, pattern)
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

function parse_args()
    config_path = nothing
    datasets = String[]
    max_new = nothing
    skip_db = false
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--config"
            i += 1
            i > length(ARGS) && error("--config requires a path")
            config_path = ARGS[i]
        elseif startswith(arg, "--config=")
            config_path = String(split(arg, "=", limit=2)[2])
        elseif arg == "--max-new"
            i += 1
            i > length(ARGS) && error("--max-new requires an integer value")
            max_new = parse(Int, ARGS[i])
        elseif startswith(arg, "--max-new=")
            max_new = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg in ("--skip-db", "--download-only")
            skip_db = true
        elseif startswith(arg, "--")
            error("Unknown flag $(arg)")
        else
            push!(datasets, String(arg))
        end
        i += 1
    end
    config_path = something(config_path, joinpath(ROOT, "config", "datasets.json"))
    return Options(config_path, datasets, max_new, skip_db)
end

function should_process_dataset(name::String, opts::Options)
    isempty(opts.datasets) && return true
    return any(d -> d == name, opts.datasets)
end

function run_pipeline(opts::Options)
    config = JSON3.read(read(opts.config_path, String))
    db = nothing
    if !opts.skip_db
        try
            db = DuckDB.DB(DBPATH)
        catch err
            @warn "Failed to open DuckDB; continuing without ingestion" error=err
            opts = Options(opts.config_path, opts.datasets, opts.max_new, true)
        end
    end
    try
        for entry in config
            name = String(entry["name"])
            should_process_dataset(name, opts) || continue
            dest_table = String(entry["dest_table"])
            rawdir = joinpath(RAW, name)
            stagedir = joinpath(STAGE, name)
            mkpath(rawdir); mkpath(stagedir)

            urls = dataset_urls(entry)
            manifest = load_manifest(name)
            files = manifest["files"]::Dict{String,Any}

            cached = String[]
            new_urls = String[]
            for url in urls
                if haskey(files, url)
                    push!(cached, url)
                else
                    push!(new_urls, url)
                end
            end

            if opts.max_new !== nothing && length(new_urls) > opts.max_new
                skipped = length(new_urls) - opts.max_new
                @info "Limiting new downloads" name requested=length(new_urls) limit=opts.max_new skipped=skipped
                new_urls = new_urls[1:opts.max_new]
            end

            for url in vcat(cached, new_urls)
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
                rawfile, staged, hash = download_and_stage(name, url, rawdir, stagedir)
                files[url] = Dict(
                    "raw" => rawfile,
                    "staged" => staged,
                    "sha256" => hash,
                    "fetched_at" => string(Dates.now()),
                )
            end

            save_manifest(name, manifest)
            if db === nothing
                @info "Skipping DuckDB ingest (download-only mode)" name stagedir
            else
                ingest_csvs!(db, dest_table, stagedir)
            end
        end
    finally
        db === nothing || close(db)
    end
    @info "Done."
end

opts = parse_args()
run_pipeline(opts)
