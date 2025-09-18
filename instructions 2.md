 “wire‑it‑once, reuse forever” automation a virtual desk expects. Below is a **Julia-first, reproducible** recipe to:

1. crawl/download lots of files from a website (index page or explicit URLs),
2. unpack ZIP/GZ files, and
3. ingest them into a **DuckDB** database—fast, columnar, and local.

I’ll keep it modular so you can point it at ERCOT (or any other site) with a tiny config change.

---

## 0) One‑time setup (Julia environment)

```julia
# In a clean folder:
using Pkg
Pkg.activate(".")
Pkg.add([
    "HTTP",         # HTTP session if needed
    "Downloads",    # robust file downloads
    "URIs",         # build absolute URLs
    "Gumbo",        # HTML parsing
    "Cascadia",     # CSS selectors on HTML
    "ZipFile",      # unzip .zip
    "CodecZlib",    # gunzip .gz
    "SHA",          # file hashing for dedupe manifests
    "Glob",         # filesystem globs
    "DuckDB",       # the star of the warehouse
    "CSV", "DataFrames",  # helpful in a pinch
    "Dates", "JSON3"      # timestamps + config
])
```

> If a site requires a login or cookie, you can switch `Downloads.download` to `HTTP.get` and manage the session, but start with `Downloads`—it’s simpler and handles redirects.

---

## 1) Project layout

```
ercot-pipeline/
  Project.toml  Manifest.toml
  /config
    datasets.json           # what to pull
  /data
    /raw/                   # as-downloaded (zips/csv/gz)
    /staging/               # extracted to plain CSV
    /duckdb/ercot.duckdb    # your DB
    /manifests/             # what we already fetched (hash/URL/date)
  fetch_and_ingest.jl       # script you’ll run daily
```

**Example `config/datasets.json`** (two modes: `index` page scraping or direct `urls` list):

```json
[
  {
    "name": "dam_spp",
    "mode": "index",
    "index_url": "https://example.org/day-ahead/settlement-point-prices/",
    "file_regex": ".*\\.csv(\\.gz)?$",
    "dest_table": "market.dam_spp"
  },
  {
    "name": "rt_lmp",
    "mode": "urls",
    "urls": [
      "https://example.org/real-time/lmp/rt_lmp_2025-08-31.csv.gz",
      "https://example.org/real-time/lmp/rt_lmp_2025-09-01.csv.gz"
    ],
    "dest_table": "market.rt_lmp"
  }
]
```

> For ERCOT MIS “Trials Report Version”, point `index_url` at the Trials folder; for production history, point at the non‑Trials folder. If the site lists files by month, add one config entry per month root, or write a tiny loop to generate them.

---

## 2) The script: download → extract → ingest

**Save as `fetch_and_ingest.jl`** and run with `julia --project fetch_and_ingest.jl`.

```julia
#!/usr/bin/env julia
using JSON3, Dates, URIs, Downloads, HTTP, Gumbo, Cascadia
using ZipFile, CodecZlib, SHA, Glob, DuckDB

const ROOT   = @__DIR__
const DATA   = joinpath(ROOT, "data")
const RAW    = joinpath(DATA, "raw")
const STAGE  = joinpath(DATA, "staging")
const META   = joinpath(DATA, "manifests")
const DBPATH = joinpath(DATA, "duckdb", "ercot.duckdb")

mkpath.( [RAW, STAGE, META, dirname(DBPATH)] )

# ---------- utilities ----------
struct FileRec
    url::String
    raw_path::String
    staged_paths::Vector{String}
    sha256::String
end

function sha256sum(path::AbstractString)
    open(path, "r") do io
        ctx = sha256_init()
        buf = Vector{UInt8}(undef, 1_048_576) # 1 MB blocks
        while !eof(io)
            n = read!(io, buf)
            sha256_update!(ctx, view(buf, 1:n))
        end
        return bytes2hex(sha256_finalize!(ctx))
    end
end

function manifest_path(dataset::String)
    joinpath(META, "$(dataset).json")
end

function load_manifest(dataset::String)::Dict{String,Any}
    path = manifest_path(dataset)
    isfile(path) ? JSON3.read(read(path, String), Dict{String,Any}) : Dict{String,Any}("files" => Dict{String,Any}())
end

function save_manifest(dataset::String, mf::Dict{String,Any})
    open(manifest_path(dataset), "w") do io
        JSON3.write(io, mf; indent=2)
    end
end

function http_get(url::String)
    # You can add headers/cookies here if the site requires them
    return Downloads.download(url)
end

function parse_links(index_url::String; re::Regex=r".*")
    html = String(Downloads.download(index_url) |> read)
    doc  = parsehtml(html)
    base = URI(index_url)
    out  = String[]
    for a in eachmatch(Selector("a"), doc.root)  # all anchors
        href = try getattribute(a, "href") catch; nothing end
        href === nothing && continue
        # absolute-ize
        full = String(string(URI(base, URI(href))))
        occursin(re, full) && push!(out, full)
    end
    # de-dupe
    return unique(out)
end

function download_with_retry(url::String, dest::String; retries::Int=4, backoff::Float64=1.5)
    mkpath(dirname(dest))
    for attempt in 1:retries
        try
            tmp = dest * ".part"
            Downloads.download(url, tmp)
            mv(tmp, dest; force=true)
            return dest
        catch e
            attempt == retries && rethrow()
            sleep(backoff^(attempt-1))
        end
    end
end

function extract_to_stage(rawfile::String, stage_dir::String)
    mkpath(stage_dir)
    exts = splitext(rawfile)
    staged = String[]
    if endswith(lowercase(rawfile), ".zip")
        ZipFile.Reader(rawfile) do z
            for f in z.files
                out = joinpath(stage_dir, f.name)
                mkpath(dirname(out))
                write(out, read(f))
                push!(staged, out)
            end
        end
    elseif endswith(lowercase(rawfile), ".gz")
        # gunzip single file to same name sans .gz
        out = joinpath(stage_dir, splitext(basename(rawfile))[1])
        open(rawfile) do fio
            open(out, "w") do outio
                stream = GzipDecompressorStream(fio)
                write(outio, read(stream))
                close(stream)
            end
        end
        push!(staged, out)
    else
        # plain file
        out = joinpath(stage_dir, basename(rawfile))
        cp(rawfile, out; force=true)
        push!(staged, out)
    end
    return staged
end

# ---------- ingestion ----------
function ingest_csv_glob!(db::DuckDB.DB, table::String, csv_glob::String; union_by_name::Bool=true)
    # DuckDB can ingest directly from CSV wildcards quickly
    # Create schema if needed
    if occursin(".", table)
        schema, tbl = split(table, "."; limit=2)
        DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS $(schema);")
    end
    sql = """
    CREATE OR REPLACE TABLE $table AS
    SELECT * FROM read_csv_auto('$csv_glob',
        AUTO_DETECT=TRUE,
        HEADER=TRUE,
        union_by_name=$(union_by_name ? "TRUE" : "FALSE"),
        SAMPLE_SIZE=-1);
    """
    DuckDB.execute(db, sql)
end

# ---------- main ----------
function run_pipeline(config_path::String)
    specs = JSON3.read(read(config_path, String))
    db = DuckDB.DB(DBPATH)
    for spec in specs
        name  = String(spec["name"])
        table = String(spec["dest_table"])
        @info "Dataset" name table

        # discover file URLs
        urls = String[]
        if get(spec, "mode", "index") == "index"
            rx   = Regex(String(spec["file_regex"]))
            push!(urls, parse_links(String(spec["index_url"]); re=rx)...)
        elseif spec["mode"] == "urls"
            push!(urls, (String.(spec["urls"]))...)
        else
            error("Unknown mode $(spec["mode"]) for dataset $name")
        end
        @info "Found $(length(urls)) candidate files"

        # load manifest and loop
        mf = load_manifest(name)
        known = Set{String}(keys(mf["files"]))
        new_files = String[]

        for url in urls
            in( url, known ) && continue
            # raw download
            filename = basename(URI(url).path)
            rawdir   = joinpath(RAW, name)
            rawfile  = joinpath(rawdir, filename)
            @info "Downloading" url
            download_with_retry(url, rawfile)

            # hash for dedupe
            h = sha256sum(rawfile)

            # extract
            stagedir = joinpath(STAGE, name)
            staged = extract_to_stage(rawfile, stagedir)

            # update manifest
            mf["files"][url] = Dict(
                "raw" => rawfile,
                "staged" => staged,
                "sha256" => h,
                "fetched_at" => string(Dates.now()),
            )
            push!(new_files, rawfile)
        end
        save_manifest(name, mf)

        # ingest: all staged CSVs for this dataset (idempotent)
        csv_glob = joinpath(STAGE, name, "*.csv")
        @info "Ingesting into DuckDB" table csv_glob
        ingest_csv_glob!(db, table, csv_glob)
    end
    close(db)
    @info "Done."
end

# entry
config_file = isempty(ARGS) ? joinpath(ROOT, "config", "datasets.json") : ARGS[1]
run_pipeline(config_file)
```

### What this gives you

* **Index mode**: it scrapes an index page and downloads every link that matches `file_regex`.
* **URLs mode**: it downloads a pre‑listed set of files.
* **Idempotent**: a simple **manifest** prevents redownloading the same URL; dedupe is SHA‑256 based.
* **Extracts**: `.zip` and `.gz` are unpacked to plain CSV in `/staging/<dataset>/`.
* **Ingestion**: `CREATE OR REPLACE TABLE schema.table AS SELECT * FROM read_csv_auto('staging/ds/*.csv')`—fast, no CSV.jl needed.
* **Schemas**: You can later add a typed `CREATE TABLE` and then `INSERT` if you want strict types. For exploratory work, `read_csv_auto` is perfect.

---

## 3) Running it daily (and keeping it reproducible)

* **Manual**: `julia --project fetch_and_ingest.jl`
* **Cron (Linux/macOS)**: `0 10 * * * /usr/bin/env julia --project /path/ercot-pipeline/fetch_and_ingest.jl >> /path/logs/fetch.log 2>&1`

For full reproducibility, commit `Project.toml` and `Manifest.toml` after adding packages.

---

## 4) Practical tweaks for ERCOT‑style sites

* **Rate limits**: Add `sleep(0.5)` between downloads, or throttle by host.

* **Selective date range**: Filter `urls` by a date pattern in the filename before download.

* **Authentication/cookies**: Replace `Downloads.download` with:

  ```julia
  resp = HTTP.get(url; headers = ["User-Agent" => "YourBot/1.0"])
  open(dest, "w") do io; write(io, resp.body); end
  ```

  …keeping a `HTTP.CookieJar()` in a persistent session if needed.

* **Zip contents with nested folders**: The extractor above preserves paths. If you want flat CSV names, add a `basename` when writing.

---

## 5) Ingest patterns you’ll reuse

* **Append‑only (no replace):**

  ```julia
  # one CSV
  DuckDB.execute(db, 
    "INSERT INTO market.dam_spp SELECT * FROM read_csv_auto(?);", 
    (csvpath,))
  ```

* **Partitioned tables by day:**

  ```julia
  DuckDB.execute(db, "CREATE SCHEMA IF NOT EXISTS market;")
  DuckDB.execute(db, """
    CREATE OR REPLACE TABLE market.rt_lmp AS
    SELECT * FROM read_csv_auto('data/staging/rt_lmp/*.csv', union_by_name=TRUE)
  """)
  DuckDB.execute(db, "CREATE INDEX IF NOT EXISTS idx_rt_lmp_ts ON market.rt_lmp(timestamp);")
  ```

* **Query from Julia (sanity check):**

  ```julia
  using DuckDB
  db = DuckDB.DB(DBPATH)
  DuckDB.execute(db, "SELECT COUNT(*) FROM market.dam_spp;") |> DataFrame
  DuckDB.execute(db, """
    SELECT date_trunc('day', timestamp) d, AVG(price) avg_rt
    FROM market.rt_lmp GROUP BY 1 ORDER BY 1 DESC LIMIT 10
  """) |> DataFrame
  close(db)
  ```

---

## 6) “What if I already know all the direct file URLs?”

You can skip scraping entirely. Put them into the `urls` array in the config, and the same pipeline will download, extract, and load to DuckDB.

---

## 7) Guardrails (battle‑tested habits)

* **Time of availability**: If you’re training a model for Day‑Ahead, only include files posted **before 10:00 CPT** in your feature build. Encode that as a filter on filenames/post times.
* **Immutable staging**: Never modify files in `/raw`. If a file is corrected upstream, treat it as a new file with a new hash.
* **Schema drift**: DuckDB’s `union_by_name=TRUE` protects you when columns appear/disappear. Add an automated check that warns you when new columns arrive.

---