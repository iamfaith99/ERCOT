#!/usr/bin/env julia
import DuckDB

const DBPATH = joinpath(@__DIR__, "..", "data", "duckdb", "ercot.duckdb")
const VIEW_SQL = joinpath(@__DIR__, "..", "sql", "create_views.sql")

function main()
    isfile(VIEW_SQL) || return
    source = read(VIEW_SQL, String)
    db = DuckDB.DB(DBPATH)
    try
        for raw_stmt in split(source, ';')
            stmt = strip(raw_stmt)
            isempty(stmt) && continue
            DuckDB.execute(db, stmt)
        end
    finally
        close(db)
    end
end

main()
