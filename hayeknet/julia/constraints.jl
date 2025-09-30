module HayekNetConstraints

using LightGraphs
using PythonCall

export validate_dag_constraints

"""
    validate_dag_constraints(edge_list, source, sink, capacity, bid)

Return `true` when the directed acyclic graph described by `edge_list` admits a
path from `source` to `sink` and the proposed `bid` respects the supplied
`capacity` limit.
"""
function validate_dag_constraints(
    edge_list,  # Accept any iterable of pairs
    source::Integer,
    sink::Integer,
    capacity::Real,
    bid::Real,
)::Bool
    bid >= 0 || return false
    capacity >= 0 || return false
    bid <= capacity || return false

    node_count = maximum(maximum, edge_list; init = max(source, sink))
    g = SimpleDiGraph(node_count)

    for edge in edge_list
        u, v = edge[1], edge[2]  # Handle both tuples and arrays
        add_edge!(g, Int(u), Int(v))
    end

    return has_path(g, Int(source), Int(sink)) && !is_cyclic(g)
end

# Export for Python (handled by juliacall automatically)

end # module
