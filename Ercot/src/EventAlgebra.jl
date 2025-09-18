module EventAlgebra

using Base: isempty
using Dates: DateTime

export EventNode, EventGraph,
       add_event!, upsert_event!, has_event, get_event,
       parents, children, ancestors,
       topological_order

const _VALID_SCOPE_KEYS = Set([:interval, :variable, :relation, :threshold,
                               :product, :location, :tags, :prior,
                               :aggregator, :weights, :target, :given,
                               :numerator, :denominator, :joint])
const _VALID_RELATIONS = Set([:gt, :ge, :lt, :le, :eq, :ne])

function _normalize_scope(scope::NamedTuple)
    isempty(scope) && return scope
    buf = Dict{Symbol,Any}()
    for (k, v) in pairs(scope)
        k in _VALID_SCOPE_KEYS || error("Unsupported scope key $(k)")
        buf[k] = _normalize_scope_value(k, v)
    end

    if haskey(buf, :relation)
        rel = buf[:relation]
        rel in _VALID_RELATIONS || error("Unsupported relation $(rel)")
        haskey(buf, :variable) || error(":relation requires :variable in scope")
        haskey(buf, :threshold) || error(":relation requires :threshold")
    end

    if haskey(buf, :threshold) && !haskey(buf, :relation)
        error(":threshold provided without :relation")
    end

    if haskey(buf, :interval)
        interval = buf[:interval]
        interval isa Tuple{DateTime,DateTime} || error(":interval must be Tuple{DateTime,DateTime}")
        interval[1] <= interval[2] || error(":interval start must be <= finish")
    end

    if haskey(buf, :prior)
        p = Float64(buf[:prior])
        (0.0 <= p <= 1.0) || error(":prior must lie in [0,1]")
        buf[:prior] = p
    end

    if haskey(buf, :threshold)
        buf[:threshold] = Float64(buf[:threshold])
    end

    return (; buf...)
end

function _normalize_scope_value(key::Symbol, value)
    if key === :variable || key === :product || key === :location
        return Symbol(value)
    elseif key === :tags
        arr = value isa AbstractVector ? collect(value) : value isa Tuple ? collect(value) : [value]
        return Symbol.(arr)
    elseif key === :relation
        return Symbol(value)
    elseif key === :aggregator
        return value isa Function ? value : Symbol(value)
    elseif key === :target || key === :given || key === :numerator ||
           key === :denominator || key === :joint
        return Symbol(value)
    elseif key === :weights
        if value isa NamedTuple || value isa Dict
            return value
        elseif value isa AbstractVector || value isa Tuple
            return collect(value)
        else
            error("Unsupported weights container $(typeof(value))")
        end
    else
        return value
    end
end

"""
    EventNode(id; parents=[], description="", scope=NamedTuple())

Define a Boolean market event identified by `id`. `parents` lists upstream
events in the Bayesian DAG, while `scope` can hold metadata such as interval,
state variable, thresholds, or product codes.

Common scope keys:
- `:variable` - state label (e.g. `:load`)
- `:relation` - comparison (`:gt`, `:ge`, `:lt`, `:le`, `:eq`, `:ne`)
- `:threshold` - numeric threshold paired with `:relation`
- `:interval` - `(DateTime, DateTime)` bucket
- `:product` / `:location` - symbolic identifiers
- `:tags` - additional symbolic markers
- `:prior` - fixed probability when not derived from a state variable
- `:aggregator` - combine parent probabilities via a builtin symbol or custom function
- `:weights` - optional weights for aggregators like `:weighted_sum`
- `:target`, `:given`, `:numerator`, `:denominator`, `:joint` - helper keys for conditional and ratio-style aggregators
"""

struct EventNode
    id::Symbol
    parents::Vector{Symbol}
    description::String
    scope::NamedTuple
end

EventNode(id::Symbol; parents::Vector{Symbol}=Symbol[], description::String="", scope::NamedTuple=NamedTuple()) =
    EventNode(id, parents, description, _normalize_scope(scope))

"""
    EventGraph()

Mutable DAG that tracks `EventNode`s and their conditional structure. Nodes
must be inserted in ancestral order; `topological_order` raises if a cycle is
introduced.

# Example

```
g = EventGraph()
add_event!(g, EventNode(:load))
add_event!(g, EventNode(:binds_congestion; parents=[:load]))
@assert topological_order(g) == [:load, :binds_congestion]
```
"""

mutable struct EventGraph
    nodes::Dict{Symbol,EventNode}
    children_map::Dict{Symbol,Vector{Symbol}}
    dirty::Bool
    order_cache::Vector{Symbol}
end

EventGraph() = EventGraph(Dict{Symbol,EventNode}(), Dict{Symbol,Vector{Symbol}}(), true, Symbol[])

function has_event(g::EventGraph, id::Symbol)
    return haskey(g.nodes, id)
end

function get_event(g::EventGraph, id::Symbol)
    has_event(g, id) || error("Unknown event $(id)")
    return g.nodes[id]
end

function _register_children!(g::EventGraph, node::EventNode)
    get!(g.children_map, node.id, Symbol[])
    for parent in node.parents
        haskey(g.nodes, parent) || error("Parent $(parent) must be registered before $(node.id)")
        push!(get!(g.children_map, parent, Symbol[]), node.id)
    end
end

function add_event!(g::EventGraph, node::EventNode)
    has_event(g, node.id) && error("Event $(node.id) already registered")
    _register_children!(g, node)
    g.nodes[node.id] = node
    g.dirty = true
    return g
end

function upsert_event!(g::EventGraph, node::EventNode)
    if has_event(g, node.id)
        g.nodes[node.id] = node
        for kids in values(g.children_map)
            filter!(!=(node.id), kids)
        end
        g.children_map[node.id] = Symbol[]
    end
    _register_children!(g, node)
    g.nodes[node.id] = node
    g.dirty = true
    return g
end

function parents(g::EventGraph, id::Symbol)
    has_event(g, id) || error("Unknown event $(id)")
    return g.nodes[id].parents
end

function children(g::EventGraph, id::Symbol)
    haskey(g.children_map, id) || return Symbol[]
    return g.children_map[id]
end

function ancestors(g::EventGraph, id::Symbol)
    has_event(g, id) || error("Unknown event $(id)")
    visited = Set{Symbol}()
    stack = copy(g.nodes[id].parents)
    while !isempty(stack)
        current = pop!(stack)
        current in visited && continue
        push!(visited, current)
        append!(stack, g.nodes[current].parents)
    end
    return collect(visited)
end

function _compute_topological_order(g::EventGraph)
    indegree = Dict{Symbol,Int}()
    for (id, node) in g.nodes
        indegree[id] = get(indegree, id, 0)
        for child in children(g, id)
            indegree[child] = get(indegree, child, 0) + 1
        end
    end

    queue = Symbol[]
    for (id, deg) in indegree
        deg == 0 && push!(queue, id)
    end

    order = Symbol[]
    while !isempty(queue)
        v = popfirst!(queue)
        push!(order, v)
        for child in children(g, v)
            indegree[child] -= 1
            indegree[child] == 0 && push!(queue, child)
        end
    end

    length(order) == length(g.nodes) || error("Cycle detected in event graph")
    return order
end

function topological_order(g::EventGraph)
    if g.dirty
        g.order_cache = _compute_topological_order(g)
        g.dirty = false
    end
    return copy(g.order_cache)
end

end
