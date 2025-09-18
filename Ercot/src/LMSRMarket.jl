module MarketScoring

export LMSRMarket, initialize_market, price, state_prices, trade!, shares

struct LMSRMarket
    b::Float64
    events::Vector{Symbol}
    shares::Dict{Symbol,Float64}
end

σ(x) = 1 / (1 + exp(-x))

function _clamp_prob(p::Float64)
    eps_val = eps(Float64)
    p < eps_val && return eps_val
    p > 1 - eps_val && return 1 - eps_val
    return p
end

_clamp_prob(p) = _clamp_prob(Float64(p))

function _log1pexp(x::Float64)
    if x > 0
        return x + log1p(exp(-x))
    else
        return log1p(exp(x))
    end
end

function initialize_market(priors; b::Real = 1.0)
    b_val = Float64(b)
    b_val > 0 || error("b must be positive")
    events = Symbol[]
    shares = Dict{Symbol,Float64}()
    for (raw_id, p_raw) in pairs(priors)
        id = Symbol(raw_id)
        p = _clamp_prob(Float64(p_raw))
        shares[id] = b_val * log(p / (1 - p))
        push!(events, id)
    end
    return LMSRMarket(b_val, events, shares)
end

function price(market::LMSRMarket, id::Symbol)
    haskey(market.shares, id) || error("Unknown event $(id)")
    scaled = market.shares[id] / market.b
    return σ(scaled)
end

function state_prices(market::LMSRMarket)
    return Dict(id => price(market, id) for id in market.events)
end

function trade!(market::LMSRMarket, id::Symbol, delta::Real)
    haskey(market.shares, id) || error("Unknown event $(id)")
    δ = Float64(delta)
    old_share = market.shares[id]
    old_cost = market.b * _log1pexp(old_share / market.b)
    new_share = old_share + δ
    market.shares[id] = new_share
    new_cost = market.b * _log1pexp(new_share / market.b)
    return new_cost - old_cost
end

shares(market::LMSRMarket) = copy(market.shares)

end
