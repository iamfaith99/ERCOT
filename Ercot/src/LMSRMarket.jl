module MarketScoring

export LMSRMarket, initialize_market, price, state_prices, trade!, shares,
       price_after_trade, simulate_trades, liquidity_from_move, clone_market

struct LMSRMarket
    b::Float64
    events::Vector{Symbol}
    shares::Dict{Symbol,Float64}
end

σ(x) = 1 / (1 + exp(-x))
logit(p) = log(p / (1 - p))

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

clone_market(market::LMSRMarket) = LMSRMarket(market.b, copy(market.events), copy(market.shares))

function price_after_trade(market::LMSRMarket, id::Symbol, delta::Real)
    haskey(market.shares, id) || error("Unknown event $(id)")
    new_share = market.shares[id] + Float64(delta)
    return σ(new_share / market.b)
end

function simulate_trades(market::LMSRMarket, trades; copy_market_state::Bool = true)
    work_market = copy_market_state ? clone_market(market) : market
    results = NamedTuple[]
    cumulative_cost = 0.0

    for (idx, trade_desc) in enumerate(trades)
        event, qty = _parse_trade(trade_desc)
        price_before = price(work_market, event)
        cost = trade!(work_market, event, qty)
        price_after = price(work_market, event)
        cumulative_cost += cost
        push!(results, (step = idx,
                        event = event,
                        quantity = qty,
                        price_before = price_before,
                        price_after = price_after,
                        cost = cost,
                        cumulative_cost = cumulative_cost))
    end

    return results, work_market
end

function liquidity_from_move(prior_price::Real, new_price::Real, quantity::Real)
    p0 = _clamp_prob(prior_price)
    p1 = _clamp_prob(new_price)
    Δlogit = logit(p1) - logit(p0)
    q = Float64(quantity)
    abs(q) < 1e-12 && error("Quantity must be non-zero to infer liquidity parameter")
    abs(Δlogit) < 1e-12 && return Inf
    return q / Δlogit
end

function _parse_trade(trade_desc)
    if trade_desc isa Pair
        return (Symbol(first(trade_desc)), Float64(last(trade_desc)))
    elseif trade_desc isa NamedTuple
        haskey(trade_desc, :event) || error("Trade NamedTuple must contain :event")
        haskey(trade_desc, :quantity) || error("Trade NamedTuple must contain :quantity")
        return (Symbol(trade_desc.event), Float64(trade_desc.quantity))
    elseif trade_desc isa Tuple && length(trade_desc) == 2
        return (Symbol(trade_desc[1]), Float64(trade_desc[2]))
    else
        error("Unsupported trade descriptor $(trade_desc)")
    end
end

end
