module ERCOTPipeline

include("Device.jl")
include("EventAlgebra.jl")
include("AssimilationModel.jl")
include("EnKF.jl")
include("LMSRMarket.jl")
include("AssimilationRunner.jl")
include("PTDFUtils.jl")
include("RLTradingEnv.jl")

using .Device: AbstractExecutionDevice, CPUDevice, GPUDevice, detect_device
using .AssimilationModel: RTCStateModel, build_rtc_state_model, simulate_ensemble, default_state_labels,
                           ensemble_event_priors, evaluate_event_priors
using .EnKF: EnKFFilter, build_enkf, update_ensemble!
using .EventAlgebra: EventNode, EventGraph, add_event!, upsert_event!, has_event, get_event,
                     parents, children, ancestors, topological_order
using .MarketScoring: LMSRMarket, initialize_market, price, state_prices, trade!, shares,
                      price_after_trade, simulate_trades, liquidity_from_move, clone_market,
                      value_claim, brier_score, log_score, calibration_metrics
using .AssimilationRunner: analyze_and_forecast!
using .PTDFUtils: EXTRA_REGRESSORS, EXTRA_REGRESSORS_STR, load_latest_snapshot, load_metadata,
                  model_is_fresh, model_improvement, build_feature_vector, predict_congestion,
                  build_event_graph, upsert_assimilation_events!, scenario_summary,
                  persist_event_prices!, persist_risk_log!, ensure_training_tables!, publish_lag_snapshot!,
                  calibrate_scenario_cone, persist_scenario_calibration!,
                  latest_scenario_calibration, what_if

export AbstractExecutionDevice, CPUDevice, GPUDevice,
       detect_device,
       RTCStateModel, build_rtc_state_model, simulate_ensemble, ensemble_event_priors, evaluate_event_priors,
       EnKFFilter, build_enkf, update_ensemble!,
       default_state_labels,
       EventNode, EventGraph, add_event!, upsert_event!, has_event, get_event,
       parents, children, ancestors, topological_order,
       LMSRMarket, initialize_market, price, state_prices, trade!, shares,
       price_after_trade, simulate_trades, liquidity_from_move, clone_market,
       value_claim, brier_score, log_score, calibration_metrics,
       analyze_and_forecast!,
       EXTRA_REGRESSORS, EXTRA_REGRESSORS_STR, load_latest_snapshot, load_metadata,
       model_is_fresh, model_improvement, build_feature_vector, predict_congestion,
       build_event_graph, upsert_assimilation_events!, scenario_summary,
       persist_event_prices!, persist_risk_log!, ensure_training_tables!, publish_lag_snapshot!,
       calibrate_scenario_cone, persist_scenario_calibration!,
       latest_scenario_calibration, what_if,
       RLTradingEnv, TradingEnv, reset!, step!, state, is_done, log_run!, summary

end
