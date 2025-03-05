# parameter_optimization.py
# Functions for parameter optimization and cross-validation

import os
import time
import numpy as np
import pandas as pd
import random
import joblib
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

#from purged_walk_forward import optimize_parameters_with_optuna  # Add this if not already there
from volatility import calculate_volatility
from regime_detection import detect_volatility_regimes
from signals import calculate_trend_strength, calculate_momentum, filter_signals, apply_min_holding_period
from risk_management import calculate_adaptive_position_size, apply_unified_risk_management, calculate_trade_returns
from performance_metrics import calculate_advanced_metrics

def ensure_parameters(config):
    """
    Ensure that the configuration has valid parameter lists.
    This prevents errors in parameter grid generation.
    
    Parameters:
        config (dict): Strategy configuration
        
    Returns:
        dict: Modified configuration with valid parameter lists
    """
    modified_config = config.copy()
    
    # Function to ensure a value is a list
    def ensure_list(value):
        if not isinstance(value, list):
            return [value]
        return value if len(value) > 0 else [value]
    
    # Ensure volatility parameters are lists
    if 'volatility' in modified_config:
        vol_config = modified_config['volatility']
        if 'methods' in vol_config:
            vol_config['methods'] = ensure_list(vol_config['methods'])
        else:
            vol_config['methods'] = ['parkinson']
            
        if 'lookback_periods' in vol_config:
            vol_config['lookback_periods'] = ensure_list(vol_config['lookback_periods'])
        else:
            vol_config['lookback_periods'] = [20]
            
        if 'regime_smoothing' in vol_config:
            vol_config['regime_smoothing'] = ensure_list(vol_config['regime_smoothing'])
        else:
            vol_config['regime_smoothing'] = [5]
    
    # Ensure regime parameters are lists
    if 'regime_detection' in modified_config:
        regime_config = modified_config['regime_detection']
        if 'method' in regime_config:
            # Convert 'method' to 'methods' list
            method_val = regime_config.pop('method', 'kmeans')
            regime_config['methods'] = ensure_list(method_val)
        elif 'methods' in regime_config:
            regime_config['methods'] = ensure_list(regime_config['methods'])
        else:
            regime_config['methods'] = ['kmeans']
            
        if 'n_regimes' in regime_config:
            regime_config['n_regimes'] = ensure_list(regime_config['n_regimes'])
        else:
            regime_config['n_regimes'] = [3]
            
        if 'regime_stability_period' in regime_config:
            regime_config['regime_stability_period'] = ensure_list(regime_config['regime_stability_period'])
        else:
            regime_config['regime_stability_period'] = [48]
    
    # Ensure SMA parameters are lists
    if 'sma' in modified_config:
        sma_config = modified_config['sma']
        if 'short_windows' in sma_config:
            sma_config['short_windows'] = ensure_list(sma_config['short_windows'])
        else:
            sma_config['short_windows'] = [5, 8, 13, 21, 34]
            
        if 'long_windows' in sma_config:
            sma_config['long_windows'] = ensure_list(sma_config['long_windows'])
        else:
            sma_config['long_windows'] = [21, 34, 55, 89, 144]
            
        if 'min_holding_period' in sma_config:
            sma_config['min_holding_period'] = ensure_list(sma_config['min_holding_period'])
        else:
            sma_config['min_holding_period'] = [24]
            
        if 'trend_filter_period' in sma_config:
            sma_config['trend_filter_period'] = ensure_list(sma_config['trend_filter_period'])
        else:
            sma_config['trend_filter_period'] = [200]
            
        if 'trend_strength_threshold' in sma_config:
            sma_config['trend_strength_threshold'] = ensure_list(sma_config['trend_strength_threshold'])
        else:
            sma_config['trend_strength_threshold'] = [0.3]
    
    # Ensure risk parameters are lists
    if 'risk_management' in modified_config:
        risk_config = modified_config['risk_management']
        for param in ['target_volatility', 'max_position_size', 'min_position_size', 
                      'max_drawdown_exit', 'profit_taking_threshold', 
                      'trailing_stop_activation', 'trailing_stop_distance']:
            if param in risk_config:
                risk_config[param] = ensure_list(risk_config[param])
    
    return modified_config

def calculate_fitness_score(metrics, config):
    """
    Calculate fitness score for parameter optimization.
    
    Parameters:
        metrics (dict): Performance metrics
        config (dict): Configuration with weights
        
    Returns:
        float: Fitness score
    """
    # Extract weights
    sharpe_weight = config['parameter_selection']['sharpe_weight']
    sortino_weight = config['parameter_selection']['sortino_weight']
    calmar_weight = config['parameter_selection']['calmar_weight']
    return_weight = config.get('parameter_selection', {}).get('return_weight', 0.3)  # Default if not present
    
    # Normalize weights
    total_weight = sharpe_weight + sortino_weight + calmar_weight + return_weight
    sharpe_weight /= total_weight
    sortino_weight /= total_weight
    calmar_weight /= total_weight
    return_weight /= total_weight
    
    # Calculate weighted score
    fitness_score = (
        sharpe_weight * metrics['sharpe_ratio'] +
        sortino_weight * metrics['sortino_ratio'] +
        calmar_weight * metrics['calmar_ratio'] +
        return_weight * metrics['total_return'] * 5  # Multiply by a factor to normalize scale
    )
    
    # Apply penalty for excessive drawdown
    if metrics['max_drawdown'] < -0.3:
        fitness_score *= (1 + metrics['max_drawdown'])
    
    return fitness_score

def create_param_key(param_set):
    """
    Create a unique string key for a parameter set.
    
    Parameters:
        param_set (dict): Dictionary of parameters
        
    Returns:
        str: Unique key representing the parameter set
    """
    key_parts = []
    for k, v in sorted(param_set.items()):
        key_parts.append(f"{k}_{v}")
    return "_".join(key_parts)

def are_params_equal(params1, params2):
    """
    Check if two parameter sets have the same values for all keys in params2.
    
    Parameters:
        params1 (dict): First parameter set
        params2 (dict): Second parameter set
        
    Returns:
        bool: True if parameter sets are equal, False otherwise
    """
    for key, value in params2.items():
        if key not in params1 or params1[key] != value:
            return False
    return True

def generate_parameter_grid(config, cv_config):
    """
    Generate parameter grid based on configuration and optimization settings.
    
    Parameters:
        config (dict): Strategy configuration
        cv_config (dict): Cross-validation parameter testing configuration
        
    Returns:
        list: Parameter grid
    """
    import random

    # Initialize parameter grid
    param_grid = []

    # Create a default parameter set that will always be included
    default_params = {
        'vol_method': 'parkinson',
        'vol_lookback': 20,
        'short_window': 13, 
        'long_window': 55,
        'min_holding_period': 24,
        'trend_strength_threshold': 0.3,
        'regime_method': 'kmeans',
        'n_regimes': 3,
        'regime_stability': 48,
        'regime_smoothing': 5,
        'target_vol': 0.15,
        'max_position_size': 1.0,
        'min_position_size': 0.1,
        'max_drawdown_exit': 0.15,
        'profit_taking_threshold': 0.05,
        'trailing_stop_activation': 0.03,
        'trailing_stop_distance': 0.02
    }
    
    # Check if we're in advanced mode
    advanced_mode = cv_config.get('advanced_mode', False)
    max_combinations = cv_config.get('max_combinations', 500)
    
    # Check which parameter groups to optimize
    optimize_sma = cv_config.get('optimize_sma_params', True)
    optimize_regime = cv_config.get('optimize_regime_params', False)
    optimize_risk = cv_config.get('optimize_risk_params', False)
    
    # Always include the default parameter set for stability
    param_grid.append(default_params.copy())
    print(f"Added default parameter set")
    
    # Extract parameter lists safely - volatility methods
    vol_methods = ['parkinson']
    if 'volatility' in config and 'methods' in config['volatility']:
        if isinstance(config['volatility']['methods'], list) and config['volatility']['methods']:
            vol_methods = config['volatility']['methods']
        elif config['volatility']['methods']:
            vol_methods = [config['volatility']['methods']]
    
    # Extract lookback periods
    vol_lookbacks = [20]
    if 'volatility' in config and 'lookback_periods' in config['volatility']:
        if isinstance(config['volatility']['lookback_periods'], list) and config['volatility']['lookback_periods']:
            vol_lookbacks = config['volatility']['lookback_periods']
        elif config['volatility']['lookback_periods']:
            vol_lookbacks = [config['volatility']['lookback_periods']]
    
    # Extract regime smoothing periods
    regime_smoothings = [5]
    if 'volatility' in config and 'regime_smoothing' in config['volatility']:
        if isinstance(config['volatility']['regime_smoothing'], list) and config['volatility']['regime_smoothing']:
            regime_smoothings = config['volatility']['regime_smoothing']
        elif config['volatility']['regime_smoothing']:
            regime_smoothings = [config['volatility']['regime_smoothing']]
    
    # Extract short windows
    short_windows = [5, 8, 13]
    if 'sma' in config and 'short_windows' in config['sma']:
        if isinstance(config['sma']['short_windows'], list) and config['sma']['short_windows']:
            short_windows = config['sma']['short_windows']
        elif config['sma']['short_windows']:
            short_windows = [config['sma']['short_windows']]
    
    # Extract long windows
    long_windows = [21, 34, 55]
    if 'sma' in config and 'long_windows' in config['sma']:
        if isinstance(config['sma']['long_windows'], list) and config['sma']['long_windows']:
            long_windows = config['sma']['long_windows']
        elif config['sma']['long_windows']:
            long_windows = [config['sma']['long_windows']]
    
    # Extract min holding periods
    min_holding_periods = [24]
    if 'sma' in config and 'min_holding_period' in config['sma']:
        if isinstance(config['sma']['min_holding_period'], list) and config['sma']['min_holding_period']:
            min_holding_periods = config['sma']['min_holding_period']
        elif config['sma']['min_holding_period']:
            min_holding_periods = [config['sma']['min_holding_period']]
    
    # Extract trend strength thresholds
    trend_thresholds = [0.3]
    if 'sma' in config and 'trend_strength_threshold' in config['sma']:
        if isinstance(config['sma']['trend_strength_threshold'], list) and config['sma']['trend_strength_threshold']:
            trend_thresholds = config['sma']['trend_strength_threshold']
        elif config['sma']['trend_strength_threshold']:
            trend_thresholds = [config['sma']['trend_strength_threshold']]
    
    # Extract trend filter periods
    trend_periods = [200]
    if 'sma' in config and 'trend_filter_period' in config['sma']:
        if isinstance(config['sma']['trend_filter_period'], list) and config['sma']['trend_filter_period']:
            trend_periods = config['sma']['trend_filter_period']
        elif config['sma']['trend_filter_period']:
            trend_periods = [config['sma']['trend_filter_period']]
    
    # Extract regime methods
    regime_methods = ['kmeans']
    if 'regime_detection' in config:
        if 'methods' in config['regime_detection'] and config['regime_detection']['methods']:
            if isinstance(config['regime_detection']['methods'], list):
                regime_methods = config['regime_detection']['methods']
            else:
                regime_methods = [config['regime_detection']['methods']]
        elif 'method' in config['regime_detection'] and config['regime_detection']['method']:
            if isinstance(config['regime_detection']['method'], list):
                regime_methods = config['regime_detection']['method']
            else:
                regime_methods = [config['regime_detection']['method']]
    
    # Extract number of regimes
    n_regimes_list = [3]
    if 'regime_detection' in config and 'n_regimes' in config['regime_detection']:
        if isinstance(config['regime_detection']['n_regimes'], list) and config['regime_detection']['n_regimes']:
            n_regimes_list = config['regime_detection']['n_regimes']
        elif config['regime_detection']['n_regimes']:
            n_regimes_list = [config['regime_detection']['n_regimes']]
    
    # Extract regime stability periods
    stability_periods = [48]
    if 'regime_detection' in config and 'regime_stability_period' in config['regime_detection']:
        if (isinstance(config['regime_detection']['regime_stability_period'], list) and 
            config['regime_detection']['regime_stability_period']):
            stability_periods = config['regime_detection']['regime_stability_period']
        elif config['regime_detection']['regime_stability_period']:
            stability_periods = [config['regime_detection']['regime_stability_period']]
    
    # Extract risk management parameters
    risk_params = {
        'target_vol': [0.15],
        'max_position_size': [1.0],
        'min_position_size': [0.1],
        'max_drawdown_exit': [0.15],
        'profit_taking_threshold': [0.05],
        'trailing_stop_activation': [0.03],
        'trailing_stop_distance': [0.02]
    }
    
    # Update risk parameters from config if available
    if 'risk_management' in config:
        for param_name, param_key in {
            'target_volatility': 'target_vol',
            'max_position_size': 'max_position_size',
            'min_position_size': 'min_position_size',
            'max_drawdown_exit': 'max_drawdown_exit',
            'profit_taking_threshold': 'profit_taking_threshold',
            'trailing_stop_activation': 'trailing_stop_activation',
            'trailing_stop_distance': 'trailing_stop_distance'
        }.items():
            if param_name in config['risk_management']:
                param_value = config['risk_management'][param_name]
                if isinstance(param_value, list) and param_value:
                    risk_params[param_key] = param_value
                elif param_value is not None:
                    risk_params[param_key] = [param_value]
    
    print(f"Building parameter grid with:")
    print(f"- {len(vol_methods)} volatility methods: {vol_methods}")
    print(f"- {len(vol_lookbacks)} lookback periods: {vol_lookbacks}")
    print(f"- {len(regime_smoothings)} regime smoothing periods: {regime_smoothings}")
    print(f"- {len(short_windows)} short windows: {short_windows}")
    print(f"- {len(long_windows)} long windows: {long_windows}")
    print(f"- {len(min_holding_periods)} min holding periods: {min_holding_periods}")
    print(f"- {len(trend_periods)} trend filter periods: {trend_periods}")
    print(f"- {len(trend_thresholds)} trend thresholds: {trend_thresholds}")
    print(f"- {len(regime_methods)} regime methods: {regime_methods}")
    print(f"- {len(n_regimes_list)} regime counts: {n_regimes_list}")
    print(f"- {len(stability_periods)} stability periods: {stability_periods}")
    print(f"- {len(risk_params['target_vol'])} target volatility values: {risk_params['target_vol']}")
    print(f"- {len(risk_params['max_position_size'])} max position sizes: {risk_params['max_position_size']}")
    print(f"- {len(risk_params['min_position_size'])} min position sizes: {risk_params['min_position_size']}")
    print(f"- {len(risk_params['max_drawdown_exit'])} max drawdown exit values: {risk_params['max_drawdown_exit']}")
    print(f"- {len(risk_params['profit_taking_threshold'])} profit taking thresholds: {risk_params['profit_taking_threshold']}")
    print(f"- {len(risk_params['trailing_stop_activation'])} trailing stop activations: {risk_params['trailing_stop_activation']}")
    print(f"- {len(risk_params['trailing_stop_distance'])} trailing stop distances: {risk_params['trailing_stop_distance']}")
    
    if advanced_mode:
        print("ADVANCED MODE ENABLED: Generating comprehensive parameter grid")
    
    # -------------------------------------------------------------------------
    # Use a flag to break out from nested loops as soon as we reach max_combinations
    # -------------------------------------------------------------------------
    limit_reached = False

    # 1) Generate basic combinations (volatility + SMA windows)
    for vol_method in vol_methods:
        if limit_reached:
            break
        for vol_lookback in vol_lookbacks:
            if limit_reached:
                break
            for short_window in short_windows:
                if limit_reached:
                    break
                for long_window in long_windows:
                    if short_window >= long_window:
                        continue
                    if len(param_grid) >= max_combinations:
                        print(f"Reached maximum combinations limit ({max_combinations})")
                        limit_reached = True
                        break
                    # Create base parameter set
                    base_params = {
                        'vol_method': vol_method,
                        'vol_lookback': vol_lookback,
                        'short_window': short_window,
                        'long_window': long_window
                    }
                    
                    # Add additional parameters based on optimization flags
                    if optimize_sma:
                        for min_hold in min_holding_periods:
                            if limit_reached:
                                break
                            for trend_threshold in trend_thresholds:
                                if len(param_grid) >= max_combinations:
                                    print(f"Reached maximum combinations limit ({max_combinations})")
                                    limit_reached = True
                                    break
                                
                                sma_params = base_params.copy()
                                sma_params['min_holding_period'] = min_hold
                                sma_params['trend_strength_threshold'] = trend_threshold
                                param_grid.append(sma_params)
                                
                                if limit_reached:
                                    break
                            if limit_reached:
                                break
                    else:
                        param_grid.append(base_params)
                    
                if limit_reached:
                    break
            if limit_reached:
                break
    
    # 2) If we need to optimize regime parameters and haven't reached the limit
    if optimize_regime and not limit_reached and len(param_grid) < max_combinations:
        current_grid = param_grid.copy()
        param_grid = []  # Clear grid to rebuild with regime parameters
        
        for base_params in current_grid:
            if limit_reached:
                break
            for regime_method in regime_methods:
                if limit_reached:
                    break
                for n_regimes in n_regimes_list:
                    if limit_reached:
                        break
                    for stability in stability_periods:
                        if limit_reached:
                            break
                        for smoothing in regime_smoothings:
                            if len(param_grid) >= max_combinations:
                                print(f"Reached maximum combinations limit ({max_combinations})")
                                limit_reached = True
                                break
                            
                            regime_params = base_params.copy()
                            regime_params['regime_method'] = regime_method
                            regime_params['n_regimes'] = n_regimes
                            regime_params['regime_stability'] = stability
                            regime_params['regime_smoothing'] = smoothing
                            param_grid.append(regime_params)
                            
                            if limit_reached:
                                break
                        if limit_reached:
                            break
                    if limit_reached:
                        break
                if limit_reached:
                    break
    
    # 3) If we need to optimize risk parameters and haven't reached the limit
    if optimize_risk and not limit_reached and len(param_grid) < max_combinations:
        current_grid = param_grid.copy()
        param_grid = []
        
        for base_params in current_grid:
            if limit_reached:
                break
            for target_vol in risk_params['target_vol']:
                if limit_reached:
                    break
                for max_position_size in risk_params['max_position_size']:
                    if limit_reached:
                        break
                    for min_position_size in risk_params['min_position_size']:
                        if limit_reached:
                            break
                        for max_dd in risk_params['max_drawdown_exit']:
                            if len(param_grid) >= max_combinations:
                                print(f"Reached maximum combinations limit ({max_combinations})")
                                limit_reached = True
                                break
                            
                            risk_opt_params = base_params.copy()
                            risk_opt_params['target_vol'] = target_vol
                            risk_opt_params['max_position_size'] = max_position_size
                            risk_opt_params['min_position_size'] = min_position_size
                            risk_opt_params['max_drawdown_exit'] = max_dd
                            
                            if advanced_mode:
                                for profit_threshold in risk_params['profit_taking_threshold']:
                                    if limit_reached:
                                        break
                                    for trailing_act in risk_params['trailing_stop_activation']:
                                        if limit_reached:
                                            break
                                        for trailing_dist in risk_params['trailing_stop_distance']:
                                            if len(param_grid) >= max_combinations:
                                                print(f"Reached maximum combinations limit ({max_combinations})")
                                                limit_reached = True
                                                break
                                            
                                            full_risk_params = risk_opt_params.copy()
                                            full_risk_params['profit_taking_threshold'] = profit_threshold
                                            full_risk_params['trailing_stop_activation'] = trailing_act
                                            full_risk_params['trailing_stop_distance'] = trailing_dist
                                            param_grid.append(full_risk_params)
                                            
                                            if limit_reached:
                                                break
                                        if limit_reached:
                                            break
                                    if limit_reached:
                                        break
                            else:
                                # Not in advanced mode; just add the partial risk set
                                param_grid.append(risk_opt_params)
                                
                            if limit_reached:
                                break
                        if limit_reached:
                            break
                    if limit_reached:
                        break
                if limit_reached:
                    break
    
    # If we ended up with zero combos (edge case), at least put the default in
    if len(param_grid) == 0:
        print("WARNING: Parameter grid generation failed or limited out, using only default parameters")
        param_grid = [default_params]
    
    # Remove any exact duplicates
    unique_params = []
    param_keys = set()
    
    for params in param_grid:
        key = tuple(sorted(params.items()))
        if key not in param_keys:
            param_keys.add(key)
            unique_params.append(params)
    
    param_grid = unique_params
    
    print(f"Final parameter grid has {len(param_grid)} combinations")
    return param_grid

def test_parameter_combination(train_df, params, config):
    """
    Test a specific parameter combination on the training data.
    
    Parameters:
        train_df (DataFrame): Training data
        params (dict): Parameter set to test
        config (dict): Full configuration
        
    Returns:
        dict: Performance metrics
        Series: Equity curve
    """
    try:
        # Extract parameters - use defaults from config if not specified
        vol_method = params.get('vol_method', 'parkinson')
        vol_lookback = params.get('vol_lookback', 20)
        short_window = params.get('short_window', 13)
        long_window = params.get('long_window', 55)
        
        # Optional parameters that may not be in the params dict
        min_holding_period = params.get('min_holding_period', config.get('sma', {}).get('min_holding_period', 24))
        if isinstance(min_holding_period, list):
            min_holding_period = min_holding_period[0]
            
        trend_strength_threshold = params.get('trend_strength_threshold', 
                                            config.get('sma', {}).get('trend_strength_threshold', 0.3))
        if isinstance(trend_strength_threshold, list):
            trend_strength_threshold = trend_strength_threshold[0]
            
        trend_filter_period = params.get('trend_filter_period', 
                                       config.get('sma', {}).get('trend_filter_period', 200))
        if isinstance(trend_filter_period, list):
            trend_filter_period = trend_filter_period[0]
        
        # Regime parameters
        regime_method = params.get('regime_method', config.get('regime_detection', {}).get('method', 'kmeans'))
        if isinstance(regime_method, list) and regime_method:
            regime_method = regime_method[0]
            
        n_regimes = params.get('n_regimes', config.get('regime_detection', {}).get('n_regimes', 3))
        if isinstance(n_regimes, list) and n_regimes:
            n_regimes = n_regimes[0]
            
        regime_stability = params.get('regime_stability', 
                                    config.get('regime_detection', {}).get('regime_stability_period', 48))
        if isinstance(regime_stability, list) and regime_stability:
            regime_stability = regime_stability[0]
            
        regime_smoothing = params.get('regime_smoothing', config.get('volatility', {}).get('regime_smoothing', 5))
        if isinstance(regime_smoothing, list) and regime_smoothing:
            regime_smoothing = regime_smoothing[0]
        
        # Risk management parameters
        target_vol = params.get('target_vol', config.get('risk_management', {}).get('target_volatility', 0.15))
        if isinstance(target_vol, list) and target_vol:
            target_vol = target_vol[0]
            
        max_position_size = params.get('max_position_size', 
                                      config.get('risk_management', {}).get('max_position_size', 1.0))
        if isinstance(max_position_size, list) and max_position_size:
            max_position_size = max_position_size[0]
            
        min_position_size = params.get('min_position_size', 
                                      config.get('risk_management', {}).get('min_position_size', 0.1))
        if isinstance(min_position_size, list) and min_position_size:
            min_position_size = min_position_size[0]
            
        max_drawdown_exit = params.get('max_drawdown_exit', 
                                      config.get('risk_management', {}).get('max_drawdown_exit', 0.15))
        if isinstance(max_drawdown_exit, list) and max_drawdown_exit:
            max_drawdown_exit = max_drawdown_exit[0]
            
        profit_taking_threshold = params.get('profit_taking_threshold', 
                                           config.get('risk_management', {}).get('profit_taking_threshold', 0.05))
        if isinstance(profit_taking_threshold, list) and profit_taking_threshold:
            profit_taking_threshold = profit_taking_threshold[0]
            
        trailing_stop_activation = params.get('trailing_stop_activation', 
                                            config.get('risk_management', {}).get('trailing_stop_activation', 0.03))
        if isinstance(trailing_stop_activation, list) and trailing_stop_activation:
            trailing_stop_activation = trailing_stop_activation[0]
            
        trailing_stop_distance = params.get('trailing_stop_distance', 
                                          config.get('risk_management', {}).get('trailing_stop_distance', 0.02))
        if isinstance(trailing_stop_distance, list) and trailing_stop_distance:
            trailing_stop_distance = trailing_stop_distance[0]
        
        # Calculate volatility
        volatility = calculate_volatility(train_df, method=vol_method, window=vol_lookback)
        
        # Detect regimes
        regimes = detect_volatility_regimes(
            train_df, 
            volatility, 
            method=regime_method,
            n_regimes=n_regimes,
            smoothing_period=regime_smoothing,
            stability_period=regime_stability
        )
        
        # Pre-calculate MAs once for each window
        ma_cache = {}
        for w in set([short_window, long_window]):
            ma_cache[w] = train_df['close_price'].rolling(window=w).mean()
        
        # Extract MAs directly from cache
        short_ma = ma_cache[short_window]
        long_ma = ma_cache[long_window]
        
        # Calculate trend strength and momentum
        trend_strength = calculate_trend_strength(train_df, window=trend_filter_period)
        momentum = calculate_momentum(train_df, window=vol_lookback)
        
        # Generate and filter signals (vectorized operations)
        raw_signal = pd.Series(0, index=train_df.index)
        raw_signal[short_ma > long_ma] = 1
        raw_signal[short_ma < long_ma] = -1
        
        filtered_signal = filter_signals(
            raw_signal, 
            trend_strength, 
            momentum,
            min_trend_strength=trend_strength_threshold
        )
        
        # Apply minimum holding period
        position = apply_min_holding_period(
            filtered_signal,
            min_holding_hours=min_holding_period
        )
        
        # Calculate position size (vectorized)
        position_size = calculate_adaptive_position_size(
            volatility,
            target_vol=target_vol,
            max_size=max_position_size,
            min_size=min_position_size
        )
        
        # Apply position sizing (vectorized)
        sized_position = position * position_size
        
        # Calculate returns (vectorized)
        returns = train_df['close_price'].pct_change().fillna(0)
        
        # Calculate trade returns
        trade_returns = calculate_trade_returns(sized_position, returns)
        
        # Create risk management config
        risk_config = {
            'max_drawdown_exit': max_drawdown_exit,
            'profit_taking_threshold': profit_taking_threshold,
            'trailing_stop_activation': trailing_stop_activation,
            'trailing_stop_distance': trailing_stop_distance
        }
        
        # Apply unified risk management
        managed_position, _ = apply_unified_risk_management(
            train_df,
            sized_position,
            returns,
            volatility,
            regimes,
            risk_config
        )
        
        # Calculate strategy returns (vectorized)
        strategy_returns = managed_position.shift(1).fillna(0) * returns
        
        # Calculate equity curve (vectorized)
        equity_curve = (1 + strategy_returns).cumprod()
        
        # Calculate performance metrics
        metrics = calculate_advanced_metrics(strategy_returns, equity_curve)
        
        return metrics, equity_curve
        
    except Exception as e:
        print(f"  Error testing parameters {params}: {e}")
        
        # Return default metrics when testing fails
        default_metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.1,
            'max_drawdown': -0.1,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'gain_to_pain': 0.0
        }
        
        # Create a simple equity curve
        index = train_df.index
        equity_curve = pd.Series(1.0, index=index)
        
        return default_metrics, equity_curve

def generate_parameter_bounds(config):
    """
    Generate parameter bounds from configuration for Optuna optimization.
    Enhanced to support more parameters and handle edge cases.
    
    Parameters:
        config (dict): Strategy configuration
        
    Returns:
        dict: Dictionary of parameter bounds
    """
    bounds = {}
    
    # Volatility parameters
    vol_methods = config.get('volatility', {}).get('methods', ['parkinson'])
    if isinstance(vol_methods, str):
        vol_methods = [vol_methods]
    
    vol_lookbacks = config.get('volatility', {}).get('lookback_periods', [20])
    if not isinstance(vol_lookbacks, list):
        vol_lookbacks = [vol_lookbacks]
    
    regime_smoothings = config.get('volatility', {}).get('regime_smoothing', [5])
    if not isinstance(regime_smoothings, list):
        regime_smoothings = [regime_smoothings]
    elif len(regime_smoothings) == 0:
        regime_smoothings = [5]
    
    bounds['vol_method'] = (vol_methods, None)  # Categorical
    bounds['vol_lookback'] = (min(vol_lookbacks), max(vol_lookbacks))
    bounds['regime_smoothing'] = (min(regime_smoothings), max(regime_smoothings))
    
    # SMA parameters
    short_windows = config.get('sma', {}).get('short_windows', [5, 8, 13])
    if not isinstance(short_windows, list) or len(short_windows) == 0:
        short_windows = [5, 8, 13]
    
    long_windows = config.get('sma', {}).get('long_windows', [21, 34, 55])
    if not isinstance(long_windows, list) or len(long_windows) == 0:
        long_windows = [21, 34, 55]
    
    min_holding = config.get('sma', {}).get('min_holding_period', [24])
    if not isinstance(min_holding, list):
        min_holding = [min_holding]
    elif len(min_holding) == 0:
        min_holding = [24]
    
    trend_threshold = config.get('sma', {}).get('trend_strength_threshold', [0.3])
    if not isinstance(trend_threshold, list):
        trend_threshold = [trend_threshold]
    elif len(trend_threshold) == 0:
        trend_threshold = [0.3]
    
    trend_periods = config.get('sma', {}).get('trend_filter_period', [200])
    if not isinstance(trend_periods, list):
        trend_periods = [trend_periods]
    elif len(trend_periods) == 0:
        trend_periods = [200]
    
    bounds['short_window'] = (min(short_windows), max(short_windows))
    bounds['long_window'] = (min(long_windows), max(long_windows))
    bounds['min_holding_period'] = (min(min_holding), max(min_holding))
    bounds['trend_strength_threshold'] = (min(trend_threshold), max(trend_threshold))
    bounds['trend_filter_period'] = (min(trend_periods), max(trend_periods))
    
    # Regime parameters
    regime_methods = config.get('regime_detection', {}).get('methods', ['kmeans'])
    if isinstance(regime_methods, str):
        regime_methods = [regime_methods]
    elif not regime_methods:
        regime_methods = ['kmeans']
    
    n_regimes = config.get('regime_detection', {}).get('n_regimes', [3])
    if not isinstance(n_regimes, list):
        n_regimes = [n_regimes]
    elif len(n_regimes) == 0:
        n_regimes = [3]
    
    stability_period = config.get('regime_detection', {}).get('regime_stability_period', [48])
    if not isinstance(stability_period, list):
        stability_period = [stability_period]
    elif len(stability_period) == 0:
        stability_period = [48]
    
    bounds['regime_method'] = (regime_methods, None)  # Categorical
    bounds['n_regimes'] = (min(n_regimes), max(n_regimes))
    bounds['regime_stability'] = (min(stability_period), max(stability_period))
    
    # Risk management parameters
    risk_config = config.get('risk_management', {})
    
    # Extract and process target volatility
    target_vol = risk_config.get('target_volatility', [0.15])
    if not isinstance(target_vol, list):
        target_vol = [target_vol]
    elif len(target_vol) == 0:
        target_vol = [0.15]
    
    # Extract and process max position size
    max_pos_size = risk_config.get('max_position_size', [1.0])
    if not isinstance(max_pos_size, list):
        max_pos_size = [max_pos_size]
    elif len(max_pos_size) == 0:
        max_pos_size = [1.0]
    
    # Extract and process min position size
    min_pos_size = risk_config.get('min_position_size', [0.1])
    if not isinstance(min_pos_size, list):
        min_pos_size = [min_pos_size]
    elif len(min_pos_size) == 0:
        min_pos_size = [0.1]
    
    # Extract and process max drawdown exit
    max_dd_exit = risk_config.get('max_drawdown_exit', [0.15])
    if not isinstance(max_dd_exit, list):
        max_dd_exit = [max_dd_exit]
    elif len(max_dd_exit) == 0:
        max_dd_exit = [0.15]
    
    # Extract and process profit taking threshold
    profit_taking = risk_config.get('profit_taking_threshold', [0.05])
    if not isinstance(profit_taking, list):
        profit_taking = [profit_taking]
    elif len(profit_taking) == 0:
        profit_taking = [0.05]
    
    # Extract and process trailing stop activation
    trailing_activation = risk_config.get('trailing_stop_activation', [0.03])
    if not isinstance(trailing_activation, list):
        trailing_activation = [trailing_activation]
    elif len(trailing_activation) == 0:
        trailing_activation = [0.03]
    
    # Extract and process trailing stop distance
    trailing_distance = risk_config.get('trailing_stop_distance', [0.02])
    if not isinstance(trailing_distance, list):
        trailing_distance = [trailing_distance]
    elif len(trailing_distance) == 0:
        trailing_distance = [0.02]
    
    bounds['target_vol'] = (min(target_vol), max(target_vol))
    bounds['max_position_size'] = (min(max_pos_size), max(max_pos_size))
    bounds['min_position_size'] = (min(min_pos_size), max(min_pos_size))
    bounds['max_drawdown_exit'] = (min(max_dd_exit), max(max_dd_exit))
    bounds['profit_taking_threshold'] = (min(profit_taking), max(profit_taking))
    bounds['trailing_stop_activation'] = (min(trailing_activation), max(trailing_activation))
    bounds['trailing_stop_distance'] = (min(trailing_distance), max(trailing_distance))
    
    # Print bounds for debugging
    print("\nParameter bounds for optimization:")
    for param, bound in bounds.items():
        print(f"  {param}: {bound}")
    
    return bounds


def create_full_parameter_set(config):
    """
    Compatibility function to create a parameter set.
    
    Parameters:
        config (dict): Strategy configuration
        
    Returns:
        dict: Default parameter set
    """
    # Create a default parameter set
    default_params = {
        'vol_method': 'parkinson',
        'vol_lookback': 20,
        'short_window': 13, 
        'long_window': 55,
        'min_holding_period': 24,
        'trend_strength_threshold': 0.3,
        'regime_method': 'kmeans',
        'n_regimes': 3,
        'regime_stability': 48,
        'regime_smoothing': 5,
        'target_vol': 0.15,
        'max_position_size': 1.0,
        'min_position_size': 0.1,
        'max_drawdown_exit': 0.15,
        'profit_taking_threshold': 0.05,
        'trailing_stop_activation': 0.03,
        'trailing_stop_distance': 0.02
    }
    return default_params
