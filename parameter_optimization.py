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

from volatility import calculate_volatility
from regime_detection import detect_volatility_regimes
from signals import calculate_trend_strength, calculate_momentum, filter_signals, apply_min_holding_period
from risk_management import calculate_adaptive_position_size, apply_unified_risk_management, calculate_trade_returns
from performance_metrics import calculate_advanced_metrics

def generate_time_series_cv_splits(start_date, end_date, n_splits=5, min_train_size=90, step_forward=30):
    """
    Generate time series cross-validation splits with expanding window.
    
    Parameters:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        n_splits (int): Number of splits
        min_train_size (int): Minimum training size in days
        step_forward (int): Step forward size in days
        
    Returns:
        list: List of (train_start, train_end, val_start, val_end) tuples
    """
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Calculate total days
    total_days = (end - start).days
    
    # Ensure we have enough data
    if total_days < min_train_size + step_forward:
        raise ValueError(f"Not enough data for cross-validation. Need at least {min_train_size + step_forward} days.")
    
    # Generate splits
    splits = []
    
    # Calculate validation period size
    val_size = min(step_forward, total_days // (n_splits + 1))
    
    for i in range(n_splits):
        # Calculate train end date
        train_size = min_train_size + i * step_forward
        if train_size > total_days - val_size:
            break
        
        train_start = start
        train_end = start + timedelta(days=train_size)
        val_start = train_end
        val_end = val_start + timedelta(days=val_size)
        
        # Ensure validation end date doesn't exceed overall end date
        if val_end > end:
            val_end = end
        
        splits.append((train_start, train_end, val_start, val_end))
    
    return splits

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
    print(f"- {len(short_windows)} short windows: {short_windows}")
    print(f"- {len(long_windows)} long windows: {long_windows}")
    print(f"- {len(trend_thresholds)} trend thresholds: {trend_thresholds}")
    print(f"- {len(regime_methods)} regime methods: {regime_methods}")
    
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

def limit_parameter_grid(param_grid, max_size):
    """
    Limit parameter grid size by selecting a representative subset.
    
    Parameters:
        param_grid (list): Full parameter grid
        max_size (int): Maximum number of combinations to test
        
    Returns:
        list: Reduced parameter grid
    """
    if len(param_grid) <= max_size:
        return param_grid
    
    # If the grid is very large, use stratified sampling
    if len(param_grid) > 10000:
        print(f"Parameter grid is very large ({len(param_grid)} combinations)")
        print("Using stratified sampling to create a more representative subset")
        
        # Extract all unique values for key parameters
        vol_methods = set()
        short_windows = set()
        long_windows = set()
        regime_methods = set()
        
        for params in param_grid:
            vol_methods.add(params.get('vol_method', 'parkinson'))
            short_windows.add(params.get('short_window', 13))
            long_windows.add(params.get('long_window', 55))
            regime_methods.add(params.get('regime_method', 'kmeans'))
        
        # Calculate samples per stratum
        samples_per_vol_method = max(1, max_size // (len(vol_methods) * 2))
        
        # Create stratified sample
        sampled_grid = []
        
        # Sample from each volatility method
        for vol_method in vol_methods:
            # Get all parameters with this volatility method
            vol_params = [p for p in param_grid if p.get('vol_method', 'parkinson') == vol_method]
            
            # Take a representative sample
            if len(vol_params) <= samples_per_vol_method:
                sampled_grid.extend(vol_params)
            else:
                # Use systematic sampling
                step = len(vol_params) // samples_per_vol_method
                sampled_grid.extend(vol_params[::step][:samples_per_vol_method])
        
        # If we haven't reached max_size, sample from window combinations
        remaining_samples = max_size - len(sampled_grid)
        if remaining_samples > 0:
            # Get all parameters not already included
            remaining_params = [p for p in param_grid if p not in sampled_grid]
            
            # Group by window combinations
            window_groups = {}
            for params in remaining_params:
                short = params.get('short_window', 13)
                long = params.get('long_window', 55)
                key = (short, long)
                if key not in window_groups:
                    window_groups[key] = []
                window_groups[key].append(params)
            
            # Sample from each window group
            samples_per_window = max(1, remaining_samples // len(window_groups))
            for window_params in window_groups.values():
                if len(window_params) <= samples_per_window:
                    sampled_grid.extend(window_params)
                else:
                    # Use systematic sampling
                    step = len(window_params) // samples_per_window
                    sampled_grid.extend(window_params[::step][:samples_per_window])
                
                # Stop if we've reached our limit
                if len(sampled_grid) >= max_size:
                    break
        
        # If we still haven't reached max_size, add more randomly
        if len(sampled_grid) < max_size:
            remaining_params = [p for p in param_grid if p not in sampled_grid]
            additional_needed = min(max_size - len(sampled_grid), len(remaining_params))
            if additional_needed > 0:
                sampled_grid.extend(random.sample(remaining_params, additional_needed))
        
        return sampled_grid
    else:
        # For smaller grids, just use systematic sampling
        step = len(param_grid) // max_size
        return param_grid[::step][:max_size]

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

def test_parameters_by_regime(train_df, params, config):
    """
    Test a parameter set separately on each regime to find regime-specific performance.
    
    Parameters:
        train_df (DataFrame): Training data
        params (dict): Parameter set to test
        config (dict): Strategy configuration
        
    Returns:
        dict: Performance metrics for each regime
        dict: Equity curves for each regime
    """
    try:
        # Extract volatility parameters
        vol_method = params.get('vol_method', 'parkinson')
        vol_lookback = params.get('vol_lookback', 20)
        
        # Extract regime detection parameters
        regime_method = params.get('regime_method', config.get('regime_detection', {}).get('method', 'kmeans'))
        n_regimes = params.get('n_regimes', config.get('regime_detection', {}).get('n_regimes', 3))
        regime_stability = params.get('regime_stability', config.get('regime_detection', {}).get('regime_stability_period', 48))
        regime_smoothing = params.get('regime_smoothing', config.get('volatility', {}).get('regime_smoothing', 5))
        
        # First step: Calculate volatility and detect regimes
        volatility = calculate_volatility(train_df, method=vol_method, window=vol_lookback)
        
        regimes = detect_volatility_regimes(
            train_df, 
            volatility, 
            method=regime_method,
            n_regimes=n_regimes,
            smoothing_period=regime_smoothing,
            stability_period=regime_stability
        )
        
        # Create return dictionaries
        regime_metrics = {}
        regime_equity_curves = {}
        regime_returns = {}
        
        # Process each regime separately
        for regime_id in range(n_regimes):
            # Get data for this regime only
            regime_mask = (regimes == regime_id)
            if regime_mask.sum() < 100:  # Skip regimes with insufficient data
                print(f"  Insufficient data for regime {regime_id} ({regime_mask.sum()} points)")
                regime_metrics[regime_id] = None
                regime_equity_curves[regime_id] = None
                continue
                
            # Create subset of data for this regime
            regime_df = train_df.loc[regime_mask].copy()
            regime_vol = volatility.loc[regime_mask]
            
            # Ensure data continuity
            if len(regime_df) < 50:
                continue
                
            # Test parameters only on this regime's data
            try:
                metrics, equity_curve = test_parameter_combination(regime_df, params, config)
                regime_metrics[regime_id] = metrics
                regime_equity_curves[regime_id] = equity_curve
                
                # Extract returns for this regime
                returns = regime_df['close_price'].pct_change().fillna(0)
                strategy_returns = calculate_strategy_returns(regime_df, params, config, volatility=regime_vol)
                regime_returns[regime_id] = strategy_returns
                
            except Exception as e:
                print(f"  Error testing parameters for regime {regime_id}: {e}")
                regime_metrics[regime_id] = None
                regime_equity_curves[regime_id] = None
        
        return regime_metrics, regime_equity_curves, regimes, regime_returns
        
    except Exception as e:
        print(f"  Error in regime-specific parameter testing: {e}")
        return {}, {}, None, {}

def calculate_strategy_returns(df, params, config, volatility=None):
    """
    Calculate strategy returns for a given set of parameters without full test.
    Optimized for regime-specific testing.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        params (dict): Parameters to test
        config (dict): Strategy configuration
        volatility (Series, optional): Pre-calculated volatility
        
    Returns:
        Series: Strategy returns
    """
    # Extract SMA parameters
    short_window = params.get('short_window', 13)
    long_window = params.get('long_window', 55)
    trend_strength_threshold = params.get('trend_strength_threshold', 0.3)
    min_holding_period = params.get('min_holding_period', 24)
    
    # Calculate MAs
    short_ma = df['close_price'].rolling(window=short_window).mean()
    long_ma = df['close_price'].rolling(window=long_window).mean()
    
    # Calculate trend strength and momentum
    trend_strength = calculate_trend_strength(df, window=params.get('trend_filter_period', 200))
    momentum = calculate_momentum(df, window=short_window)
    
    # Generate raw signal (vectorized)
    raw_signal = pd.Series(0, index=df.index)
    raw_signal[short_ma > long_ma] = 1
    raw_signal[short_ma < long_ma] = -1
    
    # Filter signals
    filtered_signal = filter_signals(raw_signal, trend_strength, momentum, trend_strength_threshold)
    
    # Apply minimum holding period
    position = apply_min_holding_period(filtered_signal, min_holding_hours=min_holding_period)
    
    # Calculate position size
    if volatility is None:
        vol_method = params.get('vol_method', 'parkinson')
        vol_lookback = params.get('vol_lookback', 20)
        volatility = calculate_volatility(df, method=vol_method, window=vol_lookback)
        
    position_size = calculate_adaptive_position_size(
        volatility,
        target_vol=params.get('target_vol', 0.15),
        max_size=params.get('max_position_size', 1.0),
        min_size=params.get('min_position_size', 0.1)
    )
    
    # Apply position sizing
    sized_position = position * position_size
    
    # Calculate strategy returns
    returns = df['close_price'].pct_change().fillna(0)
    strategy_returns = sized_position.shift(1).fillna(0) * returns
    
    return strategy_returns

def optimize_parameters_by_regime(df, config, splits):
    """
    Optimize strategy parameters separately for each regime.
    Improved version that detects regimes once across the entire training dataset,
    then optimizes parameters for each regime using all relevant data.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        config (dict): Strategy configuration
        splits (list): Cross-validation splits
        
    Returns:
        dict: Optimal parameters for each regime
        list: Results for each validation period
        dict: Regime distribution information
    """
    print("Optimizing parameters separately for each regime...")
    
    # Get CV configuration
    cv_config = config['cross_validation'].get('parameter_testing', {})
    optimization_method = cv_config.get('method', 'greedy')
    max_combinations = cv_config.get('max_combinations', 500)
    n_random_combinations = cv_config.get('n_random_combinations', 100)
    n_regimes = config['regime_detection'].get('n_regimes', 3)
    if isinstance(n_regimes, list):
        n_regimes = n_regimes[0]
    
    # Determine number of workers for parallel processing
    n_workers = max(1, os.cpu_count() // 2)
    print(f"Using {optimization_method} optimization with {n_workers} parallel workers")
    
    # Generate parameter grid
    param_grid = generate_parameter_grid(config, cv_config)
    
    # Limit parameter grid if needed
    if len(param_grid) > max_combinations and optimization_method != 'random':
        print(f"Parameter grid size ({len(param_grid)}) exceeds maximum ({max_combinations})")
        param_grid = limit_parameter_grid(param_grid, max_combinations)
        print(f"Reduced parameter grid to {len(param_grid)} combinations")
    
    if optimization_method == 'random' and len(param_grid) > n_random_combinations:
        param_grid = random.sample(param_grid, n_random_combinations)
        print(f"Randomly selected {n_random_combinations} parameter combinations")
    
    # Store results for each regime
    regime_best_params = {}
    regime_cv_results = []
    regime_distribution = {}
    
    # PHASE 1: Detect regimes across the entire training period
    print("\nPHASE 1: Detecting regimes across entire training dataset...")
    
    # Extract training period - combine all training data from splits
    min_train_start = min([start for start, _, _, _ in splits])
    max_train_end = max([end for _, end, _, _ in splits])    
    training_df = df.loc[min_train_start:max_train_end].copy()
    
    # Use default parameters from first parameter set for regime detection
    base_params = param_grid[0]
    vol_method = base_params.get('vol_method', 'parkinson')
    vol_lookback = base_params.get('vol_lookback', 20)
    regime_method = base_params.get('regime_method', 'kmeans')
    regime_stability = base_params.get('regime_stability', 48)
    regime_smoothing = base_params.get('regime_smoothing', 5)
    
    # Calculate volatility
    volatility = calculate_volatility(
        training_df, 
        method=vol_method, 
        window=vol_lookback
    )
    
    # Detect regimes
    regimes = detect_volatility_regimes(
        training_df, 
        volatility, 
        method=regime_method,
        n_regimes=n_regimes,
        smoothing_period=regime_smoothing,
        stability_period=regime_stability
    )
    
    # Print regime distribution
    regime_counts = regimes.value_counts()
    print("\nRegime distribution in training data:")
    for regime_id, count in regime_counts.items():
        percentage = (count / len(regimes)) * 100
        print(f"Regime {regime_id}: {count} periods ({percentage:.2f}%)")
        
        # Store in regime_distribution for return value
        if 0 not in regime_distribution:
            regime_distribution[0] = {}
        regime_distribution[0][int(regime_id)] = int(count)
    
    # PHASE 2: Optimize parameters for each regime using its specific data
    print("\nPHASE 2: Optimizing parameters for each regime...")
    
    for regime_id in range(n_regimes):
        # Create mask for this regime
        regime_mask = (regimes == regime_id)
        regime_count = regime_mask.sum()
        
        # Reduce the minimum data requirement to 50 (was 100)
        min_required_points = 50
        
        if regime_count < min_required_points:
            print(f"\nInsufficient data for regime {regime_id}: {regime_count} points")
            print(f"Using default parameters for regime {regime_id}")
            regime_best_params[regime_id] = param_grid[0]
            continue
        
        # Extract data for just this regime
        regime_df = training_df.loc[regime_mask].copy()
        
        print(f"\nOptimizing parameters for Regime {regime_id} using {regime_count} data points...")
        
        # Track best results for this regime
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Process candidate parameters based on optimization method
        if optimization_method == 'greedy':
            # For greedy method, process sequentially with early stopping
            early_stop_counter = 0
            early_stop_threshold = cv_config.get('early_stop_threshold', 500)
            min_combinations = cv_config.get('min_combinations', 20)
            print_frequency = cv_config.get('print_frequency', 20)
            
            print(f"  Using greedy optimization with early stopping after {early_stop_threshold} non-improving combinations")
            print(f"  Will test at least {min_combinations} combinations")
            
            # Start timer for progress tracking
            start_time = time.time()
            last_print_time = start_time
            
            for j, params in enumerate(param_grid):
                try:
                    # Test this parameter combination on this regime's data
                    metrics, _ = test_parameter_combination(regime_df, params, config)
                    score = calculate_fitness_score(metrics, config)
                    
                    # Store result for this fold
                    regime_result = {
                        'params': params,
                        'metrics': metrics,
                        'score': score
                    }
                    
                    # Check for improvement and early stopping
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_metrics = metrics
                        early_stop_counter = 0
                        # Print progress when we find a better score
                        print(f"  Found better score: {score:.4f} at combination {j+1}")
                        print(f"  Parameters: {params}")
                    else:
                        early_stop_counter += 1
                    
                    # Early stopping, but only after minimum combinations
                    if early_stop_counter >= early_stop_threshold and j >= min_combinations:
                        print(f"  Early stopping after {j+1} combinations (no improvement for {early_stop_threshold} combinations)")
                        break
                    
                    # Print progress periodically
                    current_time = time.time()
                    if j == 0 or (j+1) % print_frequency == 0 or (j+1) == len(param_grid) or (current_time - last_print_time) > 60:
                        elapsed_time = current_time - start_time
                        combinations_per_second = (j+1) / elapsed_time if elapsed_time > 0 else 0
                        estimated_total_time = len(param_grid) / combinations_per_second if combinations_per_second > 0 else 0
                        remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                        
                        print(f"  Tested {j+1}/{len(param_grid)} combinations ({(j+1)/len(param_grid)*100:.1f}%)")
                        print(f"  Elapsed time: {elapsed_time/60:.1f} mins, Est. remaining: {remaining_time/60:.1f} mins")
                        print(f"  Current best score: {best_score:.4f}, No improvement for {early_stop_counter} combinations")
                        last_print_time = current_time
                    
                except Exception as e:
                    print(f"  Error testing parameters {params}: {e}")
                    
        else:  # Grid or random search - process in parallel
            param_subset = param_grid[:min(500, len(param_grid))]  # Limit to reasonable number
            print(f"  Testing {len(param_subset)} parameter combinations in parallel...")
            
            # Process parameters in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_params = {executor.submit(test_parameter_combination, regime_df, params, config): params 
                                   for params in param_subset}
                
                for i, future in enumerate(as_completed(future_to_params)):
                    params = future_to_params[future]
                    try:
                        metrics, _ = future.result()
                        score = calculate_fitness_score(metrics, config)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_metrics = metrics
                    except Exception as e:
                        print(f"  Error testing parameters {params}: {e}")
                    
                    # Print progress periodically
                    if (i+1) % 20 == 0:
                        print(f"  Tested {i+1}/{len(param_subset)} parameter combinations")
        
        # Store best parameters for this regime
        if best_params is not None:
            print(f"\nBest parameters for regime {regime_id}:")
            print(f"  {best_params}")
            print(f"  Score: {best_score:.4f}")
            
            if best_metrics:
                print(f"  Sharpe: {best_metrics.get('sharpe_ratio', 0):.4f}, "
                      f"Sortino: {best_metrics.get('sortino_ratio', 0):.4f}")
                print(f"  Return: {best_metrics.get('total_return', 0):.4%}, "
                      f"Max DD: {best_metrics.get('max_drawdown', 0):.4%}")
            
            regime_best_params[regime_id] = best_params
            
            # Add to CV results (for compatibility)
            fold_result = {
                'fold': 0,
                'best_params': {regime_id: {'params': best_params, 'score': best_score, 'metrics': best_metrics}}
            }
            regime_cv_results.append(fold_result)
        else:
            print(f"No valid parameters found for regime {regime_id}")
            # Use default parameters
            regime_best_params[regime_id] = param_grid[0]
    
    # Ensure all regimes have parameters
    for regime_id in range(n_regimes):
        if regime_id not in regime_best_params:
            print(f"Warning: No parameters for regime {regime_id}, using default parameters")
            regime_best_params[regime_id] = param_grid[0]
    
    return regime_best_params, regime_cv_results, regime_distribution

def process_parallel_fold(train_df, param_grid, config, n_workers):
    """
    Process parameter testing in parallel for a single fold.
    
    Parameters:
        train_df (DataFrame): Training data for this fold
        param_grid (list): List of parameter combinations to test
        config (dict): Strategy configuration
        n_workers (int): Number of parallel workers
        
    Returns:
        list: Results for this fold
    """
    fold_results = []
    
    # Get configuration settings
    cv_config = config.get('cross_validation', {}).get('parameter_testing', {})
    print_frequency = cv_config.get('print_frequency', 20)
    advanced_mode = cv_config.get('advanced_mode', False)
    
    # In advanced mode, adjust batch size for better progress reporting
    batch_size = 250 if advanced_mode else 100
    
    # Define worker function for parallel processing
    def test_params(params):
        try:
            metrics, _ = test_parameter_combination(train_df, params, config)
            score = calculate_fitness_score(metrics, config)
            return {
                'params': params,
                'metrics': metrics,
                'score': score,
                'error': None
            }
        except Exception as e:
            print(f"  Error testing parameters {params}: {e}")
            return {
                'params': params,
                'metrics': None,
                'score': -np.inf,
                'error': str(e)
            }
    
    # Process parameters in parallel, but in batches for better progress reporting
    total_combinations = len(param_grid)
    processed_combinations = 0
    best_score = -np.inf
    best_result = None
    start_time = time.time()
    last_print_time = start_time
    
    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        current_batch = param_grid[batch_start:batch_end]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all parameter combinations in this batch
            futures = {executor.submit(test_params, params): params for params in current_batch}
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        fold_results.append(result)
                        if result['score'] > best_score and result['error'] is None:
                            best_score = result['score']
                            best_result = result
                except Exception as e:
                    print(f"  Unexpected error processing result: {e}")
                
                # Update progress counter
                processed_combinations += 1
                
                # Print progress
                current_time = time.time()
                if (i+1) % print_frequency == 0 or (i+1) == len(current_batch) or (current_time - last_print_time) > 60:
                    elapsed_time = current_time - start_time
                    combinations_per_second = processed_combinations / elapsed_time if elapsed_time > 0 else 0
                    estimated_total_time = total_combinations / combinations_per_second if combinations_per_second > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                    
                    print(f"  Tested {processed_combinations}/{total_combinations} combinations ({processed_combinations/total_combinations*100:.1f}%)")
                    print(f"  Elapsed time: {elapsed_time/60:.1f} mins, Est. remaining: {remaining_time/60:.1f} mins")
                    if best_result:
                        print(f"  Current best score: {best_score:.4f}")
                    last_print_time = current_time
        
        # If we found a good result, print it after each batch
        if best_result and best_result['metrics']:
            print(f"\n  Best result after {processed_combinations} combinations:")
            print(f"  Score: {best_score:.4f}")
            if advanced_mode:
                # Just print key parameters in advanced mode to avoid overwhelming output
                key_params = {k: v for k, v in best_result['params'].items() 
                              if k in ['vol_method', 'short_window', 'long_window', 'regime_method']}
                print(f"  Key parameters: {key_params}")
            else:
                print(f"  Parameters: {best_result['params']}")
            print(f"  Sharpe: {best_result['metrics']['sharpe_ratio']:.4f}, Sortino: {best_result['metrics']['sortino_ratio']:.4f}")
            print(f"  Return: {best_result['metrics']['total_return']:.4%}, Max DD: {best_result['metrics']['max_drawdown']:.4%}")
    
    # Filter out failed results
    valid_results = [r for r in fold_results if r['error'] is None]
    print(f"  Successfully tested {len(valid_results)}/{len(fold_results)} combinations")
    
    return valid_results

def process_greedy_fold(train_df, param_grid, config):
    """
    Process parameter testing sequentially with early stopping for greedy method.
    
    Parameters:
        train_df (DataFrame): Training data for this fold
        param_grid (list): List of parameter combinations to test
        config (dict): Strategy configuration
        
    Returns:
        list: Results for this fold
    """
    fold_results = []
    best_score = -np.inf
    early_stop_counter = 0
    
    # Get early stopping configuration
    cv_config = config.get('cross_validation', {}).get('parameter_testing', {})
    
    # Get early stopping threshold
    early_stop_threshold = cv_config.get('early_stop_threshold', 500)
    
    # Minimum number of combinations to test regardless of improvement
    min_combinations = cv_config.get('min_combinations', 100)
    
    # How often to print progress
    print_frequency = cv_config.get('print_frequency', 20)
    
    # Check if we're in advanced mode
    advanced_mode = cv_config.get('advanced_mode', False)
    
    # In advanced mode, increase patience to avoid early termination
    if advanced_mode:
        early_stop_threshold = max(early_stop_threshold, 1000)  # More patience in advanced mode
        min_combinations = max(min_combinations, 200)  # Test more combinations in advanced mode
    
    print(f"  Greedy optimization with early stopping after {early_stop_threshold} non-improving combinations")
    print(f"  Will test at least {min_combinations} combinations")
    
    # Start timer for progress tracking
    start_time = time.time()
    last_print_time = start_time
    
    for j, params in enumerate(param_grid):
        try:
            # Test this parameter combination
            metrics, _ = test_parameter_combination(train_df, params, config)
            score = calculate_fitness_score(metrics, config)
            
            # Store result
            fold_result = {
                'params': params,
                'metrics': metrics,
                'score': score
            }
            fold_results.append(fold_result)
            
            # Check for improvement and early stopping
            if score > best_score:
                best_score = score
                early_stop_counter = 0
                # Print progress when we find a better score
                print(f"  Found better score: {score:.4f} at combination {j+1}")
                print(f"  Parameters: {params}")
            else:
                early_stop_counter += 1
            
            # Early stopping, but only after minimum combinations
            if early_stop_counter >= early_stop_threshold and j >= min_combinations:
                print(f"  Early stopping after {j+1} combinations (no improvement for {early_stop_threshold} combinations)")
                break
            
            # Print progress with more information
            current_time = time.time()
            if j == 0 or (j+1) % print_frequency == 0 or (j+1) == len(param_grid) or (current_time - last_print_time) > 60:
                elapsed_time = current_time - start_time
                combinations_per_second = (j+1) / elapsed_time if elapsed_time > 0 else 0
                estimated_total_time = len(param_grid) / combinations_per_second if combinations_per_second > 0 else 0
                remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                
                print(f"  Tested {j+1}/{len(param_grid)} combinations ({(j+1)/len(param_grid)*100:.1f}%)")
                print(f"  Elapsed time: {elapsed_time/60:.1f} mins, Est. remaining: {remaining_time/60:.1f} mins")
                print(f"  Current best score: {best_score:.4f}, No improvement for {early_stop_counter} combinations")
                last_print_time = current_time
                
        except Exception as e:
            print(f"  Error testing parameters {params}: {e}")
    
    return fold_results

def calculate_parameter_stability(param_results):
    """
    Calculate parameter stability score across validation periods.
    
    Parameters:
        param_results (list): List of parameter results across validation periods
        
    Returns:
        float: Stability score (higher is better)
    """
    # If no results, return 0 stability
    if not param_results:
        print("Warning: No parameter results provided for stability calculation.")
        return 0.0
    
    # Extract parameter values from results, ensuring we only include non-None values
    param_values = []
    for result in param_results:
        if result and 'best_params' in result and result['best_params'] is not None:
            param_values.append(result['best_params'])
    
    # If we don't have at least 2 valid parameter sets, can't calculate stability
    if len(param_values) < 2:
        print(f"Warning: Insufficient parameter sets ({len(param_values)}) for stability calculation.")
        return 0.0
    
    # Initialize stability score
    stability_score = 0.0
    parameter_count = 0
    
    # For each parameter, calculate variation across validation periods
    for param_name in param_values[0].keys():
        param_series = [params.get(param_name) for params in param_values if params and param_name in params]
        
        # Skip if not enough data
        if len(param_series) < 2:
            continue
        
        # Filter out None values
        param_series = [p for p in param_series if p is not None]
        if len(param_series) < 2:
            continue
        
        parameter_count += 1
        
        # Handle different parameter types differently
        if all(isinstance(x, (int, float)) for x in param_series):
            # For numeric parameters: calculate coefficient of variation
            param_mean = sum(param_series) / len(param_series)
            
            # Calculate standard deviation
            if len(param_series) <= 1:
                param_std = 0
            else:
                squared_diffs = [(x - param_mean) ** 2 for x in param_series]
                param_std = (sum(squared_diffs) / (len(param_series) - 1)) ** 0.5
            
            # Avoid division by zero
            if param_mean == 0 or param_std == 0:
                cv = 0
            else:
                cv = param_std / abs(param_mean)
                
            # Convert to stability score (higher is better)
            param_stability = 1 / (1 + cv)
        else:
            # For string/categorical parameters: calculate consistency
            # (fraction of values matching the most common value)
            from collections import Counter
            counts = Counter(param_series)
            if counts:
                most_common = counts.most_common(1)
                if most_common:
                    most_common_count = most_common[0][1]
                    param_stability = most_common_count / len(param_series)
                else:
                    param_stability = 0
            else:
                param_stability = 0
        
        # Add to overall stability score
        stability_score += param_stability
    
    # Normalize by number of parameters (avoid division by zero)
    if parameter_count > 0:
        stability_score /= parameter_count
    
    return stability_score

def optimize_parameters_with_cv(df, config, splits):
    """
    Optimize strategy parameters using time series cross-validation.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        config (dict): Strategy configuration
        splits (list): Cross-validation splits
        
    Returns:
        dict: Optimal parameters
        list: Results for each validation period
    """
    print("Optimizing parameters with time series cross-validation...")
    
    # Get CV configuration
    cv_config = config['cross_validation'].get('parameter_testing', {})
    optimization_method = cv_config.get('method', 'greedy')
    max_combinations = cv_config.get('max_combinations', 500)
    n_random_combinations = cv_config.get('n_random_combinations', 100)
    
    # Determine number of workers for parallel processing
    n_workers = max(1, os.cpu_count() // 2)
    print(f"Using {optimization_method} optimization with {n_workers} parallel workers")
    print(f"Max parameter combinations to test: {max_combinations}")
    
    # Generate parameter grid and limit it as needed
    param_grid = generate_parameter_grid(config, cv_config)
    
    if len(param_grid) > max_combinations and optimization_method != 'random':
        print(f"Parameter grid size ({len(param_grid)}) exceeds maximum ({max_combinations})")
        param_grid = limit_parameter_grid(param_grid, max_combinations)
        print(f"Reduced parameter grid to {len(param_grid)} combinations")
    
    if optimization_method == 'random' and len(param_grid) > n_random_combinations:
        param_grid = random.sample(param_grid, n_random_combinations)
        print(f"Randomly selected {n_random_combinations} parameter combinations")
    
    # Default parameter set for fallbacks
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
        'max_drawdown_exit': 0.15,
        'profit_taking_threshold': 0.05,
        'trailing_stop_activation': 0.03,
        'trailing_stop_distance': 0.02
    }
    
    # Store results for each validation period
    cv_results = []
    all_fold_results = []
    
    # Process each cross-validation split
    for i, (train_start, train_end, val_start, val_end) in enumerate(splits):
        print(f"\nCV Fold {i+1}/{len(splits)}:")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Validation: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        
        # Extract training and validation data
        train_df = df.loc[train_start:train_end].copy()
        val_df = df.loc[val_start:val_end].copy()
        
        if len(train_df) < 100 or len(val_df) < 20:
            print("  Not enough data, skipping fold")
            continue
            
        # For greedy method, we need early stopping which is hard to parallelize
        # So we'll process parameters differently based on the method
        try:
            if optimization_method == 'greedy':
                # Process sequentially with early stopping for greedy method
                fold_results = process_greedy_fold(train_df, param_grid, config)
            else:
                # Process in parallel for grid or random methods
                fold_results = process_parallel_fold(train_df, param_grid, config, n_workers)
            
            # Sort results and store best parameters
            if fold_results:
                fold_results.sort(key=lambda x: x['score'], reverse=True)
                best_result = fold_results[0]
                best_params = best_result['params']
                best_score = best_result['score']
                best_metrics = best_result['metrics']
            else:
                print("  No valid results for this fold, using default")
                best_params = param_grid[0] if param_grid else default_params
                best_score = 0.0
                best_metrics = {
                    'sharpe_ratio': 0.0, 
                    'sortino_ratio': 0.0, 
                    'calmar_ratio': 0.0, 
                    'total_return': 0.0, 
                    'max_drawdown': -0.1
                }
        except Exception as e:
            print(f"  Error processing fold: {e}")
            print("  Using default parameters for this fold")
            fold_results = []
            best_params = param_grid[0] if param_grid else default_params
            best_score = 0.0
            best_metrics = {
                'sharpe_ratio': 0.0, 
                'sortino_ratio': 0.0, 
                'calmar_ratio': 0.0, 
                'total_return': 0.0, 
                'max_drawdown': -0.1
            }
        
        # Ensure we have valid parameters
        if best_params is None:
            print(f"  Warning: No best parameters found for fold {i+1}. Using default.")
            best_params = param_grid[0] if param_grid else default_params
        
        # Store fold results with guaranteed valid values
        cv_results.append({
            'fold': i,
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'all_results': fold_results[:10] if fold_results else []  # Store top 10 results
        })
        
        # Extend all fold results
        all_fold_results.extend(fold_results)
        
        # Print best parameters for this fold
        print(f"  Best parameters for fold {i+1}:")
        print(f"    {best_params}")
        print(f"  Best score: {best_score:.4f}")
        if best_metrics:
            print(f"  Sharpe: {best_metrics.get('sharpe_ratio', 0):.4f}, "
                  f"Sortino: {best_metrics.get('sortino_ratio', 0):.4f}, "
                  f"Calmar: {best_metrics.get('calmar_ratio', 0):.4f}")
            print(f"  Return: {best_metrics.get('total_return', 0):.4%}, "
                  f"Max DD: {best_metrics.get('max_drawdown', 0):.4%}")
    
    # Calculate parameter stability across folds - safely handle empty results
    valid_results = [result for result in cv_results if 'best_params' in result and result['best_params'] is not None]
    stability_score = calculate_parameter_stability(valid_results)
    
    print(f"\nParameter stability score: {stability_score:.4f}")
    
    # Combine results across all folds - safely handle potential errors
    combined_results = {}
    for param_set in param_grid:
        try:
            # Create parameter key
            param_key = create_param_key(param_set)
            
            # Find matching results
            matching_results = [result for result in all_fold_results 
                                if result and 'params' in result and are_params_equal(result['params'], param_set)]
            
            if matching_results:
                # Calculate average score across folds
                avg_score = np.mean([result['score'] for result in matching_results if 'score' in result])
                avg_sharpe = np.mean([result['metrics'].get('sharpe_ratio', 0) for result in matching_results if 'metrics' in result])
                avg_sortino = np.mean([result['metrics'].get('sortino_ratio', 0) for result in matching_results if 'metrics' in result])
                avg_calmar = np.mean([result['metrics'].get('calmar_ratio', 0) for result in matching_results if 'metrics' in result])
                avg_return = np.mean([result['metrics'].get('total_return', 0) for result in matching_results if 'metrics' in result])
                avg_drawdown = np.mean([result['metrics'].get('max_drawdown', 0) for result in matching_results if 'metrics' in result])
                
                # Count number of folds
                fold_count = len(matching_results)
                
                # Store combined result
                combined_results[param_key] = {
                    'params': param_set,
                    'avg_score': avg_score,
                    'avg_sharpe': avg_sharpe,
                    'avg_sortino': avg_sortino,
                    'avg_calmar': avg_calmar,
                    'avg_return': avg_return,
                    'avg_drawdown': avg_drawdown,
                    'fold_count': fold_count
                }
        except Exception as e:
            print(f"Error processing parameter set: {e}")
            continue
    
    # Convert to list and sort by average score
    combined_results_list = list(combined_results.values())
    
    # Check if we have any combined results
    if not combined_results_list:
        print("No valid combined results. Using first parameter set.")
        best_overall_params = param_grid[0] if param_grid else default_params
    else:
        # Sort by average score
        combined_results_list.sort(key=lambda x: x.get('avg_score', 0), reverse=True)
        
        # Apply stability weight to score
        stability_weight = config['parameter_selection'].get('stability_weight', 0.5)
        
        # Select best parameter set considering both performance and stability
        for result in combined_results_list:
            # Only consider parameter sets tested in most folds
            min_fold_count = max(1, int(len(cv_results) * 0.8))
            if result['fold_count'] >= min_fold_count:
                # Adjust score with stability
                result['final_score'] = result['avg_score'] * (1 - stability_weight) + stability_score * stability_weight
        
        # Sort by final score
        combined_results_list.sort(key=lambda x: x.get('final_score', x.get('avg_score', 0)), reverse=True)
        
        # Get best parameters
        best_overall_params = combined_results_list[0]['params']
    
    print("\nTop 5 parameter sets across all folds:")
    for i, result in enumerate(combined_results_list[:5]):
        if i < len(combined_results_list):
            print(f"{i+1}. {result['params']}")
            print(f"   Avg Score: {result.get('avg_score', 0):.4f}, Folds: {result.get('fold_count', 0)}/{len(cv_results)}")
            print(f"   Avg Sharpe: {result.get('avg_sharpe', 0):.4f}, "
                  f"Avg Sortino: {result.get('avg_sortino', 0):.4f}, "
                  f"Avg Calmar: {result.get('avg_calmar', 0):.4f}")
            print(f"   Avg Return: {result.get('avg_return', 0):.4%}, "
                  f"Avg Max DD: {result.get('avg_drawdown', 0):.4%}")
            if 'final_score' in result:
                print(f"   Final Score (with stability): {result['final_score']:.4f}")
    
    print(f"\nSelected best parameters: {best_overall_params}")
    
    # Save detailed CV results for future analysis
    if config.get('save_results', True):
        save_dir = config.get('results_dir', 'enhanced_sma_results')
        currency = config.get('currency', 'BTC/USD')
        try:
            save_cv_details(combined_results_list[:20], cv_config, save_dir, currency)
        except Exception as e:
            print(f"Error saving CV details: {e}")
    
    return best_overall_params, cv_results

def save_cv_details(top_results, cv_config, results_dir, currency):
    """
    Save detailed cross-validation results for future analysis.
    
    Parameters:
        top_results (list): Top parameter combinations and their performance
        cv_config (dict): CV configuration settings
        results_dir (str): Directory to save results
        currency (str): Currency symbol
    """
    import os
    import json
    from datetime import datetime
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"cv_results_{currency.replace('/', '_')}_{timestamp}.json")
    
    # Prepare data to save
    save_data = {
        'timestamp': timestamp,
        'currency': currency,
        'cv_config': cv_config,
        'top_results': []
    }
    
    # Convert top results to a serializable format
    for result in top_results:
        serializable_result = {
            'params': {k: str(v) for k, v in result['params'].items()},
            'avg_score': float(result['avg_score']),
            'avg_sharpe': float(result['avg_sharpe']),
            'avg_sortino': float(result['avg_sortino']),
            'avg_calmar': float(result['avg_calmar']),
            'avg_return': float(result['avg_return']),
            'avg_drawdown': float(result['avg_drawdown']),
            'fold_count': int(result['fold_count'])
        }
        if 'final_score' in result:
            serializable_result['final_score'] = float(result['final_score'])
        
        save_data['top_results'].append(serializable_result)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=4)
    
    print(f"Detailed CV results saved to {filename}")
    
    # Also save as CSV for easy analysis
    csv_filename = os.path.join(results_dir, f"cv_results_{currency.replace('/', '_')}_{timestamp}.csv")
    
    try:
        import pandas as pd
        
        # Convert to DataFrame
        results_df = pd.DataFrame([
            {
                **{k: v for k, v in r['params'].items()},
                'avg_score': r['avg_score'],
                'avg_sharpe': r['avg_sharpe'],
                'avg_sortino': r['avg_sortino'],
                'avg_calmar': r['avg_calmar'],
                'avg_return': r['avg_return'],
                'avg_drawdown': r['avg_drawdown'],
                'fold_count': r['fold_count'],
                'final_score': r.get('final_score', r['avg_score'])
            }
            for r in top_results
        ])
        
        # Save to CSV
        results_df.to_csv(csv_filename, index=False)
        print(f"Results also saved as CSV: {csv_filename}")
    except Exception as e:
        print(f"Could not save CSV version: {e}")

def create_full_parameter_set(best_params, config):
    """
    Create a complete parameter set by combining optimized parameters with defaults.
    
    Parameters:
        best_params (dict): Optimized parameters from CV
        config (dict): Full strategy configuration
        
    Returns:
        dict: Complete parameter set
    """
    # Start with the best parameters from optimization
    full_params = best_params.copy()
    
    # Add default parameters for any missing values
    # Volatility parameters
    if 'vol_method' not in full_params:
        full_params['vol_method'] = config['volatility']['methods'][0]
    if 'vol_lookback' not in full_params:
        full_params['vol_lookback'] = config['volatility']['lookback_periods'][0]
    if 'regime_smoothing' not in full_params:
        regime_smoothing = config['volatility'].get('regime_smoothing', 5)
        full_params['regime_smoothing'] = regime_smoothing[0] if isinstance(regime_smoothing, list) else regime_smoothing
    
    # Regime detection parameters
    if 'regime_method' not in full_params:
        regime_method = config['regime_detection'].get('method', 'kmeans')
        if isinstance(regime_method, list):
            regime_method = regime_method[0]
        full_params['regime_method'] = regime_method
    
    if 'n_regimes' not in full_params:
        n_regimes = config['regime_detection'].get('n_regimes', 3)
        full_params['n_regimes'] = n_regimes[0] if isinstance(n_regimes, list) else n_regimes
    
    if 'regime_stability' not in full_params:
        stability = config['regime_detection'].get('regime_stability_period', 48)
        full_params['regime_stability'] = stability[0] if isinstance(stability, list) else stability
    
    # SMA parameters
    if 'short_window' not in full_params:
        full_params['short_window'] = config['sma']['short_windows'][0]
    if 'long_window' not in full_params:
        full_params['long_window'] = config['sma']['long_windows'][0]
    if 'min_holding_period' not in full_params:
        min_holding = config['sma'].get('min_holding_period', 24)
        full_params['min_holding_period'] = min_holding[0] if isinstance(min_holding, list) else min_holding
    if 'trend_strength_threshold' not in full_params:
        threshold = config['sma'].get('trend_strength_threshold', 0.3)
        full_params['trend_strength_threshold'] = threshold[0] if isinstance(threshold, list) else threshold
    if 'trend_filter_period' not in full_params:
        period = config['sma'].get('trend_filter_period', 200)
        full_params['trend_filter_period'] = period[0] if isinstance(period, list) else period
    
    # Risk management parameters
    if 'target_vol' not in full_params:
        target_vol = config['risk_management'].get('target_volatility', 0.15)
        full_params['target_vol'] = target_vol[0] if isinstance(target_vol, list) else target_vol
    
    if 'max_position_size' not in full_params:
        max_size = config['risk_management'].get('max_position_size', 1.0)
        full_params['max_position_size'] = max_size[0] if isinstance(max_size, list) else max_size
    
    if 'min_position_size' not in full_params:
        min_size = config['risk_management'].get('min_position_size', 0.1)
        full_params['min_position_size'] = min_size[0] if isinstance(min_size, list) else min_size
    
    if 'max_drawdown_exit' not in full_params:
        max_dd = config['risk_management'].get('max_drawdown_exit', 0.15)
        full_params['max_drawdown_exit'] = max_dd[0] if isinstance(max_dd, list) else max_dd
    
    if 'profit_taking_threshold' not in full_params:
        profit_threshold = config['risk_management'].get('profit_taking_threshold', 0.05)
        full_params['profit_taking_threshold'] = profit_threshold[0] if isinstance(profit_threshold, list) else profit_threshold
    
    if 'trailing_stop_activation' not in full_params:
        trail_activation = config['risk_management'].get('trailing_stop_activation', 0.03)
        full_params['trailing_stop_activation'] = trail_activation[0] if isinstance(trail_activation, list) else trail_activation
    
    if 'trailing_stop_distance' not in full_params:
        trail_distance = config['risk_management'].get('trailing_stop_distance', 0.02)
        full_params['trailing_stop_distance'] = trail_distance[0] if isinstance(trail_distance, list) else trail_distance
    
    # Handle potential list values - always convert to single value
    for key in full_params:
        val = full_params[key]
        if isinstance(val, list) and len(val) > 0:
            full_params[key] = val[0]
    
    return full_params