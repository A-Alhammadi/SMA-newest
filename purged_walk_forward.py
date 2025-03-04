# purged_walk_forward.py
# Implementation of purged walk-forward cross-validation for financial time series

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import copy
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import our modules
from parameter_optimization import (
    generate_parameter_grid, test_parameter_combination, 
    calculate_fitness_score, ensure_parameters
)
from enhanced_sma import apply_enhanced_sma_strategy_regime_specific
from volatility import calculate_volatility
from regime_detection import detect_volatility_regimes
from performance_metrics import calculate_advanced_metrics

def generate_purged_walk_forward_periods(start_date, end_date, config):
    """
    Generate purged walk-forward periods based on configuration.
    Implements purging (removing overlap between train/test) and embargo (waiting period after test).
    Supports both expanding and sliding window approaches.
    
    Parameters:
        start_date (str or datetime): Overall start date
        end_date (str or datetime): Overall end date
        config (dict): Walk-forward configuration
        
    Returns:
        list: List of tuples (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end)
    """
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Extract configuration parameters
    window_type = config.get('window_type', 'expanding')  # 'expanding' or 'sliding'
    initial_training_days = config.get('initial_training_days', 365)
    sliding_window_size = config.get('sliding_window_size', 365)  # For sliding window approach
    validation_days = config.get('validation_days', 45)
    test_days = config.get('test_days', 90)
    step_days = config.get('step_days', 90)
    purge_days = config.get('purge_days', 30)
    embargo_days = config.get('embargo_days', 30)
    min_training_size = config.get('min_training_size', 365)
    max_training_size = config.get('max_training_size', 1095)
    max_sections = config.get('max_sections', 25)
    
    # Calculate total days
    total_days = (end - start).days
    
    # Ensure we have enough data
    min_required_days = max(initial_training_days, sliding_window_size) + validation_days + test_days
    if total_days < min_required_days:
        raise ValueError(f"Not enough data for purged walk-forward optimization. "
                        f"Need at least {min_required_days} days.")
    
    # Generate periods
    periods = []
    
    # First period start date
    current_train_start = start
    
    # Track embargo periods
    embargo_periods = []
    
    # Process until we reach the end date or max sections
    section_count = 0
    while section_count < max_sections:
        # Calculate training end date based on window type
        if window_type == 'expanding':
            # Expanding window approach
            if section_count == 0:
                # Initial training period
                train_days = initial_training_days
            else:
                # Expanding window, but not exceeding max_training_size
                train_days = min(initial_training_days + section_count * step_days, max_training_size)
        else:  # 'sliding' window type
            # Fixed-length sliding window approach
            train_days = sliding_window_size
        
        train_end = current_train_start + timedelta(days=train_days)
        
        # Calculate validation period
        val_start = train_end
        val_end = val_start + timedelta(days=validation_days)
        
        # Calculate test period
        test_start = val_end
        test_end = test_start + timedelta(days=test_days)
        
        # Check if we've gone beyond the overall end date
        if test_end > end:
            # Adjust test end to match overall end
            test_end = end
            
            # If test period is too short, skip this period
            if (test_end - test_start).days < test_days / 2:
                break
        
        # Calculate purge and embargo dates
        purge_start = test_start - timedelta(days=purge_days)
        embargo_end = test_end + timedelta(days=embargo_days)
        
        # Store period
        periods.append((
            current_train_start, 
            train_end,
            purge_start,  # For purging from training
            val_start, 
            val_end, 
            test_start, 
            test_end,
            embargo_end  # For next section's training
        ))
        
        # Track this embargo period
        embargo_periods.append((test_end, embargo_end))
        
        # Update for next section - step forward by step_days
        next_section_start = current_train_start + timedelta(days=step_days)
        
        # Check if next section would overlap with an embargo period
        while any(embargo_start <= next_section_start < embargo_end for embargo_start, embargo_end in embargo_periods):
            # Find the latest embargo period that overlaps
            latest_embargo_end = max([e_end for e_start, e_end in embargo_periods if e_start <= next_section_start < e_end])
            # Skip to after the embargo period
            next_section_start = latest_embargo_end
        
        # Set the start of the next training period
        current_train_start = next_section_start
            
        section_count += 1
        
        # Check if next section would go beyond end date
        if current_train_start + timedelta(days=min_training_size) > end:
            break
    
    return periods

def apply_purging_to_training_data(train_df, test_start, purge_days):
    """
    Apply purging to training data by removing data points that overlap with test period events.
    
    Parameters:
        train_df (DataFrame): Training data
        test_start (datetime): Start of test period
        purge_days (int): Number of days to purge before test start
        
    Returns:
        DataFrame: Purged training data
    """
    purge_start = test_start - timedelta(days=purge_days)
    
    # Create purged training data by removing points near test period
    purged_train_df = train_df[train_df.index < purge_start].copy()
    
    # Print purging info
    total_points = len(train_df)
    purged_points = total_points - len(purged_train_df)
    print(f"Purged {purged_points} points ({purged_points/total_points*100:.2f}%) from training data")
    
    return purged_train_df

def optimize_section_parameters_purged(train_df, val_df, test_start, config, section_index, previous_sections):
    """
    Optimize parameters for a purged walk-forward section.
    
    Parameters:
        train_df (DataFrame): Full training data before purging
        val_df (DataFrame): Validation data
        test_start (datetime): Start of test period (for purging)
        config (dict): Full configuration
        section_index (int): Current section index
        previous_sections (list): Results from previous sections
        
    Returns:
        dict: Optimized parameters for each regime
    """
    print(f"\nOptimizing parameters for purged walk-forward section {section_index+1}")
    
    # Get walk-forward configuration
    walk_forward_config = config.get('WALK_FORWARD_CONFIG', {})
    purge_days = walk_forward_config.get('purge_days', 30)
    min_regime_data_points = walk_forward_config.get('min_regime_data_points', 50)
    
    # Apply purging to training data
    purged_train_df = apply_purging_to_training_data(train_df, test_start, purge_days)
    
    # If too much data was purged, use original training data with a warning
    if len(purged_train_df) < walk_forward_config.get('min_training_size', 365) / 2:
        print(f"WARNING: Too much data purged. Using original training data with {len(train_df)} points")
        purged_train_df = train_df
    
    # Get strategy configuration
    strategy_config = config.get('STRATEGY_CONFIG', {})
    
    # Generate parameter combinations with learning from previous sections
    param_grid = generate_parameter_set_with_learning(previous_sections, config, section_index)
    
    # Step 1: Determine volatility and regimes
    # Use parameters from first parameter set for regime detection
    base_params = param_grid[0]
    vol_method = base_params.get('vol_method', 'parkinson')
    vol_lookback = base_params.get('vol_lookback', 20)
    regime_method = base_params.get('regime_method', 'kmeans')
    n_regimes = base_params.get('n_regimes', 3)
    regime_stability = base_params.get('regime_stability', 48)
    regime_smoothing = base_params.get('regime_smoothing', 5)
    
    # Calculate volatility
    volatility = calculate_volatility(
        purged_train_df, 
        method=vol_method, 
        window=vol_lookback
    )
    
    # Detect regimes
    regimes = detect_volatility_regimes(
        purged_train_df, 
        volatility, 
        method=regime_method,
        n_regimes=n_regimes,
        smoothing_period=regime_smoothing,
        stability_period=regime_stability
    )
    
    # Ensure each regime has sufficient data
    regimes = ensure_sufficient_regime_data(purged_train_df, regimes, min_regime_data_points)
    
    # Print regime distribution
    n_regimes = len(regimes.unique())
    print(f"\nDetected {n_regimes} regimes in purged training data")
    
    # Step 2: Optimize parameters for each regime
    regime_best_params = {}
    
    for regime_id in range(n_regimes):
        # Create mask for this regime
        regime_mask = (regimes == regime_id)
        regime_count = regime_mask.sum()
        
        if regime_count < min_regime_data_points:
            print(f"\nInsufficient data for regime {regime_id}: {regime_count} points")
            print(f"Using default parameters for regime {regime_id}")
            regime_best_params[regime_id] = param_grid[0]
            continue
        
        # Extract data for just this regime
        regime_df = purged_train_df.loc[regime_mask].copy()
        
        print(f"\nOptimizing parameters for Regime {regime_id} using {regime_count} data points")
        
        # Track best results for this regime
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        # Get configuration parameters
        print_frequency = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}).get('print_frequency', 20)
        early_stop_threshold = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}).get('early_stop_threshold', 1000)
        min_combinations = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}).get('min_combinations', 100)
        
        # Track non-improvement counter
        early_stop_counter = 0
        
        # Test each parameter combination
        for i, params in enumerate(param_grid):
            try:
                # Test this parameter combination on this regime's data
                metrics, _ = test_parameter_combination(regime_df, params, strategy_config)
                
                # Calculate simplified score
                score = calculate_simple_fitness_score(metrics, config)
                
                # Store best result
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    early_stop_counter = 0  # Reset counter when finding a better score
                    
                    # Print progress when we find a better score
                    print(f"  Found better score: {score:.4f} at combination {i+1}/{len(param_grid)}")
                    print(f"  Return: {metrics.get('total_return', 0):.4%}, "
                         f"Sharpe: {metrics.get('sharpe_ratio', 0):.4f}")
                else:
                    early_stop_counter += 1
                
                # Print progress periodically based on print_frequency
                if (i+1) % print_frequency == 0:
                    print(f"  Tested {i+1}/{len(param_grid)} combinations... (No improvement for {early_stop_counter} combinations)")
                
                # Check for early stopping
                if early_stop_counter >= early_stop_threshold and i+1 >= min_combinations:
                    print(f"  Early stopping after {i+1} combinations (no improvement for {early_stop_counter} combinations)")
                    break
                    
            except Exception as e:
                print(f"  Error testing parameters {params}: {e}")
        
        # Store best parameters for this regime
        if best_params:
            print(f"\nBest parameters for regime {regime_id}:")
            print(f"  {best_params}")
            print(f"  Score: {best_score:.4f}")
            
            if best_metrics:
                print(f"  Sharpe: {best_metrics.get('sharpe_ratio', 0):.4f}")
                print(f"  Return: {best_metrics.get('total_return', 0):.4%}, "
                     f"Max DD: {best_metrics.get('max_drawdown', 0):.4%}")
            
            regime_best_params[regime_id] = best_params
        else:
            print(f"No valid parameters found for regime {regime_id}")
            # Use default parameters
            regime_best_params[regime_id] = param_grid[0]
    
    # Step 3: Validate on validation data
    print("\nValidating regime-specific parameters on validation data")
    
    val_result_df = apply_enhanced_sma_strategy_regime_specific(val_df, regime_best_params, strategy_config)
    val_metrics = calculate_advanced_metrics(val_result_df['strategy_returns'], val_result_df['strategy_cumulative'])
    
    # Print validation results
    print(f"\nValidation Results:")
    print(f"Total Return: {val_metrics['total_return']:.4%}")
    print(f"Sharpe Ratio: {val_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {val_metrics['max_drawdown']:.4%}")
    
    return regime_best_params

def run_purged_walk_forward_optimization(df, config):
    """
    Run purged walk-forward optimization for the strategy.
    
    Parameters:
        df (DataFrame): Full dataset
        config (dict): Full configuration including walk-forward settings
        
    Returns:
        dict: Overall results from purged walk-forward optimization
    """
    print("\nRunning purged walk-forward optimization...")
    start_time = time.time()
    
    # Get walk-forward configuration
    walk_forward_config = config.get('WALK_FORWARD_CONFIG', {})
    save_sections = walk_forward_config.get('save_sections', True)
    window_type = walk_forward_config.get('window_type', 'expanding')
    
    # Get date range from data
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Generate purged walk-forward periods
    periods = generate_purged_walk_forward_periods(start_date, end_date, walk_forward_config)
    
    window_type_str = "sliding window" if window_type == 'sliding' else "expanding window"
    print(f"Generated {len(periods)} purged walk-forward periods using {window_type_str} approach")
    for i, (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end) in enumerate(periods):
        print(f"Period {i+1}:")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} (purged from {purge_start.strftime('%Y-%m-%d')})")
        print(f"  Validation: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        print(f"  Testing: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        print(f"  Embargo until: {embargo_end.strftime('%Y-%m-%d')}")
    
    # Initialize results storage
    all_section_results = []
    combined_test_results = pd.DataFrame()
    
    # Process each walk-forward period
    for i, (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end) in enumerate(periods):
        print(f"\n{'='*80}")
        print(f"Processing purged walk-forward section {i+1}/{len(periods)} using {window_type_str}")
        print(f"{'='*80}")
        
        # Extract data for this period
        train_df = df.loc[train_start:train_end].copy()
        val_df = df.loc[val_start:val_end].copy()
        test_df = df.loc[test_start:test_end].copy()
        
        print(f"Training data (before purging): {len(train_df)} points")
        print(f"Validation data: {len(val_df)} points")
        print(f"Test data: {len(test_df)} points")
        
        # Step 1: Optimize parameters for this section
        section_params = optimize_section_parameters_purged(
            train_df, 
            val_df, 
            test_start,
            config, 
            i, 
            all_section_results
        )
        
        # Step 2: Evaluate on test data
        section_results = evaluate_section_performance(test_df, section_params, config)
        
        # Store section results
        section_results.update({
            'section_index': i,
            'train_start': train_start,
            'train_end': train_end,
            'purge_start': purge_start,
            'val_start': val_start,
            'val_end': val_end,
            'test_start': test_start,
            'test_end': test_end,
            'embargo_end': embargo_end
        })
        
        all_section_results.append(section_results)
        
        # Apply strategy to test data to get equity curve
        result_df = apply_enhanced_sma_strategy_regime_specific(test_df, section_params, config.get('STRATEGY_CONFIG', {}))
        
        # Extract equity curve and combine with test period
        test_equity = result_df[['strategy_cumulative', 'buy_hold_cumulative']].copy()
        test_equity['section'] = i + 1
        combined_test_results = pd.concat([combined_test_results, test_equity])
        
        # Save section results if enabled
        if save_sections:
            save_section_results(section_results, config, i, purged=True)
    
    # Calculate overall performance metrics from combined test results
    overall_return = combined_test_results['strategy_cumulative'].iloc[-1] / combined_test_results['strategy_cumulative'].iloc[0] - 1
    overall_buy_hold = combined_test_results['buy_hold_cumulative'].iloc[-1] / combined_test_results['buy_hold_cumulative'].iloc[0] - 1
    
    # Calculate Sharpe ratio across all test periods
    strategy_returns = combined_test_results['strategy_cumulative'].pct_change().dropna()
    overall_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    # Calculate max drawdown
    running_max = combined_test_results['strategy_cumulative'].cummax()
    drawdown = combined_test_results['strategy_cumulative'] / running_max - 1
    max_drawdown = drawdown.min()
    
    # Print overall results
    print(f"\n{'='*80}")
    print(f"PURGED WALK-FORWARD OPTIMIZATION OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Total Return: {overall_return:.4%}")
    print(f"Buy & Hold Return: {overall_buy_hold:.4%}")
    print(f"Outperformance: {overall_return - overall_buy_hold:.4%}")
    print(f"Overall Sharpe Ratio: {overall_sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4%}")
    print(f"Total sections: {len(all_section_results)}")
    
    # Print sectional performance
    print("\nPerformance by section:")
    for i, section in enumerate(all_section_results):
        metrics = section.get('metrics', {})
        print(f"Section {i+1}: Return={metrics.get('total_return', 0):.4%}, "
              f"Sharpe={metrics.get('sharpe_ratio', 0):.4f}, "
              f"MaxDD={metrics.get('max_drawdown', 0):.4%}")
    
    # Calculate summary statistics
    section_returns = [s.get('metrics', {}).get('total_return', 0) for s in all_section_results]
    section_sharpes = [s.get('metrics', {}).get('sharpe_ratio', 0) for s in all_section_results]
    section_drawdowns = [s.get('metrics', {}).get('max_drawdown', 0) for s in all_section_results]
    
    # Print summary
    print("\nSummary Statistics:")
    print(f"Average Section Return: {np.mean(section_returns):.4%}")
    print(f"Average Section Sharpe: {np.mean(section_sharpes):.4f}")
    print(f"Average Section MaxDD: {np.mean(section_drawdowns):.4%}")
    print(f"% Profitable Sections: {sum(r > 0 for r in section_returns) / len(section_returns):.4%}")
    print(f"% Sections Outperforming Buy & Hold: {sum(s.get('outperformance', 0) > 0 for s in all_section_results) / len(all_section_results):.4%}")
    
    # Plot combined equity curve
    plot_purged_walk_forward_results(combined_test_results, all_section_results, config)
    
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
    
    # Create final result dictionary
    overall_results = {
        'overall_return': overall_return,
        'overall_buy_hold': overall_buy_hold,
        'overall_outperformance': overall_return - overall_buy_hold,
        'overall_sharpe': overall_sharpe,
        'overall_max_drawdown': max_drawdown,
        'section_results': all_section_results,
        'combined_equity': combined_test_results,
        'section_count': len(all_section_results),
        'purged': True  # Mark as purged results
    }
    
    save_optimal_parameters_summary(all_section_results, overall_results, config)

    return overall_results

def generate_parameter_set_with_learning(previous_sections, config, section_index):
    """
    Generate parameter combinations with learning from previous sections.
    
    Parameters:
        previous_sections (list): Results from previous walk-forward sections
        config (dict): Strategy configuration including learning parameters
        section_index (int): Current section index
        
    Returns:
        list: Parameter combinations to test
    """
    # Get learning configuration
    learning_config = config.get('LEARNING_CONFIG', {})
    exploration_pct = learning_config.get('exploration_pct', 0.3)
    exploitation_pct = learning_config.get('exploitation_pct', 0.7)
    top_params_pct = learning_config.get('top_params_pct', 0.2)
    mutation_prob = learning_config.get('mutation_probability', 0.25)
    mutation_factor = learning_config.get('mutation_factor', 0.2)
    max_history = learning_config.get('max_history_sections', 3)
    
    # Get parameter bounds
    param_bounds = generate_parameter_bounds(config.get('STRATEGY_CONFIG', {}))
    
    # Generate base parameter grid for exploration
    base_param_grid = generate_parameter_grid(config.get('STRATEGY_CONFIG', {}), 
                                             config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}))
    
    # For first section or if learning is disabled, just return the base grid
    if section_index == 0 or not learning_config.get('enabled', True) or not previous_sections:
        print(f"Using base parameter grid with {len(base_param_grid)} combinations")
        return base_param_grid
    
    # Calculate how many parameter combinations to test in total
    cv_config = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {})
    max_combinations = cv_config.get('max_combinations', 500)
    
    # Calculate exploration and exploitation counts
    explore_count = int(max_combinations * exploration_pct)
    exploit_count = int(max_combinations * exploitation_pct)
    
    # Make sure we have at least some exploration and exploitation
    explore_count = max(10, min(explore_count, len(base_param_grid)))
    exploit_count = max(10, exploit_count)
    
    print(f"Generating parameter set with learning: {explore_count} exploration, {exploit_count} exploitation")
    
    # Select random parameter sets for exploration
    exploration_params = random.sample(base_param_grid, min(explore_count, len(base_param_grid)))
    
    # Get parameters from previous sections for exploitation
    exploitation_params = []
    
    # Limit to recent history based on max_history
    history_sections = previous_sections[-max_history:] if len(previous_sections) > max_history else previous_sections
    
    # Collect all successful parameter sets from history
    all_params = []
    for section_results in history_sections:
        if 'regime_params' in section_results:
            # Collect parameters from each regime
            for regime_id, params in section_results['regime_params'].items():
                if params and isinstance(params, dict):
                    # Include the regime ID in the parameter set for tracking
                    params_copy = params.copy()
                    params_copy['regime_id'] = regime_id
                    params_copy['score'] = section_results.get('metrics', {}).get('sharpe_ratio', 0)
                    all_params.append(params_copy)
    
    # Sort by performance (score or Sharpe ratio)
    all_params.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Take top performing parameter sets
    top_count = max(1, int(len(all_params) * top_params_pct))
    top_params = all_params[:top_count]
    
    # Create exploitation parameters by mutating top performers
    for i in range(exploit_count):
        # Randomly select a parameter set from top performers
        if not top_params:
            # Fall back to random if no top parameters
            base_param = random.choice(base_param_grid)
        else:
            base_param = random.choice(top_params).copy()
            
            # Remove tracking fields
            if 'regime_id' in base_param:
                del base_param['regime_id']
            if 'score' in base_param:
                del base_param['score']
        
        # Create a mutated copy
        mutated_param = base_param.copy()
        
        # Apply mutations with some probability
        for param_name, param_value in base_param.items():
            # Skip parameters that are not in bounds
            if param_name not in param_bounds:
                continue
                
            # Apply mutation with some probability
            if random.random() < mutation_prob:
                bounds = param_bounds[param_name]
                
                # Handle categorical parameters
                if isinstance(bounds[0], list):
                    # Categorical parameter - select a random value from the list
                    mutated_param[param_name] = random.choice(bounds[0])
                else:
                    # Numerical parameter - mutate within bounds
                    min_val, max_val = bounds
                    mutated_param[param_name] = mutate_parameter(
                        param_value, min_val, max_val, mutation_factor
                    )
        
        # Ensure short window < long window if both parameters exist
        if 'short_window' in mutated_param and 'long_window' in mutated_param:
            if mutated_param['short_window'] >= mutated_param['long_window']:
                # Fix by adjusting short window to be less than long window
                short_bounds = param_bounds['short_window']
                mutated_param['short_window'] = random.randint(
                    short_bounds[0], 
                    min(short_bounds[1], mutated_param['long_window'] - 1)
                )
        
        exploitation_params.append(mutated_param)
    
    # Combine exploration and exploitation parameters
    combined_params = exploration_params + exploitation_params
    
    # Ensure uniqueness
    unique_param_dict = {}
    for param_set in combined_params:
        # Create a key from sorted parameter items
        key = tuple(sorted((k, str(v)) for k, v in param_set.items()))
        unique_param_dict[key] = param_set
    
    final_params = list(unique_param_dict.values())
    
    print(f"Generated {len(final_params)} unique parameter combinations")
    
    return final_params

def generate_parameter_bounds(config):
    """
    Generate parameter bounds from configuration.
    
    Parameters:
        config (dict): Strategy configuration
        
    Returns:
        dict: Dictionary of parameter bounds
    """
    bounds = {}
    
    # Volatility parameters
    vol_methods = config.get('volatility', {}).get('methods', ['parkinson'])
    vol_lookbacks = config.get('volatility', {}).get('lookback_periods', [20])
    bounds['vol_method'] = (vol_methods, None)  # Categorical
    bounds['vol_lookback'] = (min(vol_lookbacks), max(vol_lookbacks))
    
    # SMA parameters
    short_windows = config.get('sma', {}).get('short_windows', [5, 8, 13])
    long_windows = config.get('sma', {}).get('long_windows', [21, 34, 55])
    min_holding = config.get('sma', {}).get('min_holding_period', [24])
    trend_threshold = config.get('sma', {}).get('trend_strength_threshold', [0.3])
    
    bounds['short_window'] = (min(short_windows), max(short_windows))
    bounds['long_window'] = (min(long_windows), max(long_windows))
    bounds['min_holding_period'] = (min(min_holding) if isinstance(min_holding, list) else min_holding, 
                                   max(min_holding) if isinstance(min_holding, list) else min_holding)
    bounds['trend_strength_threshold'] = (min(trend_threshold) if isinstance(trend_threshold, list) else 0.1, 
                                         max(trend_threshold) if isinstance(trend_threshold, list) else 0.5)
    
    # Regime parameters
    regime_methods = config.get('regime_detection', {}).get('methods', ['kmeans'])
    n_regimes = config.get('regime_detection', {}).get('n_regimes', [3])
    stability_period = config.get('regime_detection', {}).get('regime_stability_period', [48])
    
    bounds['regime_method'] = (regime_methods, None)  # Categorical
    bounds['n_regimes'] = (min(n_regimes) if isinstance(n_regimes, list) else n_regimes, 
                          max(n_regimes) if isinstance(n_regimes, list) else n_regimes)
    bounds['regime_stability'] = (min(stability_period) if isinstance(stability_period, list) else 0, 
                                 max(stability_period) if isinstance(stability_period, list) else 72)
    
    # Risk parameters
    target_vol = config.get('risk_management', {}).get('target_volatility', [0.15])
    max_dd_exit = config.get('risk_management', {}).get('max_drawdown_exit', [0.15])
    
    bounds['target_vol'] = (min(target_vol) if isinstance(target_vol, list) else 0.05, 
                           max(target_vol) if isinstance(target_vol, list) else 0.3)
    bounds['max_drawdown_exit'] = (min(max_dd_exit) if isinstance(max_dd_exit, list) else 0.05, 
                                  max(max_dd_exit) if isinstance(max_dd_exit, list) else 0.25)
    
    return bounds

def mutate_parameter(param_value, min_val, max_val, mutation_factor=0.2):
    """
    Mutate a parameter value within bounds.
    
    Parameters:
        param_value (float or int): Original parameter value
        min_val (float or int): Minimum allowable value
        max_val (float or int): Maximum allowable value
        mutation_factor (float): How much to mutate (as percentage of range)
        
    Returns:
        float or int: Mutated parameter value
    """
    if isinstance(param_value, int):
        # For integer parameters
        param_range = max_val - min_val
        mutation_amount = max(1, int(param_range * mutation_factor))
        # Random change within mutation bounds
        change = random.randint(-mutation_amount, mutation_amount)
        mutated_value = int(param_value + change)
        # Ensure within bounds
        return max(min_val, min(max_val, mutated_value))
    else:
        # For float parameters
        param_range = max_val - min_val
        mutation_amount = param_range * mutation_factor
        # Random change within mutation bounds
        change = random.uniform(-mutation_amount, mutation_amount)
        mutated_value = param_value + change
        # Ensure within bounds
        return max(min_val, min(max_val, mutated_value))

def calculate_simple_fitness_score(metrics, config):
    """
    Calculate a simplified fitness score focused on Sharpe ratio and returns.
    
    Parameters:
        metrics (dict): Performance metrics
        config (dict): Configuration with scoring weights
        
    Returns:
        float: Fitness score
    """
    # Extract weights from parameter selection config
    param_selection = config.get('STRATEGY_CONFIG', {}).get('parameter_selection', {})
    sharpe_weight = param_selection.get('sharpe_weight', 0.7)
    return_weight = param_selection.get('return_weight', 0.3)
    
    # Normalize weights
    total_weight = sharpe_weight + return_weight
    sharpe_weight = sharpe_weight / total_weight
    return_weight = return_weight / total_weight
    
    # Calculate weighted score
    fitness_score = (
        sharpe_weight * metrics.get('sharpe_ratio', 0) +
        return_weight * metrics.get('total_return', 0) * 10  # Scale returns to be comparable to Sharpe
    )
    
    # Add penalty for extreme drawdowns
    if metrics.get('max_drawdown', 0) < -0.3:
        fitness_score *= 0.5  # Significant penalty for large drawdowns
    
    return fitness_score

def ensure_sufficient_regime_data(df, regimes, min_data_points=50):
    """
    Ensure each regime has sufficient data points, merging regimes if necessary.
    
    Parameters:
        df (DataFrame): Data frame
        regimes (Series): Detected regimes
        min_data_points (int): Minimum required data points per regime
        
    Returns:
        Series: Adjusted regimes
    """
    # Count data points in each regime
    regime_counts = regimes.value_counts()
    
    # Check if any regime has insufficient data
    insufficient_regimes = [r for r in regime_counts.index if regime_counts[r] < min_data_points]
    
    # If all regimes have sufficient data, return original regimes
    if not insufficient_regimes:
        return regimes
    
    print(f"Found {len(insufficient_regimes)} regimes with insufficient data: {insufficient_regimes}")
    
    # Create a mapping from original regimes to merged regimes
    n_regimes = len(regime_counts)
    merged_regimes = list(range(n_regimes))
    
    # Sort regimes by volatility level
    volatility = calculate_volatility(df)
    regime_vol = {}
    for r in range(n_regimes):
        if r in regime_counts:
            regime_mask = (regimes == r)
            regime_vol[r] = volatility[regime_mask].mean()
    
    # Sort regimes by volatility
    sorted_regimes = sorted(regime_vol.items(), key=lambda x: x[1])
    
    # Map insufficient regimes to their nearest neighbor
    for r in insufficient_regimes:
        # Find nearest neighbor by volatility
        r_vol = regime_vol.get(r, 0)
        
        nearest_idx = 0
        min_dist = float('inf')
        
        for i, (other_r, other_vol) in enumerate(sorted_regimes):
            if other_r != r and other_r not in insufficient_regimes:
                dist = abs(r_vol - other_vol)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        
        # Map to nearest neighbor
        nearest_r = sorted_regimes[nearest_idx][0]
        merged_regimes[r] = nearest_r
        
        print(f"Merging regime {r} into regime {nearest_r}")
    
    # Apply mapping to create new regimes
    adjusted_regimes = regimes.copy()
    for r in range(n_regimes):
        if merged_regimes[r] != r:
            adjusted_regimes[regimes == r] = merged_regimes[r]
    
    # Renumber regimes to be sequential
    regime_map = {}
    next_id = 0
    
    for r in sorted(adjusted_regimes.unique()):
        if r not in regime_map:
            regime_map[r] = next_id
            next_id += 1
    
    # Apply mapping
    final_regimes = adjusted_regimes.map(regime_map)
    
    # Print final regime distribution
    final_counts = final_regimes.value_counts()
    print("\nFinal regime distribution after merging:")
    for r, count in final_counts.items():
        print(f"Regime {r}: {count} data points ({count/len(final_regimes)*100:.2f}%)")
    
    return final_regimes

def evaluate_section_performance(test_df, section_params, config):
    """
    Evaluate performance on a purged walk-forward section.
    
    Parameters:
        test_df (DataFrame): Test data
        section_params (dict): Parameters for the section
        config (dict): Strategy configuration
        
    Returns:
        dict: Performance metrics and results
    """
    # Apply strategy with optimized parameters
    result_df = apply_enhanced_sma_strategy_regime_specific(test_df, section_params, config.get('STRATEGY_CONFIG', {}))
    
    # Calculate performance metrics
    metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
    
    # Calculate buy & hold metrics
    buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
    
    # Print summary
    print(f"\nSection Test Results:")
    print(f"Total Return: {metrics['total_return']:.4%}")
    print(f"Annualized Return: {metrics['annualized_return']:.4%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4%}")
    print(f"Win Rate: {metrics['win_rate']:.4%}")
    print(f"Buy & Hold Return: {buy_hold_return:.4%}")
    print(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}")
    
    return {
        'metrics': metrics,
        'regime_params': section_params,
        'buy_hold_return': buy_hold_return,
        'outperformance': metrics['total_return'] - buy_hold_return
    }

def save_section_results(section_results, config, section_index, purged=True):
    """
    Save results from a purged walk-forward section.
    
    Parameters:
        section_results (dict): Results from the section
        config (dict): Full configuration
        section_index (int): Current section index
        purged (bool): Whether these are purged results
    """
    # Get config settings
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    currency = config.get('STRATEGY_CONFIG', {}).get('CURRENCY', 'BTC/USD')
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d")
    prefix = "purged_walk_forward" if purged else "walk_forward"
    filename = os.path.join(results_dir, f'{prefix}_section_{section_index+1}_{currency.replace("/", "_")}_{timestamp}.pkl')
    
    # Save to file
    joblib.dump(section_results, filename)
    print(f"Section results saved to {filename}")

def plot_purged_walk_forward_results(combined_results, section_results, config):
    """
    Plot results from purged walk-forward optimization.
    
    Parameters:
        combined_results (DataFrame): Combined equity curve
        section_results (list): Results from each section
        config (dict): Full configuration
    """
    # Get config settings
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    currency = config.get('STRATEGY_CONFIG', {}).get('CURRENCY', 'BTC/USD')
    window_type = config.get('WALK_FORWARD_CONFIG', {}).get('window_type', 'expanding')
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Set window type string for title
    window_type_str = "Sliding Window" if window_type == 'sliding' else "Expanding Window"
    
    # Plot equity curve
    ax1 = axs[0]
    ax1.set_title(f'Purged Walk-Forward Optimization Results ({window_type_str}) for {currency}', fontsize=16)
    
    # Plot strategy equity
    ax1.plot(combined_results.index, combined_results['strategy_cumulative'], 'b-', label='Strategy')
    
    # Plot buy & hold equity
    ax1.plot(combined_results.index, combined_results['buy_hold_cumulative'], 'r--', label='Buy & Hold')
    
    # Add section markers and purge/embargo zones
    section_boundaries = []
    for i, section in enumerate(section_results):
        if 'test_start' in section:
            # Add vertical line at section boundary
            test_start = section['test_start']
            ax1.axvline(x=test_start, color='gray', linestyle='--', alpha=0.7)
            
            # Add section label
            ax1.text(test_start, combined_results['strategy_cumulative'].max() * 0.95,
                    f"S{i+1}", horizontalalignment='center')
            
            # Highlight purged and embargo periods
            if 'purge_start' in section and 'embargo_end' in section:
                purge_start = section['purge_start']
                test_end = section['test_end']
                embargo_end = section['embargo_end']
                
                # Highlight purged period
                ax1.axvspan(purge_start, test_start, color='yellow', alpha=0.15, label='_nolegend_')
                
                # Highlight embargo period
                ax1.axvspan(test_end, embargo_end, color='orange', alpha=0.15, label='_nolegend_')
            
            section_boundaries.append(test_start)
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))
    
    # Add legend
    from matplotlib.patches import Patch
    purge_patch = Patch(color='yellow', alpha=0.15, label='Purge Zone')
    embargo_patch = Patch(color='orange', alpha=0.15, label='Embargo Zone')
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([purge_patch, embargo_patch])
    labels.extend(['Purge Zone', 'Embargo Zone'])
    ax1.legend(handles, labels, loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Label axes
    ax1.set_ylabel('Portfolio Value (Multiple of Initial)', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    
    # Add summary statistics as text
    strategy_final = combined_results['strategy_cumulative'].iloc[-1]
    market_final = combined_results['buy_hold_cumulative'].iloc[-1]
    total_return = strategy_final - 1
    market_return = market_final - 1
    outperformance = total_return - market_return
    
    stats_text = (
        f"Total Return: {total_return:.2%}\n"
        f"Market Return: {market_return:.2%}\n"
        f"Outperformance: {outperformance:.2%}"
    )
    
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, 
            bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Plot drawdown
    ax2 = axs[1]
    ax2.set_title('Strategy Drawdown', fontsize=14)
    
    # Calculate drawdown
    running_max = combined_results['strategy_cumulative'].cummax()
    drawdown = combined_results['strategy_cumulative'] / running_max - 1
    
    # Plot drawdown
    ax2.fill_between(combined_results.index, drawdown, 0, color='red', alpha=0.3)
    
    # Format y-axis as percentage
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add section boundaries
    for test_start in section_boundaries:
        ax2.axvline(x=test_start, color='gray', linestyle='--', alpha=0.7)
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Label axes
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    
    # Add maximum drawdown annotation
    max_dd = drawdown.min()
    ax2.text(0.02, 0.05, f"Max Drawdown: {max_dd:.2%}", transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    # Format x-axis dates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_formatter(date_format)
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(results_dir, f'purged_walk_forward_results_{currency.replace("/", "_")}_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Purged walk-forward results plot saved to {filename}")
    
    # Close figure
    plt.close(fig)

def save_optimal_parameters_summary(all_section_results, overall_results, config):
    """
    Save a clear text file summary of optimal parameters and performance for each section.
    
    Parameters:
        all_section_results (list): Results from all sections
        overall_results (dict): Overall performance metrics
        config (dict): Configuration settings
    """
    # Get config settings
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    currency = config.get('STRATEGY_CONFIG', {}).get('CURRENCY', 'BTC/USD')
    window_type = config.get('WALK_FORWARD_CONFIG', {}).get('window_type', 'expanding')
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_type_str = "sliding" if window_type == 'sliding' else "expanding"
    filename = os.path.join(results_dir, f'optimal_parameters_summary_{window_type_str}_{currency.replace("/", "_")}_{timestamp}.txt')
    
    with open(filename, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write(f"PURGED WALK-FORWARD OPTIMIZATION SUMMARY FOR {currency} ({window_type_str.upper()} WINDOW)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write window configuration
        f.write("WINDOW CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"Window Type: {window_type_str.capitalize()}\n")
        
        if window_type == 'sliding':
            sliding_window_size = config.get('WALK_FORWARD_CONFIG', {}).get('sliding_window_size', 365)
            f.write(f"Sliding Window Size: {sliding_window_size} days\n")
        else:
            initial_training = config.get('WALK_FORWARD_CONFIG', {}).get('initial_training_days', 365)
            max_training = config.get('WALK_FORWARD_CONFIG', {}).get('max_training_size', 1095)
            f.write(f"Initial Training Size: {initial_training} days\n")
            f.write(f"Maximum Training Size: {max_training} days\n")
            
        f.write(f"Step Size: {config.get('WALK_FORWARD_CONFIG', {}).get('step_days', 90)} days\n")
        f.write(f"Validation Period: {config.get('WALK_FORWARD_CONFIG', {}).get('validation_days', 45)} days\n")
        f.write(f"Test Period: {config.get('WALK_FORWARD_CONFIG', {}).get('test_days', 90)} days\n")
        f.write(f"Purge Period: {config.get('WALK_FORWARD_CONFIG', {}).get('purge_days', 30)} days\n")
        f.write(f"Embargo Period: {config.get('WALK_FORWARD_CONFIG', {}).get('embargo_days', 30)} days\n\n")
        
        
        # Write overall results
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Return: {overall_results['overall_return']:.4%}\n")
        f.write(f"Buy & Hold Return: {overall_results['overall_buy_hold']:.4%}\n")
        f.write(f"Outperformance: {overall_results['overall_outperformance']:.4%}\n")
        f.write(f"Overall Sharpe Ratio: {overall_results['overall_sharpe']:.4f}\n")
        f.write(f"Overall Max Drawdown: {overall_results['overall_max_drawdown']:.4%}\n")
        f.write(f"Total Sections: {overall_results['section_count']}\n\n")
        
        # Write section-by-section results
        f.write("SECTION-BY-SECTION RESULTS\n")
        f.write("-" * 50 + "\n\n")
        
        for i, section in enumerate(all_section_results):
            metrics = section.get('metrics', {})
            regime_params = section.get('regime_params', {})
            
            f.write(f"SECTION {i+1}\n")
            f.write("-" * 30 + "\n")
            
            # Write date ranges
            if all(k in section for k in ['train_start', 'test_end']):
                f.write(f"Period: {section['train_start'].strftime('%Y-%m-%d')} to {section['test_end'].strftime('%Y-%m-%d')}\n")
                f.write(f"Test Period: {section['test_start'].strftime('%Y-%m-%d')} to {section['test_end'].strftime('%Y-%m-%d')}\n")
            
            # Write performance metrics
            f.write("\nPerformance Metrics:\n")
            f.write(f"  Return: {metrics.get('total_return', 0):.4%}\n")
            f.write(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n")
            f.write(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.4%}\n")
            f.write(f"  Win Rate: {metrics.get('win_rate', 0):.4%}\n")
            f.write(f"  Buy & Hold Return: {section.get('buy_hold_return', 0):.4%}\n")
            f.write(f"  Outperformance: {section.get('outperformance', 0):.4%}\n")
            
            # Write optimal parameters for each regime
            f.write("\nOptimal Parameters:\n")
            for regime_id, params in regime_params.items():
                f.write(f"  REGIME {regime_id}\n")
                
                # Extract key parameters by category
                # SMA parameters
                f.write("    SMA Parameters:\n")
                f.write(f"      Short Window: {params.get('short_window', 'N/A')}\n")
                f.write(f"      Long Window: {params.get('long_window', 'N/A')}\n")
                f.write(f"      Min Holding Period: {params.get('min_holding_period', 'N/A')}\n")
                f.write(f"      Trend Strength Threshold: {params.get('trend_strength_threshold', 'N/A')}\n")
                
                # Volatility parameters
                f.write("    Volatility Parameters:\n")
                f.write(f"      Method: {params.get('vol_method', 'N/A')}\n")
                f.write(f"      Lookback Period: {params.get('vol_lookback', 'N/A')}\n")
                
                # Regime detection parameters
                f.write("    Regime Parameters:\n")
                f.write(f"      Method: {params.get('regime_method', 'N/A')}\n")
                f.write(f"      Stability Period: {params.get('regime_stability', 'N/A')}\n")
                
                # Risk management parameters
                f.write("    Risk Management Parameters:\n")
                f.write(f"      Target Volatility: {params.get('target_vol', 'N/A')}\n")
                f.write(f"      Max Drawdown Exit: {params.get('max_drawdown_exit', 'N/A')}\n")
                f.write(f"      Profit Taking Threshold: {params.get('profit_taking_threshold', 'N/A')}\n")
                f.write(f"      Trailing Stop Activation: {params.get('trailing_stop_activation', 'N/A')}\n")
                f.write(f"      Trailing Stop Distance: {params.get('trailing_stop_distance', 'N/A')}\n")
            
            f.write("\n" + "-" * 30 + "\n\n")
        
        # Write CSV-formatted parameter summary for each section
        f.write("\nCSV-FORMATTED PARAMETER SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        # Write CSV header
        csv_header = "Section,Regime,Return,Sharpe,MaxDD,ShortWindow,LongWindow,TrendThreshold,VolMethod,VolLookback,MaxDDExit,ProfitThreshold\n"
        f.write(csv_header)
        
        # Write CSV rows
        for i, section in enumerate(all_section_results):
            metrics = section.get('metrics', {})
            regime_params = section.get('regime_params', {})
            
            for regime_id, params in regime_params.items():
                row = (f"{i+1},{regime_id},"
                       f"{metrics.get('total_return', 0):.4f},"
                       f"{metrics.get('sharpe_ratio', 0):.4f},"
                       f"{metrics.get('max_drawdown', 0):.4f},"
                       f"{params.get('short_window', 'N/A')},"
                       f"{params.get('long_window', 'N/A')},"
                       f"{params.get('trend_strength_threshold', 'N/A')},"
                       f"{params.get('vol_method', 'N/A')},"
                       f"{params.get('vol_lookback', 'N/A')},"
                       f"{params.get('max_drawdown_exit', 'N/A')},"
                       f"{params.get('profit_taking_threshold', 'N/A')}\n")
                f.write(row)
    
    print(f"Optimal parameters summary saved to: {filename}")
    
    # Also save a simplified CSV file for easy import into Excel/other tools
    csv_filename = os.path.join(results_dir, f'optimal_parameters_{currency.replace("/", "_")}_{timestamp}.csv')
    
    with open(csv_filename, 'w') as f:
        # Write CSV header
        csv_header = "Section,Regime,Return,Sharpe,MaxDD,ShortWindow,LongWindow,TrendThreshold,VolMethod,VolLookback,MaxDDExit,ProfitThreshold\n"
        f.write(csv_header)
        
        # Write CSV rows
        for i, section in enumerate(all_section_results):
            metrics = section.get('metrics', {})
            regime_params = section.get('regime_params', {})
            
            for regime_id, params in regime_params.items():
                row = (f"{i+1},{regime_id},"
                       f"{metrics.get('total_return', 0):.4f},"
                       f"{metrics.get('sharpe_ratio', 0):.4f},"
                       f"{metrics.get('max_drawdown', 0):.4f},"
                       f"{params.get('short_window', 'N/A')},"
                       f"{params.get('long_window', 'N/A')},"
                       f"{params.get('trend_strength_threshold', 'N/A')},"
                       f"{params.get('vol_method', 'N/A')},"
                       f"{params.get('vol_lookback', 'N/A')},"
                       f"{params.get('max_drawdown_exit', 'N/A')},"
                       f"{params.get('profit_taking_threshold', 'N/A')}\n")
                f.write(row)
    
    print(f"Simplified CSV of parameters saved to: {csv_filename}")