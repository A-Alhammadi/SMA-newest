# purged_walk_forward.py
# Implementation of purged walk-forward cross-validation for financial time series
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_contour
from optuna.importance import get_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

# Add this function to purged_walk_forward.py
def standardize_parameter_names(importance_dict):
    """
    Standardize parameter names by mapping shortened names to full parameter names.
    
    Parameters:
        importance_dict (dict): Dictionary of parameter importance values
        
    Returns:
        dict: Dictionary with standardized parameter names
    """
    # Define mapping from shortened to full parameter names
    param_map = {
        'vol': 'vol_method',
        'long': 'long_window',
        'short': 'short_window',
        'trend': 'trend_strength_threshold',
        'regime': 'regime_method',
        'max': 'max_drawdown_exit',
        'min': 'min_position_size',
        'target': 'target_vol',
        'n': 'n_regimes',
        'trailing': 'trailing_stop_distance',
        'profit': 'profit_taking_threshold'
    }
    
    # Create a new dictionary with standardized names
    standardized_dict = {}
    
    # Process each parameter in the importance dictionary
    for param, importance in importance_dict.items():
        # Check if this is a shortened parameter name
        if param in param_map:
            # Use the full parameter name
            full_name = param_map[param]
            
            # If full name already exists, add the importance values
            if full_name in standardized_dict:
                standardized_dict[full_name] += importance
            else:
                standardized_dict[full_name] = importance
        else:
            # Keep original parameter name
            standardized_dict[param] = importance
    
    return standardized_dict

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
    Optimize parameters for a purged walk-forward section using Optuna.
    Enhanced to ensure proper regime-specific optimization and learning.
    
    Parameters:
        train_df (DataFrame): Full training data before purging
        val_df (DataFrame): Validation data
        test_start (datetime): Start of test period (for purging)
        config (dict): Full configuration
        section_index (int): Current section index
        previous_sections (list): Results from previous sections
        
    Returns:
        dict: Optimized parameters for each regime
        dict: Studies by regime for later analysis
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
    
    # Step 1: Determine volatility and regimes
    # Use parameters from previous section's first regime if available for regime detection
    base_params = None
    if previous_sections and section_index > 0 and 'regime_params' in previous_sections[-1]:
        # Try to use parameters from previous section's regime 0 for consistency
        if 0 in previous_sections[-1]['regime_params']:
            base_params = previous_sections[-1]['regime_params'][0]
            print("Using regime detection parameters from previous section's regime 0")
        else:
            # Use first available regime
            regime_id = next(iter(previous_sections[-1]['regime_params']))
            base_params = previous_sections[-1]['regime_params'][regime_id]
            print(f"Using regime detection parameters from previous section's regime {regime_id}")
    
    if base_params is None:
        # Generate default parameters if no previous section
        param_grid = generate_parameter_grid(strategy_config, 
                                           strategy_config.get('cross_validation', {}).get('parameter_testing', {}))
        base_params = param_grid[0]
        print("Using default parameters for regime detection")
    
    # Extract parameters for regime detection
    vol_method = base_params.get('vol_method', 'parkinson')
    vol_lookback = base_params.get('vol_lookback', 20)
    regime_method = base_params.get('regime_method', 'hmm')  # Default to HMM
    n_regimes = base_params.get('n_regimes', 3)
    regime_stability = base_params.get('regime_stability', 48)
    regime_smoothing = base_params.get('regime_smoothing', 5)
    
    # Calculate volatility
    volatility = calculate_volatility(
        purged_train_df, 
        method=vol_method, 
        window=vol_lookback
    )
    
    # Detect regimes - preferring HMM
    try:
        print(f"Attempting regime detection using HMM with {n_regimes} regimes")
        regimes = detect_volatility_regimes(
            purged_train_df, 
            volatility, 
            method='hmm',
            n_regimes=n_regimes,
            smoothing_period=regime_smoothing,
            stability_period=regime_stability,
            verbose=True
        )
    except Exception as e:
        print(f"HMM regime detection failed: {e}")
        print(f"Falling back to {regime_method} method")
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
    regime_counts = regimes.value_counts()
    for regime_id, count in regime_counts.items():
        print(f"Regime {regime_id}: {count} data points ({count/len(regimes)*100:.2f}%)")
    
    # Step 2: Optimize parameters for each regime separately using Optuna
    regime_best_params = {}
    regime_metrics = {}
    regime_studies = {}  # Store studies for later analysis
    
    for regime_id in range(n_regimes):
        # Create mask for this regime
        regime_mask = (regimes == regime_id)
        regime_count = regime_mask.sum()
        
        if regime_count < min_regime_data_points:
            print(f"\nInsufficient data for regime {regime_id}: {regime_count} points")
            print(f"Using default parameters for regime {regime_id}")
            # If we have parameters from the same regime in a previous section, use those
            found_prev_params = False
            if previous_sections:
                for prev_section in reversed(previous_sections):  # Start from most recent
                    if 'regime_params' in prev_section and regime_id in prev_section['regime_params']:
                        regime_best_params[regime_id] = prev_section['regime_params'][regime_id]
                        print(f"  Using parameters from previous section for regime {regime_id}")
                        found_prev_params = True
                        break
            
            if not found_prev_params:
                # Generate a default parameter set
                param_grid = generate_parameter_grid(strategy_config, 
                                                  strategy_config.get('cross_validation', {}).get('parameter_testing', {}))
                regime_best_params[regime_id] = param_grid[0]
            continue
        
        # Extract data for just this regime
        regime_df = purged_train_df.loc[regime_mask].copy()
        
        print(f"\nOptimizing parameters for Regime {regime_id} using {regime_count} data points")
        
        # Generate parameter bounds
        param_bounds = generate_parameter_bounds(strategy_config)
        
        # Determine number of trials based on available data points
        base_n_trials = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}).get('n_random_combinations', 100)
        
        # Scale trials based on data size, but limit to reasonable bounds
        data_size_factor = min(2.0, max(0.5, regime_count / 500))
        n_trials = int(base_n_trials * data_size_factor)
        n_trials = min(300, max(50, n_trials))  # Cap between 50 and 300 trials
        
        print(f"Running {n_trials} Optuna trials for regime {regime_id}")
        
        # Run Optuna optimization
        best_params, study = optimize_parameters_with_optuna(
            regime_df, 
            param_bounds, 
            strategy_config,
            n_trials=n_trials,
            regime_id=regime_id
        )
        
        if best_params:
            # Extract performance metrics from best trial
            best_trial = study.best_trial
            metrics = {
                'sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0),
                'total_return': best_trial.user_attrs.get('total_return', 0),
                'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0),
                'sortino_ratio': best_trial.user_attrs.get('sortino_ratio', 0),
                'calmar_ratio': best_trial.user_attrs.get('calmar_ratio', 0)
            }
            
            # Print best parameters and metrics
            print(f"\nBest parameters for regime {regime_id}:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            print(f"Performance metrics on training data:")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  Total Return: {metrics['total_return']:.4%}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.4%}")
            
            # Store results
            regime_best_params[regime_id] = best_params
            regime_metrics[regime_id] = metrics
            regime_studies[regime_id] = study
            
            # Analyze parameter importance
            try:
                print(f"\nAnalyzing parameter importance for regime {regime_id}")
                param_importance = analyze_parameter_importance(study, config, regime_id=regime_id)
                
                # Print top 5 most important parameters
                top_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Top 5 most important parameters for regime {regime_id}:")
                for param, importance in top_params:
                    print(f"  {param}: {importance:.4f}")
            except Exception as e:
                print(f"Error analyzing parameter importance: {e}")
        else:
            print(f"No valid parameters found for regime {regime_id}")
            # Use default parameters
            param_grid = generate_parameter_grid(
                strategy_config, 
                strategy_config.get('cross_validation', {}).get('parameter_testing', {})
            )
            regime_best_params[regime_id] = param_grid[0]
    
    # Ensure all regimes have parameters
    for regime_id in range(n_regimes):
        if regime_id not in regime_best_params:
            print(f"Warning: No parameters for regime {regime_id}, using default parameters")
            param_grid = generate_parameter_grid(
                strategy_config, 
                strategy_config.get('cross_validation', {}).get('parameter_testing', {})
            )
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
    
    # Calculate regime-specific performance in validation set
    validation_regime_metrics = {}
    for regime_id in range(n_regimes):
        regime_mask = (val_result_df['regime'] == regime_id)
        if regime_mask.any():
            regime_returns = val_result_df.loc[regime_mask, 'strategy_returns']
            if len(regime_returns) > 0:
                regime_cumulative = (1 + regime_returns).cumprod()
                regime_return = regime_cumulative.iloc[-1] - 1
                print(f"  Regime {regime_id} Validation Return: {regime_return:.4%}")
                
                # Calculate Sharpe ratio
                try:
                    from performance_metrics import calculate_sharpe_ratio
                    regime_sharpe = calculate_sharpe_ratio(regime_returns)
                    print(f"  Regime {regime_id} Validation Sharpe: {regime_sharpe:.4f}")
                    validation_regime_metrics[regime_id] = {
                        'return': regime_return,
                        'sharpe': regime_sharpe
                    }
                except Exception as e:
                    print(f"  Error calculating regime-specific metrics: {e}")
    
    # Store overall metrics for return
    for regime_id, params in regime_best_params.items():
        params['validation_score'] = val_metrics['sharpe_ratio']
        
        # Add validation metrics for this regime
        if regime_id in validation_regime_metrics:
            params['validation_regime_return'] = validation_regime_metrics[regime_id]['return']
            params['validation_regime_sharpe'] = validation_regime_metrics[regime_id]['sharpe']
    
    return regime_best_params, regime_studies

def evaluate_section_with_precomputed_regimes(test_df, section_params, config):
    """
    Evaluate performance on a walk-forward section using precomputed regimes.
    """
    # Apply strategy with optimized parameters and precomputed regimes
    result_df = apply_enhanced_sma_strategy_with_precomputed_regimes(test_df, section_params, config.get('STRATEGY_CONFIG', {}))
    
    # Calculate overall performance metrics
    metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
    
    # Calculate buy & hold metrics
    buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
    
    # Calculate regime-specific performance
    regimes = result_df['regime']
    regime_metrics = {}
    
    for regime_id in sorted(section_params.keys()):
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            regime_returns = result_df.loc[regime_mask, 'strategy_returns']
            
            if len(regime_returns) > 0:
                # Calculate metrics for this regime
                regime_cumulative = (1 + regime_returns).cumprod()
                regime_specific_metrics = calculate_advanced_metrics(regime_returns, regime_cumulative)
                regime_metrics[regime_id] = regime_specific_metrics
    
    # Print summary
    print(f"\nSection Test Results:")
    print(f"Total Return: {metrics['total_return']:.4%}")
    print(f"Annualized Return: {metrics['annualized_return']:.4%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.4%}")
    print(f"Win Rate: {metrics['win_rate']:.4%}")
    print(f"Buy & Hold Return: {buy_hold_return:.4%}")
    print(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}")
    
    # Print regime-specific results
    print("\nRegime-Specific Results:")
    for regime_id, regime_metric in regime_metrics.items():
        print(f"Regime {regime_id}:")
        print(f"  Return: {regime_metric['total_return']:.4%}")
        print(f"  Sharpe: {regime_metric['sharpe_ratio']:.4f}")
        print(f"  Max DD: {regime_metric['max_drawdown']:.4%}")
    
    return {
        'metrics': metrics,
        'regime_params': section_params,
        'regime_metrics': regime_metrics,
        'buy_hold_return': buy_hold_return,
        'outperformance': metrics['total_return'] - buy_hold_return
    }

def run_purged_walk_forward_optimization(df, config):
    """
    Run purged walk-forward optimization with regime awareness.
    """
    print("\nRunning regime-aware purged walk-forward optimization...")
    start_time = time.time()
    
    # Get walk-forward configuration
    walk_forward_config = config.get('WALK_FORWARD_CONFIG', {})
    save_sections = walk_forward_config.get('save_sections', True)
    window_type = walk_forward_config.get('window_type', 'expanding')
    regime_aware = walk_forward_config.get('regime_aware_sections', True)
    precompute_regimes = walk_forward_config.get('precompute_regimes', True)
    
    # Get date range from data
    start_date = df.index.min()
    end_date = df.index.max()
    
    # Step 1: Precompute regimes if enabled
    global_regimes = None
    global_volatility = None
    
    if precompute_regimes:
        print("Precomputing regimes for the entire dataset...")
        global_regimes, global_volatility = perform_regime_precomputation(df, config)
        # Add precomputed regimes to dataframe for later use
        df['precomputed_regime'] = global_regimes
    
    # Step 2: Generate walk-forward periods
    if regime_aware and global_regimes is not None:
        print("Generating regime-aware walk-forward periods...")
        periods = generate_regime_aware_walk_forward_periods(start_date, end_date, global_regimes, walk_forward_config)
    else:
        print("Generating standard walk-forward periods...")
        periods = generate_purged_walk_forward_periods(start_date, end_date, walk_forward_config)
    
    window_type_str = "sliding window" if window_type == 'sliding' else "expanding window"
    regime_aware_str = "regime-aware" if regime_aware else "standard"
    print(f"Generated {len(periods)} {regime_aware_str} purged walk-forward periods using {window_type_str} approach")
    
    # Initialize results storage
    all_section_results = []
    combined_test_results = pd.DataFrame()
    all_section_studies = {}
    
    # Process each walk-forward period
    for i, (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end) in enumerate(periods):
        print(f"\n{'='*80}")
        print(f"Processing purged walk-forward section {i+1}/{len(periods)} using {regime_aware_str} {window_type_str}")
        print(f"{'='*80}")
        
        # Extract data for this period
        train_df = df.loc[train_start:train_end].copy()
        val_df = df.loc[val_start:val_end].copy()
        test_df = df.loc[test_start:test_end].copy()
        
        print(f"Training data (before purging): {len(train_df)} points")
        print(f"Validation data: {len(val_df)} points")
        print(f"Test data: {len(test_df)} points")
        
        # Step 3: Optimize parameters for this section
        if precompute_regimes:
            # Use precomputed regimes for optimization
            regime_params, regime_studies = optimize_section_parameters_with_precomputed_regimes(
                train_df, 
                val_df, 
                test_start,
                config, 
                i, 
                all_section_results,
                global_regimes,
                global_volatility
            )
        else:
            # Use standard optimization
            regime_params, regime_studies = optimize_section_parameters_purged(
                train_df, 
                val_df, 
                test_start,
                config, 
                i, 
                all_section_results
            )
        
        # Store studies for this section
        all_section_studies[i] = regime_studies
        
        # Step 4: Evaluate on test data
        if precompute_regimes:
            # Add precomputed regimes to test data
            test_df_with_regime = test_df.copy()
            test_df_with_regime['precomputed_regime'] = global_regimes.loc[test_df.index]
            
            # Evaluate with precomputed regimes
            section_results = evaluate_section_with_precomputed_regimes(
                test_df_with_regime, 
                regime_params, 
                config
            )
        else:
            # Standard evaluation
            section_results = evaluate_section_performance(test_df, regime_params, config)
        
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
            'embargo_end': embargo_end,
            'regime_params': regime_params  # Store optimized parameters
        })
        
        all_section_results.append(section_results)
        
        # Apply strategy to test data to get equity curve
        if precompute_regimes:
            # Apply with precomputed regimes
            result_df = apply_enhanced_sma_strategy_with_precomputed_regimes(
                test_df_with_regime,
                regime_params,
                config.get('STRATEGY_CONFIG', {})
            )
        else:
            # Standard strategy application
            result_df = apply_enhanced_sma_strategy_regime_specific(
                test_df,
                regime_params,
                config.get('STRATEGY_CONFIG', {})
            )
        
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
    print(f"REGIME-AWARE PURGED WALK-FORWARD OPTIMIZATION OVERALL RESULTS")
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
    
    # Perform global parameter sensitivity analysis
    print("\nPerforming global parameter sensitivity analysis...")
    sensitivity_results = track_parameter_performance(all_section_results, all_section_studies, config)
    
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
        'purged': True,  # Mark as purged results
        'regime_aware': regime_aware,
        'precomputed_regimes': precompute_regimes,
        'global_regimes': global_regimes,
        'global_volatility': global_volatility,
        'parameter_importance': sensitivity_results.get('global_importance', {})
    }
    
    save_optimal_parameters_summary(all_section_results, overall_results, config)
    
    # Save final model and sensitivity results
    save_final_model_with_sensitivity(overall_results, config)

    return overall_results

def save_final_model_with_sensitivity(overall_results, config):
    """
    Save the final model along with parameter sensitivity analysis.
    
    Parameters:
        overall_results (dict): Overall results from purged walk-forward optimization
        config (dict): Full configuration
    """
    import os
    import joblib
    from datetime import datetime
    
    # Get output directory
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get currency
    currency = config.get('STRATEGY_CONFIG', {}).get('CURRENCY', 'BTC/USD')
    currency_str = currency.replace("/", "_")
    
    # Create filename
    filename = os.path.join(results_dir, f"final_model_with_sensitivity_{currency_str}_{timestamp}.pkl")
    
    # Standardize parameter importance names
    if 'parameter_importance' in overall_results:
        overall_results['parameter_importance'] = standardize_parameter_names(overall_results['parameter_importance'])
    
    # Prepare data to save
    final_data = {
        'overall_results': overall_results,
        'timestamp': timestamp,
        'currency': currency,
        'parameter_importance': overall_results.get('parameter_importance', {})
    }
    
    # Save to file
    joblib.dump(final_data, filename)
    print(f"Final model with sensitivity analysis saved to {filename}")
    
    # Create a simplified parameter importance CSV
    csv_filename = os.path.join(results_dir, f"parameter_importance_{currency_str}_{timestamp}.csv")
    
    with open(csv_filename, 'w') as f:
        f.write("Parameter,Importance\n")
        for param, importance in sorted(overall_results.get('parameter_importance', {}).items(), 
                                    key=lambda x: x[1], reverse=True):
            f.write(f"{param},{importance:.6f}\n")
    
    print(f"Parameter importance CSV saved to {csv_filename}")

def optimize_parameters_with_optuna(df, param_bounds, strategy_config, n_trials=100, regime_id=None):
    """
    Optimize parameters using Optuna for a specific regime.
    Updated to use Optuna settings from config.
    
    Parameters:
        df (DataFrame): Training data for the regime
        param_bounds (dict): Dictionary of parameter bounds
        strategy_config (dict): Strategy configuration
        n_trials (int): Number of Optuna trials to run (will be overridden by config if specified)
        regime_id (int, optional): Regime ID being optimized, for logging
        
    Returns:
        dict: Best parameters found by Optuna
        optuna.Study: Completed Optuna study for further analysis
    """
    # Get Optuna settings from config
    cv_config = strategy_config.get('cross_validation', {}).get('parameter_testing', {})
    optuna_config = cv_config.get('optuna_settings', {})
    
    # Override n_trials with config value if specified
    config_n_trials = cv_config.get('n_trials')
    if config_n_trials is not None:
        n_trials = config_n_trials
    
    # Get timeout setting (optional)
    timeout = cv_config.get('timeout', None)
    
    print(f"\nStarting Optuna optimization" + (f" for regime {regime_id}" if regime_id is not None else ""))
    print(f"Running {n_trials} trials with Optuna" + (f" (timeout: {timeout}s)" if timeout else ""))
    
    # Extract parameter selection weights
    param_selection = strategy_config.get('parameter_selection', {})
    sharpe_weight = param_selection.get('sharpe_weight', 0.7)
    return_weight = param_selection.get('return_weight', 0.3)
    
    # Create objective function for Optuna
    def objective(trial):
        # Build parameter set from trial suggestions
        params = {}
        
        # Volatility parameters
        if 'vol_method' in param_bounds:
            methods = param_bounds['vol_method'][0]
            params['vol_method'] = trial.suggest_categorical('vol_method', methods)
            
        if 'vol_lookback' in param_bounds:
            min_val, max_val = param_bounds['vol_lookback']
            params['vol_lookback'] = trial.suggest_int('vol_lookback', min_val, max_val)
        
        # SMA parameters
        if 'short_window' in param_bounds and 'long_window' in param_bounds:
            short_min, short_max = param_bounds['short_window']
            long_min, long_max = param_bounds['long_window']
            
            # Ensure short_window < long_window by using relative suggestion
            params['short_window'] = trial.suggest_int('short_window', short_min, short_max)
            # Long window should be at least 1.5x the short window
            params['long_window'] = trial.suggest_int('long_window', 
                                                    max(long_min, int(params['short_window'] * 1.5)), 
                                                    long_max)
        
        if 'min_holding_period' in param_bounds:
            min_val, max_val = param_bounds['min_holding_period']
            params['min_holding_period'] = trial.suggest_int('min_holding_period', min_val, max_val, step=6)
        
        if 'trend_strength_threshold' in param_bounds:
            min_val, max_val = param_bounds['trend_strength_threshold']
            params['trend_strength_threshold'] = trial.suggest_float('trend_strength_threshold', 
                                                                  min_val, max_val, step=0.05)
        
        # Regime parameters
        if 'regime_method' in param_bounds:
            methods = param_bounds['regime_method'][0]
            params['regime_method'] = trial.suggest_categorical('regime_method', methods)
            
        if 'n_regimes' in param_bounds:
            min_val, max_val = param_bounds['n_regimes']
            params['n_regimes'] = trial.suggest_int('n_regimes', min_val, max_val)
            
        if 'regime_stability' in param_bounds:
            min_val, max_val = param_bounds['regime_stability']
            params['regime_stability'] = trial.suggest_int('regime_stability', min_val, max_val, step=12)
            
        if 'regime_smoothing' in param_bounds:
            min_val, max_val = param_bounds.get('regime_smoothing', (3, 21))
            params['regime_smoothing'] = trial.suggest_int('regime_smoothing', min_val, max_val)
        
        # Risk management parameters
        if 'target_vol' in param_bounds:
            min_val, max_val = param_bounds['target_vol']
            params['target_vol'] = trial.suggest_float('target_vol', min_val, max_val, step=0.05)
            
        if 'max_drawdown_exit' in param_bounds:
            min_val, max_val = param_bounds['max_drawdown_exit']
            params['max_drawdown_exit'] = trial.suggest_float('max_drawdown_exit', min_val, max_val, step=0.01)
            
        if 'profit_taking_threshold' in param_bounds:
            min_val, max_val = param_bounds.get('profit_taking_threshold', (0.03, 0.3))
            params['profit_taking_threshold'] = trial.suggest_float('profit_taking_threshold', 
                                                                 min_val, max_val, step=0.01)
            
        if 'trailing_stop_activation' in param_bounds:
            min_val, max_val = param_bounds.get('trailing_stop_activation', (0.01, 0.15))
            params['trailing_stop_activation'] = trial.suggest_float('trailing_stop_activation', 
                                                                  min_val, max_val, step=0.01)
            
        if 'trailing_stop_distance' in param_bounds:
            min_val, max_val = param_bounds.get('trailing_stop_distance', (0.01, 0.1))
            params['trailing_stop_distance'] = trial.suggest_float('trailing_stop_distance', 
                                                                min_val, max_val, step=0.005)
            
        # Max/min position size
        if 'max_position_size' in param_bounds:
            min_val, max_val = param_bounds.get('max_position_size', (0.5, 1.0))
            params['max_position_size'] = trial.suggest_float('max_position_size', min_val, max_val, step=0.1)
            
        if 'min_position_size' in param_bounds:
            min_val, max_val = param_bounds.get('min_position_size', (0.0, 0.3))
            params['min_position_size'] = trial.suggest_float('min_position_size', min_val, max_val, step=0.05)
        
        try:
            # Test this parameter combination
            metrics, _ = test_parameter_combination(df, params, strategy_config)
            
            # Calculate score based on weights
            score = (sharpe_weight * metrics.get('sharpe_ratio', 0) + 
                    return_weight * metrics.get('total_return', 0) * 10)
            
            # Penalize for extreme drawdowns
            if metrics.get('max_drawdown', 0) < -0.3:
                score *= (1 + metrics.get('max_drawdown', 0))  # Penalty factor
                
            # Store additional metrics for later analysis
            # These are pruned by Optuna but helpful for reporting
            trial.set_user_attr('sharpe_ratio', metrics.get('sharpe_ratio', 0))
            trial.set_user_attr('total_return', metrics.get('total_return', 0))
            trial.set_user_attr('max_drawdown', metrics.get('max_drawdown', 0))
            trial.set_user_attr('sortino_ratio', metrics.get('sortino_ratio', 0))
            trial.set_user_attr('calmar_ratio', metrics.get('calmar_ratio', 0))
            
            return score
            
        except Exception as e:
            print(f"Error in Optuna trial: {e}")
            # Return a very low score for failed trials
            return -100.0
    
    # Create Optuna study
    study_name = f"regime_optimization_{regime_id}" if regime_id is not None else "regime_optimization"
    
    # Create pruner based on config
    pruner_type = optuna_config.get('pruner', 'median')
    n_startup_trials = optuna_config.get('n_startup_trials', 10)
    n_warmup_steps = optuna_config.get('n_warmup_steps', 5)
    
    if pruner_type == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, 
            n_warmup_steps=n_warmup_steps, 
            interval_steps=1
        )
    elif pruner_type == 'percentile':
        pruner = optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps
        )
    elif pruner_type == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=n_warmup_steps,
            max_resource=100,
            reduction_factor=3
        )
    elif pruner_type == 'none':
        pruner = optuna.pruners.NopPruner()
    else:
        print(f"Unknown pruner type '{pruner_type}', using MedianPruner")
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, 
            n_warmup_steps=n_warmup_steps
        )
    
    # Create study with configured pruner
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=pruner
    )
    
    # Get progress bar setting
    show_progress_bar = optuna_config.get('show_progress_bar', True)
    
    # Run optimization with timeout if specified
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=show_progress_bar
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Print best parameters
    print(f"\nBest Optuna parameters" + (f" for regime {regime_id}" if regime_id is not None else "") + ":")
    print(f"Optimization score: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Convert Optuna params to our parameter format
    final_params = {}
    for param_name, param_value in best_params.items():
        final_params[param_name] = param_value
    
    return final_params, study

def analyze_parameter_importance(study, config, output_dir=None, regime_id=None):
    """
    Analyze parameter importance from an Optuna study.
    
    Parameters:
        study (optuna.Study): Completed Optuna study
        config (dict): Strategy configuration
        output_dir (str, optional): Directory to save plots, if None uses config.RESULTS_DIR
        regime_id (int, optional): Regime ID for file naming
        
    Returns:
        dict: Dictionary of parameter importance scores
    """
    if output_dir is None:
        output_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    
    # Ensure directory exists
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    regime_suffix = f"_regime_{regime_id}" if regime_id is not None else ""
    
    try:
        # Use Optuna's built-in parameter importance
        param_importance = get_param_importances(study)
        
        # Create formatted param importance
        formatted_importance = {}
        for param_name, importance in param_importance.items():
            formatted_importance[param_name] = importance
        
        # Standardize parameter names
        formatted_importance = standardize_parameter_names(formatted_importance)
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        filename = os.path.join(output_dir, f"param_importance{regime_suffix}_{timestamp}.png")
        fig.write_image(filename)
        print(f"Parameter importance plot saved to {filename}")
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        filename = os.path.join(output_dir, f"optimization_history{regime_suffix}_{timestamp}.png")
        fig.write_image(filename)
        print(f"Optimization history saved to {filename}")
        
        # Try to plot contour for most important parameters
        if len(param_importance) >= 2:
            top_params = list(param_importance.keys())[:2]
            try:
                fig = plot_contour(study, params=top_params)
                filename = os.path.join(output_dir, f"contour_plot{regime_suffix}_{timestamp}.png")
                fig.write_image(filename)
                print(f"Contour plot saved to {filename}")
            except Exception as e:
                print(f"Could not create contour plot: {e}")
        
        return formatted_importance
    
    except Exception as e:
        print(f"Error in parameter importance analysis: {e}")
        
        # Fall back to custom importance analysis
        return calculate_custom_parameter_importance(study, output_dir, regime_id)

def calculate_custom_parameter_importance(study, output_dir, regime_id=None):
    """
    Calculate parameter importance using a custom approach when Optuna's built-in
    method fails (e.g., with too few trials or strongly correlated parameters).
    
    Parameters:
        study (optuna.Study): Completed Optuna study
        output_dir (str): Directory to save plots
        regime_id (int, optional): Regime ID for file naming
        
    Returns:
        dict: Dictionary of parameter importance scores
    """
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    regime_suffix = f"_regime_{regime_id}" if regime_id is not None else ""
    
    # Extract trials data into DataFrame
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # Get parameters
            params = trial.params.copy()
            
            # Add trial value (optimization score)
            params['score'] = trial.value
            
            # Add metrics if available
            for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'sortino_ratio', 'calmar_ratio']:
                if metric in trial.user_attrs:
                    params[metric] = trial.user_attrs[metric]
            
            trials_data.append(params)
    
    if not trials_data:
        print("No complete trials found for importance analysis")
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(trials_data)
    
    # Ensure we have a score column
    if 'score' not in df.columns:
        print("No score column found in trials data")
        return {}
    
    # Remove non-parameter columns for correlation analysis
    metric_columns = ['score', 'sharpe_ratio', 'total_return', 'max_drawdown', 'sortino_ratio', 'calmar_ratio']
    param_columns = [col for col in df.columns if col not in metric_columns]
    
    # Calculate correlation with score
    correlations = {}
    for param in param_columns:
        # Skip categorical parameters
        if df[param].dtype == 'object':
            continue
        
        corr = df[param].corr(df['score'])
        correlations[param] = abs(corr)  # Use absolute correlation as importance
    
    # Try to use RandomForest for feature importance
    rf_importances = {}
    try:
        # Prepare data
        X = df[param_columns]
        y = df['score']
        
        # Handle categorical features
        X_processed = pd.get_dummies(X, drop_first=True)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        
        # Train RandomForest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # Get feature importances
        feature_importances = rf.feature_importances_
        
        # Map back to original parameter names
        feature_names = X_processed.columns
        for name, importance in zip(feature_names, feature_importances):
            # Extract base parameter name from dummy variables
            if '_' in name:
                base_param = name.split('_')[0]
            else:
                base_param = name
                
            if base_param not in rf_importances:
                rf_importances[base_param] = 0
            rf_importances[base_param] += importance
            
    except Exception as e:
        print(f"RandomForest importance calculation failed: {e}")
    
    # Combine correlation and RF importances
    combined_importances = {}
    for param in set(list(correlations.keys()) + list(rf_importances.keys())):
        # Get importances, defaulting to 0 if not available
        corr_imp = correlations.get(param, 0)
        rf_imp = rf_importances.get(param, 0)
        
        # Average the importances, with more weight to RF if available
        if param in rf_importances:
            combined_importances[param] = 0.3 * corr_imp + 0.7 * rf_imp
        else:
            combined_importances[param] = corr_imp
    
    # Normalize importances to sum to 1
    total_importance = sum(combined_importances.values())
    if total_importance > 0:
        for param in combined_importances:
            combined_importances[param] /= total_importance
    
    # Standardize parameter names
    combined_importances = standardize_parameter_names(combined_importances)
    
    # Generate plots
    try:
        # 1. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[param_columns + ['score']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        filename = os.path.join(output_dir, f"correlation_heatmap{regime_suffix}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to {filename}")
        
        # 2. Parameter importance bar chart
        plt.figure(figsize=(12, 8))
        
        # Sort importances
        sorted_importances = sorted(combined_importances.items(), key=lambda x: x[1], reverse=True)
        params = [x[0] for x in sorted_importances]
        importances = [x[1] for x in sorted_importances]
        
        # Plot
        sns.barplot(x=importances, y=params, palette='viridis')
        plt.title('Parameter Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Parameter')
        plt.tight_layout()
        filename = os.path.join(output_dir, f"custom_param_importance{regime_suffix}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Custom parameter importance plot saved to {filename}")
        
    except Exception as e:
        print(f"Error generating importance plots: {e}")
    
    return combined_importances

def track_parameter_performance(section_results, all_regime_studies, config):
    """
    Create a comprehensive parameter performance tracking database and analyze
    which parameters consistently lead to good performance across different regimes.
    
    Parameters:
        section_results (list): Results from all sections
        all_regime_studies (dict): Dictionary of completed Optuna studies by section and regime
        config (dict): Strategy configuration
        
    Returns:
        dict: Analysis results including global parameter importance
    """
    # Ensure we have the output directory
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    import os
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize data collection
    all_trials_data = []
    
    # Process all studies
    for section_idx, regime_studies in all_regime_studies.items():
        for regime_id, study in regime_studies.items():
            # Extract data from this study
            for trial in study.trials:
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue
                
                # Get trial data
                trial_data = {
                    'section': section_idx,
                    'regime': regime_id,
                    'trial': trial.number,
                    'score': trial.value
                }
                
                # Add parameters
                for param_name, param_value in trial.params.items():
                    trial_data[param_name] = param_value
                
                # Add user attributes (metrics)
                for attr_name, attr_value in trial.user_attrs.items():
                    trial_data[attr_name] = attr_value
                
                all_trials_data.append(trial_data)
    
    # Create DataFrame
    if not all_trials_data:
        print("No trial data to analyze")
        return {}
    
    trials_df = pd.DataFrame(all_trials_data)
    
    # Save the complete trial database
    trials_csv = os.path.join(results_dir, f"all_trials_data_{timestamp}.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Complete trials database saved to {trials_csv}")
    
    # Analyze global parameter importance
    global_importance = {}
    
    # Extract parameter columns (exclude metadata and metrics)
    meta_columns = ['section', 'regime', 'trial', 'score', 'sharpe_ratio', 
                    'total_return', 'max_drawdown', 'sortino_ratio', 'calmar_ratio']
    param_columns = [col for col in trials_df.columns if col not in meta_columns]
    
    try:
        # Prepare data for ML-based importance
        X = trials_df[param_columns]
        y = trials_df['score']
        
        # Handle categorical features
        X_processed = pd.get_dummies(X, drop_first=True)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed)
        
        # 1. Linear regression for coefficient-based importance
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        
        lr_importances = {}
        for i, col in enumerate(X_processed.columns):
            base_param = col.split('_')[0] if '_' in col else col
            if base_param not in lr_importances:
                lr_importances[base_param] = 0
            lr_importances[base_param] += abs(lr.coef_[i])
        
        # 2. RandomForest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        rf_importances = {}
        for i, col in enumerate(X_processed.columns):
            base_param = col.split('_')[0] if '_' in col else col
            if base_param not in rf_importances:
                rf_importances[base_param] = 0
            rf_importances[base_param] += rf.feature_importances_[i]
        
        # 3. Correlation-based importance
        corr_importances = {}
        for param in param_columns:
            if trials_df[param].dtype != 'object':  # Skip categorical
                corr = abs(trials_df[param].corr(trials_df['score']))
                if not np.isnan(corr):
                    corr_importances[param] = corr
        
        # Combine importance measures
        for param in set(list(lr_importances.keys()) + list(rf_importances.keys()) + list(corr_importances.keys())):
            lr_imp = lr_importances.get(param, 0)
            rf_imp = rf_importances.get(param, 0)
            corr_imp = corr_importances.get(param, 0)
            
            # Weighted combination
            global_importance[param] = 0.2 * lr_imp + 0.5 * rf_imp + 0.3 * corr_imp
        
        # Normalize to sum to 1
        total_importance = sum(global_importance.values())
        if total_importance > 0:
            for param in global_importance:
                global_importance[param] /= total_importance
        
        # Standardize parameter names for consistency
        global_importance = standardize_parameter_names(global_importance)
        
        # Sort and prepare for visualization with standardized names
        sorted_importances = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Plot global parameter importance
        plt.figure(figsize=(12, 8))
        
        # Extract sorted data for plotting
        params = [x[0] for x in sorted_importances]
        importances = [x[1] for x in sorted_importances]
        
        # Plot
        sns.barplot(x=importances, y=params, palette='viridis')
        plt.title('Global Parameter Importance Across All Regimes and Sections')
        plt.xlabel('Importance Score')
        plt.ylabel('Parameter')
        plt.tight_layout()
        filename = os.path.join(results_dir, f"global_param_importance_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Global parameter importance plot saved to {filename}")
        
        # Create a text summary of parameter rankings
        with open(os.path.join(results_dir, f"parameter_importance_summary_{timestamp}.txt"), 'w') as f:
            f.write("PARAMETER IMPORTANCE SUMMARY\n")
            f.write("===========================\n\n")
            f.write("Parameters ranked by global importance:\n")
            for param, importance in sorted_importances:
                f.write(f"{param}: {importance:.4f}\n")
            
            f.write("\n\nRecommended focus parameters (top 30%):\n")
            top_n = max(1, int(len(sorted_importances) * 0.3))
            for param, importance in sorted_importances[:top_n]:
                f.write(f"{param}: {importance:.4f}\n")
            
            f.write("\n\nLess important parameters (bottom 30%):\n")
            for param, importance in sorted_importances[-top_n:]:
                f.write(f"{param}: {importance:.4f}\n")
        
        # Create a correlation matrix between parameters
        plt.figure(figsize=(16, 14))
        param_correlation = trials_df[param_columns].corr()
        sns.heatmap(param_correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Parameter Correlation Matrix (Across All Trials)')
        plt.tight_layout()
        filename = os.path.join(results_dir, f"parameter_correlation_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter correlation matrix saved to {filename}")
        
    except Exception as e:
        print(f"Error in global parameter importance analysis: {e}")
    
    return {
        'global_importance': global_importance,
        'trials_df': trials_df
    }

def generate_parameter_set_with_learning(previous_sections, config, section_index, target_regime=None):
    """
    Generate parameter combinations with learning from previous sections.
    Enhanced to support regime-specific learning - each regime only learns from same regime in past.
    
    Parameters:
        previous_sections (list): Results from previous walk-forward sections
        config (dict): Strategy configuration including learning parameters
        section_index (int): Current section index
        target_regime (int, optional): Specific regime to generate parameters for (if None, generates for all)
        
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
    
    if target_regime is not None:
        print(f"Generating parameter set with learning for regime {target_regime}: {explore_count} exploration, {exploit_count} exploitation")
    else:
        print(f"Generating parameter set with learning: {explore_count} exploration, {exploit_count} exploitation")
    
    # Select random parameter sets for exploration
    exploration_params = random.sample(base_param_grid, min(explore_count, len(base_param_grid)))
    
    # Get parameters from previous sections for exploitation - WITH REGIME FILTERING
    exploitation_params = []
    
    # Limit to recent history based on max_history
    history_sections = previous_sections[-max_history:] if len(previous_sections) > max_history else previous_sections
    
    # Collect all successful parameter sets from history - FILTERED BY REGIME if requested
    all_params = []
    for section_results in history_sections:
        if 'regime_params' in section_results:
            # Collect parameters from each regime
            for regime_id, params in section_results['regime_params'].items():
                # Skip if we're targeting a specific regime and this isn't it
                if target_regime is not None and int(regime_id) != int(target_regime):
                    continue
                    
                if params and isinstance(params, dict):
                    # Include the regime ID in the parameter set for tracking
                    params_copy = params.copy()
                    params_copy['regime_id'] = regime_id
                    
                    # Get performance metrics for this regime
                    if 'regime_metrics' in section_results and regime_id in section_results['regime_metrics']:
                        regime_metrics = section_results['regime_metrics'][regime_id]
                        params_copy['score'] = regime_metrics.get('sharpe_ratio', 0)
                    else:
                        # Fallback - use overall metrics
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
    
    print(f"Generated {len(final_params)} unique parameter combinations" + 
          (f" for regime {target_regime}" if target_regime is not None else ""))
    
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

def perform_regime_precomputation(df, config):
    """
    Precompute volatility regimes for the entire dataset using HMM.
    """
    # Get regime detection configuration
    regime_config = config.get('WALK_FORWARD_CONFIG', {})
    method = regime_config.get('precomputed_regime_method', 'hmm')
    n_regimes = regime_config.get('precomputed_n_regimes', 3)
    smoothing_period = regime_config.get('regime_smoothing', 5)
    stability_period = regime_config.get('regime_stability_period', 48)
    
    # Calculate volatility for entire dataset
    volatility = calculate_volatility(
        df,
        method=config.get('STRATEGY_CONFIG', {}).get('volatility', {}).get('methods', ['parkinson'])[0],
        window=config.get('STRATEGY_CONFIG', {}).get('volatility', {}).get('lookback_periods', [20])[0]
    )
    
    # Detect regimes using the specified method (preferring HMM)
    print(f"Precomputing regimes for entire dataset using {method} with {n_regimes} regimes")
    
    try:
        # Try HMM with multiple attempts and parameters to ensure robustness
        regimes = detect_volatility_regimes(
            df,
            volatility,
            method=method,
            n_regimes=n_regimes,
            smoothing_period=smoothing_period,
            stability_period=stability_period,
            verbose=True
        )
    except Exception as e:
        print(f"Error in primary regime detection: {e}")
        print("Falling back to kmeans for global regime detection")
        regimes = detect_volatility_regimes(
            df,
            volatility,
            method='kmeans',
            n_regimes=n_regimes,
            smoothing_period=smoothing_period,
            stability_period=stability_period
        )
    
    # Print regime distribution
    print(f"\nGlobal regime distribution for the entire dataset:")
    regime_counts = regimes.value_counts()
    for regime_id, count in regime_counts.items():
        print(f"Regime {regime_id}: {count} data points ({count/len(regimes)*100:.2f}%)")
    
    return regimes, volatility

def generate_regime_aware_walk_forward_periods(start_date, end_date, regimes, config):
    """
    Generate regime-aware purged walk-forward periods to ensure sufficient data for each regime.
    """
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Extract basic configuration parameters
    walk_forward_config = config.get('WALK_FORWARD_CONFIG', {})
    window_type = walk_forward_config.get('window_type', 'expanding')
    initial_training_days = walk_forward_config.get('initial_training_days', 365)
    sliding_window_size = walk_forward_config.get('sliding_window_size', 365)
    validation_days = walk_forward_config.get('validation_days', 45)
    test_days = walk_forward_config.get('test_days', 90)
    step_days = walk_forward_config.get('step_days', 90)
    purge_days = walk_forward_config.get('purge_days', 30)
    embargo_days = walk_forward_config.get('embargo_days', 30)
    max_sections = walk_forward_config.get('max_sections', 25)
    
    # Extract regime-aware configuration parameters
    min_regime_points = walk_forward_config.get('min_regime_points_per_section', 30)
    adaptive_length = walk_forward_config.get('adaptive_section_length', True)
    max_expansion = walk_forward_config.get('max_section_expansion', 30)
    balanced_regimes = walk_forward_config.get('balanced_regime_sections', True)
    
    # Generate standard periods first (time-based)
    standard_periods = generate_purged_walk_forward_periods(start_date, end_date, walk_forward_config)
    
    if not standard_periods:
        raise ValueError("Failed to generate base walk-forward periods")
    
    # Now adjust periods to ensure sufficient regime representation
    adjusted_periods = []
    embargo_periods = []
    
    for i, (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end) in enumerate(standard_periods):
        # Check regime representation in validation period
        val_regimes = regimes[val_start:val_end]
        val_regime_counts = val_regimes.value_counts()
        
        # Check regime representation in test period
        test_regimes = regimes[test_start:test_end]
        test_regime_counts = test_regimes.value_counts()
        
        # Check if any regime has insufficient representation
        needs_adjustment = False
        all_regimes = set(regimes.unique())
        
        # Check validation and test period regime counts
        for regime_id in all_regimes:
            if regime_id not in val_regime_counts or val_regime_counts[regime_id] < min_regime_points:
                needs_adjustment = True
                print(f"Section {i+1}: Validation period has insufficient data for regime {regime_id} "
                     f"({val_regime_counts.get(regime_id, 0)} points)")
            
            if regime_id not in test_regime_counts or test_regime_counts[regime_id] < min_regime_points:
                needs_adjustment = True
                print(f"Section {i+1}: Test period has insufficient data for regime {regime_id} "
                     f"({test_regime_counts.get(regime_id, 0)} points)")
        
        # If adjustment is needed and adaptive length is enabled
        if needs_adjustment and adaptive_length:
            # Try to expand test period first
            print(f"Adjusting section {i+1} to ensure regime representation")
            
            # Calculate how much we can expand test period
            next_section_start = train_start + timedelta(days=step_days) if i + 1 < len(standard_periods) else end
            max_test_expansion = min(
                max_expansion,
                (next_section_start - test_end).days if next_section_start > test_end else 0
            )
            
            if max_test_expansion > 0:
                # Try expanding test period
                expanded_test_end = test_end + timedelta(days=max_test_expansion)
                expanded_embargo_end = expanded_test_end + timedelta(days=embargo_days)
                
                # Check if expanded test period improves regime representation
                expanded_test_regimes = regimes[test_start:expanded_test_end]
                expanded_test_counts = expanded_test_regimes.value_counts()
                
                improved = True
                for regime_id in all_regimes:
                    if regime_id not in expanded_test_counts or expanded_test_counts[regime_id] < min_regime_points:
                        if regime_id not in test_regime_counts or expanded_test_counts.get(regime_id, 0) <= test_regime_counts.get(regime_id, 0):
                            improved = False
                
                if improved:
                    print(f"Expanded test period from {test_end} to {expanded_test_end} (+{max_test_expansion} days)")
                    test_end = expanded_test_end
                    embargo_end = expanded_embargo_end
                    test_regimes = expanded_test_regimes
                    test_regime_counts = expanded_test_counts
            
            # Check if we need to expand validation period
            needs_val_adjustment = False
            for regime_id in all_regimes:
                if regime_id not in val_regime_counts or val_regime_counts[regime_id] < min_regime_points:
                    needs_val_adjustment = True
            
            if needs_val_adjustment:
                # Calculate how much we can expand validation period
                max_val_expansion = min(
                    max_expansion,
                    (test_start - val_end).days if test_start > val_end else 0
                )
                
                if max_val_expansion > 0:
                    # Try expanding validation period
                    expanded_val_end = val_end + timedelta(days=max_val_expansion)
                    
                    # Check if expanded validation period improves regime representation
                    expanded_val_regimes = regimes[val_start:expanded_val_end]
                    expanded_val_counts = expanded_val_regimes.value_counts()
                    
                    improved = True
                    for regime_id in all_regimes:
                        if regime_id not in expanded_val_counts or expanded_val_counts[regime_id] < min_regime_points:
                            if regime_id not in val_regime_counts or expanded_val_counts.get(regime_id, 0) <= val_regime_counts.get(regime_id, 0):
                                improved = False
                    
                    if improved:
                        print(f"Expanded validation period from {val_end} to {expanded_val_end} (+{max_val_expansion} days)")
                        val_end = expanded_val_end
                        val_regimes = expanded_val_regimes
                        val_regime_counts = expanded_val_counts
                
                # If validation end changed, update purge start and test start
                purge_start = test_start - timedelta(days=purge_days)
        
        # Store adjusted period
        adjusted_periods.append((
            train_start, 
            train_end,
            purge_start,
            val_start, 
            val_end, 
            test_start, 
            test_end,
            embargo_end
        ))
        
        # Track this embargo period
        embargo_periods.append((test_end, embargo_end))
    
    # Print summary of adjustments
    print(f"\nGenerated {len(adjusted_periods)} regime-aware walk-forward periods")
    for i, (train_start, train_end, purge_start, val_start, val_end, test_start, test_end, embargo_end) in enumerate(adjusted_periods):
        print(f"Period {i+1}:")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')} (purged from {purge_start.strftime('%Y-%m-%d')})")
        print(f"  Validation: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        print(f"  Testing: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        
        # Print regime distribution for this section
        val_regimes = regimes[val_start:val_end]
        test_regimes = regimes[test_start:test_end]
        
        print("  Validation regime distribution:")
        val_counts = val_regimes.value_counts()
        for regime_id in sorted(regimes.unique()):
            count = val_counts.get(regime_id, 0)
            print(f"    Regime {regime_id}: {count} points ({count/len(val_regimes)*100:.1f}% of validation)")
        
        print("  Test regime distribution:")
        test_counts = test_regimes.value_counts()
        for regime_id in sorted(regimes.unique()):
            count = test_counts.get(regime_id, 0)
            print(f"    Regime {regime_id}: {count} points ({count/len(test_regimes)*100:.1f}% of test)")
    
    return adjusted_periods

def optimize_section_parameters_with_precomputed_regimes(train_df, val_df, test_start, config, section_index, previous_sections, global_regimes, global_volatility):
    """
    Optimize parameters for a walk-forward section using precomputed regimes.
    """
    print(f"\nOptimizing parameters for section {section_index+1} using precomputed regimes")
    
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
    
    # Get precomputed regimes for this section
    regimes = purged_train_df['precomputed_regime']
    
    # Ensure each regime has sufficient data
    regimes = ensure_sufficient_regime_data(purged_train_df, regimes, min_regime_data_points)
    
    # Print regime distribution
    n_regimes = len(regimes.unique())
    print(f"\nUsing {n_regimes} precomputed regimes in purged training data")
    regime_counts = regimes.value_counts()
    for regime_id, count in regime_counts.items():
        print(f"Regime {regime_id}: {count} data points ({count/len(regimes)*100:.2f}%)")
    
    # Optimize parameters for each regime separately using Optuna
    regime_best_params = {}
    regime_metrics = {}
    regime_studies = {}  # Store studies for later analysis
    
    for regime_id in sorted(regimes.unique()):
        # Create mask for this regime
        regime_mask = (regimes == regime_id)
        regime_count = regime_mask.sum()
        
        if regime_count < min_regime_data_points:
            print(f"\nInsufficient data for regime {regime_id}: {regime_count} points")
            print(f"Using default or prior parameters for regime {regime_id}")
            # Try to find parameters from previous sections
            found_prev_params = False
            if previous_sections:
                for prev_section in reversed(previous_sections):
                    if 'regime_params' in prev_section and int(regime_id) in prev_section['regime_params']:
                        regime_best_params[regime_id] = prev_section['regime_params'][int(regime_id)]
                        print(f"  Using parameters from previous section for regime {regime_id}")
                        found_prev_params = True
                        break
            
            if not found_prev_params:
                # Generate a default parameter set
                param_grid = generate_parameter_grid(strategy_config, 
                                                   strategy_config.get('cross_validation', {}).get('parameter_testing', {}))
                regime_best_params[regime_id] = param_grid[0]
            continue
        
        # Extract data for just this regime
        regime_df = purged_train_df.loc[regime_mask].copy()
        
        print(f"\nOptimizing parameters for Regime {regime_id} using {regime_count} data points")
        
        # Generate parameter bounds
        param_bounds = generate_parameter_bounds(strategy_config)
        
        # Determine number of trials based on available data points
        base_n_trials = config.get('STRATEGY_CONFIG', {}).get('cross_validation', {}).get('parameter_testing', {}).get('n_random_combinations', 100)
        
        # Scale trials based on data size
        data_size_factor = min(2.0, max(0.5, regime_count / 500))
        n_trials = int(base_n_trials * data_size_factor)
        n_trials = min(300, max(50, n_trials))  # Cap between 50 and 300 trials
        
        print(f"Running {n_trials} Optuna trials for regime {regime_id}")
        
        # Run Optuna optimization
        best_params, study = optimize_parameters_with_optuna(
            regime_df, 
            param_bounds, 
            strategy_config,
            n_trials=n_trials,
            regime_id=regime_id
        )
        
        # Store the results
        if best_params:
            # Store results and print metrics
            # (existing code)
            regime_best_params[regime_id] = best_params
            regime_studies[regime_id] = study
        else:
            # Use default parameters
            param_grid = generate_parameter_grid(
                strategy_config, 
                strategy_config.get('cross_validation', {}).get('parameter_testing', {})
            )
            regime_best_params[regime_id] = param_grid[0]
    
    # Ensure all regimes have parameters
    # Ensure all regimes have parameters
    for regime_id in sorted(regimes.unique()):
        if regime_id not in regime_best_params:
            print(f"Warning: No parameters for regime {regime_id}, using default parameters")
            param_grid = generate_parameter_grid(
                strategy_config, 
                strategy_config.get('cross_validation', {}).get('parameter_testing', {})
            )
            regime_best_params[regime_id] = param_grid[0]
    
    # Validate on validation data with precomputed regimes
    print("\nValidating regime-specific parameters on validation data")
    
    # Set the precomputed regime in validation data
    val_df_with_regime = val_df.copy()
    val_df_with_regime['precomputed_regime'] = global_regimes.loc[val_df.index]
    
    # Apply enhanced SMA strategy with precomputed regimes
    val_result_df = apply_enhanced_sma_strategy_with_precomputed_regimes(
        val_df_with_regime,
        regime_best_params,
        strategy_config
    )
    
    # Calculate metrics
    val_metrics = calculate_advanced_metrics(val_result_df['strategy_returns'], val_result_df['strategy_cumulative'])
    
    # Print validation results
    print(f"\nValidation Results:")
    print(f"Total Return: {val_metrics['total_return']:.4%}")
    print(f"Sharpe Ratio: {val_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {val_metrics['max_drawdown']:.4%}")
    
    # Calculate regime-specific performance in validation set
    validation_regime_metrics = {}
    for regime_id in sorted(regimes.unique()):
        regime_mask = (val_result_df['regime'] == regime_id)
        if regime_mask.any():
            regime_returns = val_result_df.loc[regime_mask, 'strategy_returns']
            if len(regime_returns) > 0:
                regime_cumulative = (1 + regime_returns).cumprod()
                regime_return = regime_cumulative.iloc[-1] - 1
                print(f"  Regime {regime_id} Validation Return: {regime_return:.4%}")
                
                # Calculate Sharpe ratio
                try:
                    from performance_metrics import calculate_sharpe_ratio
                    regime_sharpe = calculate_sharpe_ratio(regime_returns)
                    print(f"  Regime {regime_id} Validation Sharpe: {regime_sharpe:.4f}")
                    validation_regime_metrics[regime_id] = {
                        'return': regime_return,
                        'sharpe': regime_sharpe
                    }
                except Exception as e:
                    print(f"  Error calculating regime-specific metrics: {e}")
    
    # Store overall metrics for return
    for regime_id, params in regime_best_params.items():
        params['validation_score'] = val_metrics['sharpe_ratio']
        
        # Add validation metrics for this regime
        if regime_id in validation_regime_metrics:
            params['validation_regime_return'] = validation_regime_metrics[regime_id]['return']
            params['validation_regime_sharpe'] = validation_regime_metrics[regime_id]['sharpe']
    
    return regime_best_params, regime_studies

def apply_enhanced_sma_strategy_with_precomputed_regimes(df, regime_params, strategy_config):
    """
    Apply enhanced SMA strategy using precomputed regimes with improved error handling.
    
    Parameters:
        df (DataFrame): DataFrame with price data and precomputed_regime column
        regime_params (dict): Dictionary of parameters for each regime
        strategy_config (dict): Strategy configuration
        
    Returns:
        DataFrame: DataFrame with strategy results
    """
    import numpy as np
    
    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Get precomputed regimes from the dataframe
    regimes = result_df['precomputed_regime']
    
    # Determine which price column to use (handle different column names)
    price_column = 'close_price'  # Default to your database's column name
    high_column = 'high_price'
    low_column = 'low_price'
    
    # Check which regimes are actually present in the data vs in parameters
    present_regimes = sorted(regimes.unique())
    param_regimes = sorted([int(r) if isinstance(r, str) else r for r in regime_params.keys()])
    print(f"Regimes present in data: {present_regimes}")
    print(f"Regimes in parameters: {param_regimes}")
    
    # Check if the price column exists
    if price_column not in result_df.columns:
        # Try alternative names
        for alt_name in ['close', 'price', 'Close', 'ClosePrice']:
            if alt_name in result_df.columns:
                price_column = alt_name
                break
        else:
            raise ValueError(f"Missing required price column. Available columns: {result_df.columns.tolist()}")
    
    # Ensure 'returns' column exists with NaN handling
    if 'returns' not in result_df.columns:
        # Calculate returns based on close price
        result_df['returns'] = result_df[price_column].pct_change().fillna(0)
    
    # Initialize strategy columns
    result_df['position'] = 0.0
    result_df['signal'] = 0
    result_df['strategy_returns'] = 0.0
    result_df['regime'] = regimes
    result_df['trailing_stop_exits'] = 0
    result_df['profit_taking_exits'] = 0
    result_df['max_drawdown_exits'] = 0
    result_df['managed_position'] = 0.0
    result_df['filtered_signal'] = 0
    result_df['raw_signal'] = 0
    
    # Apply regime-specific strategy
    for regime_id in regime_params:
        # Convert regime_id to int if it's a string
        regime_id_int = int(regime_id) if isinstance(regime_id, str) else regime_id
        
        # Create mask for this regime
        regime_mask = (regimes == regime_id_int)
        regime_count = regime_mask.sum()
        
        if regime_count == 0:
            print(f"No data points found for regime {regime_id_int}")
            continue
        
        print(f"Processing regime {regime_id_int} with {regime_count} data points")
        
        # Extract data for just this regime
        regime_df = result_df.loc[regime_mask].copy()
        
        # Get parameters for this regime
        params = regime_params[regime_id]
        
        # Add back some preceding data for SMA calculation (lookback window)
        max_window = max(
            params.get('long_window', 55), 
            params.get('vol_lookback', 20) * 2
        )
        
        # Find preceding data safely without exact index matching
        regime_start_date = regime_df.index[0]
        
        # Get all data up to the start of this regime for lookback
        preceding_data = df[df.index < regime_start_date].iloc[-max_window:].copy() if max_window > 0 else pd.DataFrame()
        
        # Concatenate data with safe handling for empty dataframes
        if not preceding_data.empty:
            extended_df = pd.concat([preceding_data, regime_df])
        else:
            extended_df = regime_df
        
        # Calculate technical indicators using the specified parameters
        short_window = params.get('short_window', 21)
        long_window = params.get('long_window', 55)
        
        # Calculate SMAs using the correct price column with NaN handling
        extended_df['short_sma'] = extended_df[price_column].rolling(window=short_window).mean()
        extended_df['long_sma'] = extended_df[price_column].rolling(window=long_window).mean()
        
        # Calculate trend strength with NaN protection
        extended_df['trend_strength'] = (
            extended_df['short_sma'] / extended_df['long_sma'] - 1
        ).abs().fillna(0)
        
        # Calculate volatility with improved error handling
        vol_method = params.get('vol_method', 'parkinson')
        vol_lookback = params.get('vol_lookback', 20)
        
        try:
            from volatility import calculate_volatility
            volatility = calculate_volatility(
                extended_df,
                method=vol_method,
                window=vol_lookback
            )
            extended_df['volatility'] = volatility
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            # Fallback to simple volatility calculation
            extended_df['volatility'] = extended_df[price_column].pct_change().rolling(window=vol_lookback).std() * np.sqrt(252)
        
        # Ensure volatility has no NaN or zero values
        extended_df['volatility'] = extended_df['volatility'].fillna(
            extended_df['volatility'].median() if not extended_df['volatility'].isna().all() else 0.01
        )
        extended_df['volatility'] = extended_df['volatility'].replace(0, 0.01)  # Replace zeros with small positive value
        
        # Calculate signals based on SMA crossovers and trend strength
        trend_threshold = params.get('trend_strength_threshold', 0.01)
        
        # Calculate raw signals first (before applying risk management)
        extended_df['raw_signal'] = 0
        
        # Buy signal: short SMA crosses above long SMA with sufficient trend strength
        buy_mask = (
            extended_df['short_sma'].notna() & 
            extended_df['long_sma'].notna() & 
            (extended_df['short_sma'] > extended_df['long_sma']) &
            (extended_df['trend_strength'] > trend_threshold)
        )
        extended_df.loc[buy_mask, 'raw_signal'] = 1
        
        # Sell signal: short SMA crosses below long SMA or trend weakens
        sell_mask = (
            extended_df['short_sma'].notna() & 
            extended_df['long_sma'].notna() & 
            ((extended_df['short_sma'] < extended_df['long_sma']) |
            (extended_df['trend_strength'] < trend_threshold * 0.5))
        )
        extended_df.loc[sell_mask, 'raw_signal'] = -1
        
        # Copy raw signals to filtered signals (will be modified by risk management)
        extended_df['filtered_signal'] = extended_df['raw_signal'].copy()
        
        # Apply minimum holding period if specified
        min_holding = params.get('min_holding_period', 1)
        if min_holding > 1:
            # Track position entry time
            in_position = False
            entry_time = 0
            
            for i in range(len(extended_df)):
                if i >= len(extended_df.index):
                    break
                    
                curr_idx = extended_df.index[i]
                curr_signal = extended_df.loc[curr_idx, 'filtered_signal']
                
                if not in_position and curr_signal > 0:
                    # Enter position
                    in_position = True
                    entry_time = i
                    
                elif in_position:
                    # Check if we've held for minimum period
                    if i - entry_time < min_holding and curr_signal < 0:
                        # Override exit signal if we haven't held long enough
                        if i > 0 and i-1 < len(extended_df.index):
                            extended_df.loc[curr_idx, 'filtered_signal'] = extended_df.loc[extended_df.index[i-1], 'filtered_signal']
                    elif curr_signal <= 0:
                        # Exit position
                        in_position = False
        
        # Apply risk management parameters
        target_vol = params.get('target_vol', 0.2)
        max_position_size = params.get('max_position_size', 1.0)
        min_position_size = params.get('min_position_size', 0.0)
        
        # Position sizing based on volatility targeting
        extended_df['position_size'] = target_vol / extended_df['volatility'].clip(0.001)
        extended_df['position_size'] = extended_df['position_size'].clip(min_position_size, max_position_size)
        
        # Initialize managed position (to be modified by risk management)
        extended_df['managed_position'] = extended_df['filtered_signal'] * extended_df['position_size']
        
        # Apply trailing stops if enabled
        if params.get('trailing_stop_activation', 0) > 0:
            # Apply trailing stop logic
            trailing_activation = params.get('trailing_stop_activation', 0.05)
            trailing_distance = params.get('trailing_stop_distance', 0.02)
            
            # Initialize trailing stop variables
            in_position = False
            entry_price = 0
            trailing_level = 0
            
            for i in range(len(extended_df)):
                if i >= len(extended_df.index):
                    break
                    
                curr_idx = extended_df.index[i]
                curr_price = extended_df.loc[curr_idx, price_column]
                curr_position = extended_df.loc[curr_idx, 'managed_position']
                
                if not in_position and curr_position > 0:
                    # Enter position
                    in_position = True
                    entry_price = curr_price
                    trailing_level = curr_price * (1 - trailing_distance)
                    
                elif in_position:
                    # Update trailing stop if price increases
                    if curr_price > entry_price * (1 + trailing_activation):
                        new_stop = curr_price * (1 - trailing_distance)
                        trailing_level = max(trailing_level, new_stop)
                    
                    # Check if we hit the trailing stop
                    if curr_price < trailing_level:
                        # Exit position due to trailing stop
                        extended_df.loc[curr_idx, 'managed_position'] = 0
                        extended_df.loc[curr_idx, 'trailing_stop_exits'] = 1
                        in_position = False
                    elif curr_position == 0:
                        # Exit position due to signal
                        in_position = False
        
        # Apply stop loss if enabled
        if params.get('max_drawdown_exit', 0) > 0:
            max_drawdown = params.get('max_drawdown_exit', 0.15)
            
            # Track drawdown for each position
            in_position = False
            entry_price = 0
            
            for i in range(len(extended_df)):
                if i >= len(extended_df.index):
                    break
                    
                curr_idx = extended_df.index[i]
                curr_price = extended_df.loc[curr_idx, price_column]
                curr_position = extended_df.loc[curr_idx, 'managed_position']
                
                if not in_position and curr_position > 0:
                    # Enter position
                    in_position = True
                    entry_price = curr_price
                    
                elif in_position:
                    # Calculate current drawdown
                    drawdown = (curr_price / entry_price - 1) * -1
                    
                    # Exit if drawdown exceeds threshold
                    if drawdown > max_drawdown:
                        extended_df.loc[curr_idx, 'managed_position'] = 0
                        extended_df.loc[curr_idx, 'max_drawdown_exits'] = 1
                        in_position = False
                    elif curr_position == 0:
                        # Exit position
                        in_position = False
        
        # Apply take profit if enabled
        if params.get('profit_taking_threshold', 0) > 0:
            profit_threshold = params.get('profit_taking_threshold', 0.1)
            
            # Track profit for each position
            in_position = False
            entry_price = 0
            
            for i in range(len(extended_df)):
                if i >= len(extended_df.index):
                    break
                    
                curr_idx = extended_df.index[i]
                curr_price = extended_df.loc[curr_idx, price_column]
                curr_position = extended_df.loc[curr_idx, 'managed_position']
                
                if not in_position and curr_position > 0:
                    # Enter position
                    in_position = True
                    entry_price = curr_price
                    
                elif in_position:
                    # Calculate current profit
                    profit_pct = curr_price / entry_price - 1
                    
                    # Take profit if threshold is reached
                    if profit_pct > profit_threshold:
                        extended_df.loc[curr_idx, 'managed_position'] = 0
                        extended_df.loc[curr_idx, 'profit_taking_exits'] = 1
                        in_position = False
                    elif curr_position == 0:
                        # Exit position
                        in_position = False
        
        # Use managed position for final position
        extended_df['position'] = extended_df['managed_position']
        
        # Ensure no NaN values in key columns
        for col in ['position', 'signal', 'managed_position', 'filtered_signal', 'raw_signal']:
            if col in extended_df.columns:
                extended_df[col] = extended_df[col].fillna(0)
        
        # Use direct copying for regime indices to avoid missing data
        for idx in regime_df.index:
            if idx in extended_df.index:
                for col in ['signal', 'position', 'raw_signal', 'filtered_signal', 'managed_position', 
                           'trailing_stop_exits', 'profit_taking_exits', 'max_drawdown_exits', 'volatility']:
                    if col in extended_df.columns and col in result_df.columns:
                        result_df.loc[idx, col] = extended_df.loc[idx, col]
    
    # Calculate strategy returns safely
    result_df['position'] = result_df['position'].fillna(0)
    result_df['returns'] = result_df['returns'].fillna(0)
    result_df['strategy_returns'] = result_df['position'].shift(1).fillna(0) * result_df['returns']
    
    # Check for NaN values before calculating cumulative returns
    nan_count = result_df['strategy_returns'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in strategy returns. Filling with zeros.")
        result_df['strategy_returns'] = result_df['strategy_returns'].fillna(0)
    
    # Calculate cumulative returns safely
    if 'strategy_cumulative' not in result_df.columns:
        result_df['strategy_cumulative'] = (1 + result_df['strategy_returns']).cumprod()
    
    # Check for NaN values in cumulative returns
    nan_count = result_df['strategy_cumulative'].isna().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in strategy_cumulative. Setting to 1.0")
        result_df['strategy_cumulative'] = result_df['strategy_cumulative'].fillna(1.0)
    
    # Calculate buy & hold returns
    if 'buy_hold_cumulative' not in result_df.columns:
        result_df['buy_hold_cumulative'] = (1 + result_df['returns']).cumprod()
        # Check for NaN values in buy_hold_cumulative
        nan_count = result_df['buy_hold_cumulative'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in buy_hold_cumulative. Setting to 1.0")
            result_df['buy_hold_cumulative'] = result_df['buy_hold_cumulative'].fillna(1.0)
    
    # Print stats about the final result
    print("\nFinal data check:")
    print(f"Total rows: {len(result_df)}")
    print(f"NaN in strategy_returns: {result_df['strategy_returns'].isna().sum()}")
    print(f"NaN in strategy_cumulative: {result_df['strategy_cumulative'].isna().sum()}")
    print(f"NaN in position: {result_df['position'].isna().sum()}")
    print(f"Min strategy_cumulative: {result_df['strategy_cumulative'].min()}")
    print(f"Max strategy_cumulative: {result_df['strategy_cumulative'].max()}")
    
    # Calculate overall return for validation
    first_value = result_df['strategy_cumulative'].iloc[0]
    last_value = result_df['strategy_cumulative'].iloc[-1]
    overall_return = (last_value / first_value) - 1
    print(f"Overall calculated return: {overall_return:.4%}")
    
    return result_df

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
    Enhanced to store regime-specific performance metrics with improved NaN handling.
    
    Parameters:
        test_df (DataFrame): Test data
        section_params (dict): Parameters for the section
        config (dict): Strategy configuration
        
    Returns:
        dict: Performance metrics and results
    """
    import numpy as np
    
    # Apply strategy with optimized parameters
    result_df = apply_enhanced_sma_strategy_regime_specific(test_df, section_params, config.get('STRATEGY_CONFIG', {}))
    
    # Check for and fix NaN values in key columns
    for col in ['strategy_returns', 'returns']:
        if col in result_df.columns:
            nan_count = result_df[col].isna().sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} NaN values found in {col}. Filling with zeros.")
                result_df[col] = result_df[col].fillna(0)
    
    # Ensure cumulative returns are calculated correctly
    if 'strategy_returns' in result_df.columns:
        # Recalculate cumulative returns to ensure no NaN propagation
        result_df['strategy_cumulative'] = (1 + result_df['strategy_returns']).cumprod()
        result_df['buy_hold_cumulative'] = (1 + result_df['returns']).cumprod()
    
    # Calculate overall performance metrics
    metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
    
    # Double-check total return calculation
    if np.isnan(metrics.get('total_return', np.nan)):
        initial_value = result_df['strategy_cumulative'].iloc[0]
        final_value = result_df['strategy_cumulative'].iloc[-1]
        
        if not np.isnan(initial_value) and not np.isnan(final_value) and initial_value > 0:
            total_return = (final_value / initial_value) - 1
            print(f"Fixing NaN total return with calculated value: {total_return:.4%}")
            metrics['total_return'] = total_return
        else:
            print(f"Cannot calculate total return. Initial: {initial_value}, Final: {final_value}")
            metrics['total_return'] = 0.0
    
    # Calculate buy & hold metrics with NaN protection
    if np.isnan(result_df['buy_hold_cumulative'].iloc[-1]):
        print("Warning: NaN in buy_hold_cumulative. Recalculating.")
        result_df['buy_hold_cumulative'] = (1 + result_df['returns'].fillna(0)).cumprod()
    
    buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
    
    # Calculate regime-specific performance
    regimes = result_df['regime']
    regime_metrics = {}
    
    for regime_id in sorted(section_params.keys()):
        regime_id_int = int(regime_id) if isinstance(regime_id, str) else regime_id
        regime_mask = (regimes == regime_id_int)
        
        if regime_mask.any():
            regime_returns = result_df.loc[regime_mask, 'strategy_returns'].fillna(0)
            
            if len(regime_returns) > 0:
                # Calculate metrics for this regime with NaN handling
                regime_cumulative = (1 + regime_returns).cumprod()
                
                try:
                    regime_specific_metrics = calculate_advanced_metrics(regime_returns, regime_cumulative)
                    
                    # Verify total return calculation for regime
                    if np.isnan(regime_specific_metrics.get('total_return', np.nan)):
                        if len(regime_cumulative) > 0:
                            regime_return = regime_cumulative.iloc[-1] - 1
                            regime_specific_metrics['total_return'] = regime_return
                        else:
                            regime_specific_metrics['total_return'] = 0.0
                    
                    regime_metrics[regime_id_int] = regime_specific_metrics
                except Exception as e:
                    print(f"Error calculating metrics for regime {regime_id_int}: {e}")
                    # Create basic metrics if calculation fails
                    if len(regime_cumulative) > 0:
                        total_return = regime_cumulative.iloc[-1] - 1 if not np.isnan(regime_cumulative.iloc[-1]) else 0.0
                    else:
                        total_return = 0.0
                        
                    regime_metrics[regime_id_int] = {
                        'total_return': total_return,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    }
    
    # Print summary
    print(f"\nSection Test Results:")
    print(f"Total Return: {metrics.get('total_return', 0):.4%}")
    print(f"Annualized Return: {metrics.get('annualized_return', 0):.4%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4%}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.4%}")
    print(f"Buy & Hold Return: {buy_hold_return:.4%}")
    print(f"Outperformance: {metrics.get('total_return', 0) - buy_hold_return:.4%}")
    
    # Print regime-specific results
    print("\nRegime-Specific Results:")
    for regime_id, regime_metric in regime_metrics.items():
        print(f"Regime {regime_id}:")
        print(f"  Return: {regime_metric.get('total_return', 0):.4%}")
        print(f"  Sharpe: {regime_metric.get('sharpe_ratio', 0):.4f}")
        print(f"  Max DD: {regime_metric.get('max_drawdown', 0):.4%}")
    
    return {
        'metrics': metrics,
        'regime_params': section_params,
        'regime_metrics': regime_metrics,
        'buy_hold_return': buy_hold_return,
        'outperformance': metrics.get('total_return', 0) - buy_hold_return
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