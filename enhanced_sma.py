# enhanced_sma.py
# Main entry point for the Enhanced SMA strategy

import os
import sys
import time
from datetime import datetime, timedelta

# Import our modules
from database import DatabaseHandler
from volatility import calculate_volatility
from regime_detection import detect_volatility_regimes
from signals import calculate_trend_strength, calculate_momentum, filter_signals, apply_min_holding_period
from risk_management import (
    calculate_adaptive_position_size, calculate_adaptive_position_size_with_schedule,
    apply_unified_risk_management, calculate_trade_returns
)
from performance_metrics import (
    calculate_advanced_metrics, calculate_sharpe_ratio, calculate_sortino_ratio, 
    calculate_max_drawdown, calculate_calmar_ratio
)
from parameter_optimization import (
    generate_time_series_cv_splits, optimize_parameters_with_cv, 
    create_full_parameter_set, test_parameter_combination,
    generate_parameter_grid, ensure_parameters,
    # New regime-specific functions:
    test_parameters_by_regime, calculate_strategy_returns,
    optimize_parameters_by_regime
)
from visualization import (
    # New regime-specific functions:
    plot_enhanced_results_regime_specific, save_enhanced_results_regime_specific
)
# Try to import configuration
try:
    from enhanced_config import (
        TRADING_FREQUENCY,
        TRAINING_START,
        TRAINING_END,
        TESTING_START,
        TESTING_END,
        CURRENCY,
        INITIAL_CAPITAL,
        TRADING_FEE_PCT,
        RESULTS_DIR,
        SAVE_RESULTS,
        PLOT_RESULTS,
        STRATEGY_CONFIG
    )
    print("Successfully imported configuration from enhanced_config.py")
except ImportError as e:
    print(f"Error importing enhanced_config: {e}")
    try:
        # Try to import from config.py as fallback
        from config import (
            TRADING_FREQUENCY,
            TRAINING_START,
            TRAINING_END,
            TESTING_START,
            TESTING_END,
            CURRENCY,
            INITIAL_CAPITAL,
            TRADING_FEE_PCT,
            RESULTS_DIR,
            SAVE_RESULTS,
            PLOT_RESULTS,
            STRATEGY_CONFIG
        )
        print("Successfully imported configuration from config.py")
    except ImportError:
        print("Using default configuration values")
        
        # Default configuration if import fails
        TRADING_FREQUENCY = "1H"
        TRAINING_START = "2018-05-20"
        TRAINING_END = "2020-12-31"
        TESTING_START = "2021-01-01"
        TESTING_END = "2024-10-20"
        CURRENCY = "XRP/USD"
        INITIAL_CAPITAL = 10000
        TRADING_FEE_PCT = 0.001
        RESULTS_DIR = "enhanced_sma_results"
        SAVE_RESULTS = True
        PLOT_RESULTS = True

        # Default strategy config
        STRATEGY_CONFIG = {
            # Volatility calculation settings
            'volatility': {
                'methods': ['parkinson', 'garch', 'yang_zhang'],  # Different volatility calculation methods
                'lookback_periods': [20, 50, 100],  # Different lookback periods for volatility
                'regime_smoothing': 5,  # Days to smooth regime transitions
                'min_history_multiplier': 5,  # Minimum history required as multiplier of lookback
            },
            
            # Regime detection settings
            'regime_detection': {
                'method': 'kmeans',  # Options: 'kmeans', 'kde', 'quantile'
                'n_regimes': 3,  # Number of distinct volatility regimes
                'quantile_thresholds': [0.33, 0.67],  # Percentile thresholds for regime transitions
                'regime_stability_period': 48,  # Hours required before confirming regime change
            },
            
            # SMA strategy settings
            'sma': {
                'short_windows': [5, 8, 13, 21, 34],  # Fibonacci-based short windows
                'long_windows': [21, 34, 55, 89, 144],  # Fibonacci-based long windows
                'min_holding_period': 24,  # Minimum holding period in hours
                'trend_filter_period': 200,  # Period for trend strength calculation
                'trend_strength_threshold': 0.3,  # Minimum trend strength to take a position
            },
            
            # Risk management settings
            'risk_management': {
                'target_volatility': 0.15,  # Target annualized volatility
                'max_position_size': 1.0,  # Maximum position size
                'min_position_size': 0.1,  # Minimum position size
                'max_drawdown_exit': 0.15,  # Exit if drawdown exceeds this threshold
                'profit_taking_threshold': 0.05,  # Take profit at this threshold
                'trailing_stop_activation': 0.03,  # Activate trailing stop after this gain
                'trailing_stop_distance': 0.02,  # Trailing stop distance
            },
            
            # Cross-validation settings
            'cross_validation': {
                'n_splits': 5,  # Number of time series cross-validation splits
                'min_train_size': 90,  # Minimum training size in days
                'step_forward': 30,  # Step forward size in days for expanding window
                'validation_ratio': 0.3,  # Portion of training data to use for validation
            },
            
            # Parameter selection settings
            'parameter_selection': {
                'stability_weight': 0.5,  # Weight for parameter stability vs. performance
                'sharpe_weight': 0.4,  # Weight for Sharpe ratio in fitness function
                'sortino_weight': 0.3,  # Weight for Sortino ratio in fitness function
                'calmar_weight': 0.3,  # Weight for Calmar ratio in fitness function
            }
        }

def apply_enhanced_sma_strategy_regime_specific(df, regime_params, config):
    """
    Apply enhanced SMA strategy with regime-specific optimized parameters.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        regime_params (dict): Optimized parameters for each regime
        config (dict): Strategy configuration
        
    Returns:
        DataFrame: Results DataFrame
    """
    # Import pandas here to ensure it's available in this function's scope
    import pandas as pd
    import numpy as np
    
    print("Applying enhanced SMA strategy with regime-specific parameters...")
    
    # First, calculate volatility and detect regimes
    # Use parameters from regime 0 as default for regime detection
    base_params = regime_params.get(0, next(iter(regime_params.values())))
    vol_method = base_params.get('vol_method', 'parkinson')
    vol_lookback = base_params.get('vol_lookback', 20)
    
    # Extract additional parameters with defaults for regime detection
    regime_method = base_params.get('regime_method', config.get('regime_detection', {}).get('method', 'kmeans'))
    n_regimes = base_params.get('n_regimes', config.get('regime_detection', {}).get('n_regimes', 3))
    regime_stability = base_params.get('regime_stability', 
                                      config.get('regime_detection', {}).get('regime_stability_period', 48))
    regime_smoothing = base_params.get('regime_smoothing', config.get('volatility', {}).get('regime_smoothing', 5))
    
    # Calculate volatility
    volatility = calculate_volatility(df, method=vol_method, window=vol_lookback)
    
    # Detect regimes
    regimes = detect_volatility_regimes(
        df, 
        volatility, 
        method=regime_method,
        n_regimes=n_regimes,
        smoothing_period=regime_smoothing,
        stability_period=regime_stability
    )
    
    # Print regime distribution for debugging
    regime_counts = regimes.value_counts()
    total_periods = len(df)
    print("\nREGIME DISTRIBUTION:")
    for regime, count in regime_counts.items():
        percentage = (count / total_periods) * 100
        print(f"Regime {regime}: {count} periods ({percentage:.2f}%)")
    
    # Initialize result series
    raw_signal = pd.Series(0, index=df.index)
    filtered_signal = pd.Series(0, index=df.index)
    position = pd.Series(0, index=df.index)
    sized_position = pd.Series(0, index=df.index)
    position_size = pd.Series(0.0, index=df.index)
    buy_hold_mask = pd.Series(False, index=df.index)
    
    # Get regime opt-out and buy & hold settings
    regime_opt_out = config['regime_detection'].get('regime_opt_out', None)
    regime_buy_hold = config['regime_detection'].get('regime_buy_hold', None)
    
    # Pre-calculate MAs for all window sizes
    ma_cache = {}
    window_sizes = set()
    
    # Collect all window sizes needed across regimes
    for regime_id, params in regime_params.items():
        short_window = params.get('short_window', 13)
        long_window = params.get('long_window', 55)
        window_sizes.add(short_window)
        window_sizes.add(long_window)
    
    # Calculate all required MAs
    for window in window_sizes:
        ma_cache[window] = df['close_price'].rolling(window=window).mean()
    
    # Initialize series to store MA values
    short_ma = pd.Series(0.0, index=df.index)
    long_ma = pd.Series(0.0, index=df.index)
    
    # Process each regime separately
    for regime_id, params in regime_params.items():
        # Create mask for this regime
        regime_mask = (regimes == regime_id)
        if not regime_mask.any():
            print(f"No data for regime {regime_id}, skipping")
            continue
            
        print(f"\nProcessing regime {regime_id} ({regime_mask.sum()} data points)...")
        
        # Extract regime-specific parameters
        short_window = params.get('short_window', 13)
        long_window = params.get('long_window', 55)
        trend_strength_threshold = params.get('trend_strength_threshold', 0.3)
        min_holding_period = params.get('min_holding_period', 24)
        trend_filter_period = params.get('trend_filter_period', 200)
        
        # Risk management parameters
        target_vol = params.get('target_vol', 0.15)
        max_position_size = params.get('max_position_size', 1.0)
        min_position_size = params.get('min_position_size', 0.1)
        max_drawdown_exit = params.get('max_drawdown_exit', 0.15)
        profit_taking_threshold = params.get('profit_taking_threshold', 0.05)
        trailing_stop_activation = params.get('trailing_stop_activation', 0.03)
        trailing_stop_distance = params.get('trailing_stop_distance', 0.02)
        
        # Print regime-specific parameters
        print(f"  Regime {regime_id} parameters:")
        print(f"  SMA: Short={short_window}, Long={long_window}, Min Hold={min_holding_period}")
        print(f"  Trend Threshold: {trend_strength_threshold}, Filter Period: {trend_filter_period}")
        print(f"  Risk: Target Vol={target_vol}, Max DD Exit={max_drawdown_exit}")
        
        # Get MAs from cache for this regime
        regime_short_ma = ma_cache[short_window]
        regime_long_ma = ma_cache[long_window]
        
        # Store regime-specific MAs
        short_ma[regime_mask] = regime_short_ma[regime_mask]
        long_ma[regime_mask] = regime_long_ma[regime_mask]
        
        # Calculate trend strength and momentum for this regime
        # We calculate these globally rather than per-regime to avoid edge effects
        trend_strength = calculate_trend_strength(df, window=trend_filter_period)
        momentum = calculate_momentum(df, window=short_window)
        
        # Generate raw signal for this regime
        regime_raw_signal = pd.Series(0, index=df.index)
        regime_raw_signal[regime_short_ma > regime_long_ma] = 1
        regime_raw_signal[regime_short_ma < regime_long_ma] = -1
        
        # Apply signal only to this regime's data
        raw_signal[regime_mask] = regime_raw_signal[regime_mask]
        
        # Filter signals for this regime
        regime_filtered_signal = filter_signals(
            regime_raw_signal, 
            trend_strength, 
            momentum,
            min_trend_strength=trend_strength_threshold
        )
        
        # Apply filtered signal to this regime's data
        filtered_signal[regime_mask] = regime_filtered_signal[regime_mask]
        
        # Apply minimum holding period to this regime
        regime_position = apply_min_holding_period(
            regime_filtered_signal,
            min_holding_hours=min_holding_period
        )
        
        # Apply position to this regime's data
        position[regime_mask] = regime_position[regime_mask]
        
        # Calculate position size for this regime
        regime_position_size = calculate_adaptive_position_size(
            volatility,
            target_vol=target_vol,
            max_size=max_position_size,
            min_size=min_position_size
        )
        
        # Apply position size to this regime's data
        position_size[regime_mask] = regime_position_size[regime_mask]
        
        # Check if we should opt out of this regime
        if regime_opt_out is not None and regime_opt_out.get(regime_id, False):
            # Zero out positions for this regime
            position[regime_mask] = 0
            position_size[regime_mask] = 0
            print(f"  Opted out of trading in regime {regime_id}")
        
        # Check if we should use buy & hold for this regime
        if regime_buy_hold is not None and regime_buy_hold.get(regime_id, False):
            # Apply buy & hold mask
            buy_hold_mask[regime_mask] = True
            # Override position size for buy & hold
            position_size[regime_mask] = max_position_size
            # Set position to long for buy & hold
            position[regime_mask] = 1
            print(f"  Using buy & hold for regime {regime_id}")
    
    # Apply position sizing
    sized_position = position * position_size
    
    # Apply buy & hold strategy where indicated
    if buy_hold_mask.any():
        print(f"\nBuy & Hold strategy applied to {buy_hold_mask.sum()} periods ({buy_hold_mask.sum()/len(df)*100:.2f}%)")
        # Long position with max size for buy & hold periods
        sized_position[buy_hold_mask] = 1 * max_position_size
    
    # Calculate returns
    returns = df['close_price'].pct_change().fillna(0)
    
    # Calculate trade returns
    trade_returns = calculate_trade_returns(sized_position, returns)
    
    # Apply unified risk management - don't apply to buy & hold periods
    managed_position = sized_position.copy()
    non_buy_hold_mask = ~buy_hold_mask
    
    # Define exit statistics
    exit_stats = {
        'max_drawdown_exits': 0,
        'profit_taking_exits': 0,
        'trailing_stop_exits': 0,
        'total_trades': 0
    }
    
    # Apply risk management to each regime separately
    for regime_id, params in regime_params.items():
        # Create regime mask excluding buy & hold periods
        regime_mask = (regimes == regime_id) & non_buy_hold_mask
        if not regime_mask.any():
            continue
            
        # Create risk management config for this regime
        risk_config = {
            'max_drawdown_exit': params.get('max_drawdown_exit', 0.15),
            'profit_taking_threshold': params.get('profit_taking_threshold', 0.05),
            'trailing_stop_activation': params.get('trailing_stop_activation', 0.03),
            'trailing_stop_distance': params.get('trailing_stop_distance', 0.02)
        }
        
        # Extract data for this regime
        regime_df = df.loc[regime_mask].copy()
        regime_position = sized_position[regime_mask].copy()
        regime_returns = returns[regime_mask].copy()
        regime_volatility = volatility[regime_mask].copy()
        regime_regimes = regimes[regime_mask].copy()
        
        # Apply risk management to this regime
        regime_managed_position, regime_exit_stats = apply_unified_risk_management(
            regime_df,
            regime_position,
            regime_returns,
            regime_volatility,
            regime_regimes,
            risk_config
        )
        
        # Update managed position
        managed_position.loc[regime_mask] = regime_managed_position.values
        
        # Update exit statistics
        exit_stats['max_drawdown_exits'] += regime_exit_stats['max_drawdown_exits']
        exit_stats['profit_taking_exits'] += regime_exit_stats['profit_taking_exits']
        exit_stats['trailing_stop_exits'] += regime_exit_stats['trailing_stop_exits']
        exit_stats['total_trades'] += regime_exit_stats['total_trades']
    
    # Calculate position changes (when a trade occurs)
    position_changes = managed_position.diff().fillna(0).abs()
    num_trades = int((position_changes != 0).sum())
    
    # Calculate strategy returns
    strategy_returns = managed_position.shift(1).fillna(0) * returns
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    
    # Calculate buy and hold returns
    buy_hold_returns = returns
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'close_price': df['close_price'],
        'volatility': volatility,
        'regime': regimes,
        'trend_strength': trend_strength,
        'momentum': momentum,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'raw_signal': raw_signal,
        'filtered_signal': filtered_signal,
        'position': position,
        'position_size': position_size,
        'sized_position': sized_position,
        'managed_position': managed_position,
        'returns': returns,
        'strategy_returns': strategy_returns,
        'strategy_cumulative': strategy_cumulative,
        'buy_hold_cumulative': buy_hold_cumulative,
        'trade_returns': trade_returns,
        'buy_hold_mask': buy_hold_mask
    })
    
    # Add exit statistics to result_df for analysis
    result_df['max_drawdown_exits'] = exit_stats['max_drawdown_exits']
    result_df['profit_taking_exits'] = exit_stats['profit_taking_exits']
    result_df['trailing_stop_exits'] = exit_stats['trailing_stop_exits']
    
    # Print summary of exit types
    print(f"\nRisk Management Exit Summary:")
    print(f"Max Drawdown Exits: {exit_stats['max_drawdown_exits']}")
    print(f"Profit Taking Exits: {exit_stats['profit_taking_exits']}")
    print(f"Trailing Stop Exits: {exit_stats['trailing_stop_exits']}")
    print(f"Total Trades: {exit_stats['total_trades']}")
    
    # Print buy & hold summary
    if buy_hold_mask.any():
        print(f"\nBuy & Hold Summary:")
        print(f"Total periods with buy & hold: {buy_hold_mask.sum()} ({buy_hold_mask.mean()*100:.2f}%)")
        
        # Calculate performance during buy & hold periods
        bh_returns = returns[buy_hold_mask]
        bh_cumulative = (1 + bh_returns).cumprod().iloc[-1] if len(bh_returns) > 0 else 1
        print(f"Buy & hold performance: {bh_cumulative - 1:.4%}")
    
    print(f"\nStrategy applied with {num_trades} trades")
    
    # Calculate performance by regime
    print("\nPerformance by Regime:")
    for regime_id in range(n_regimes):
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            regime_strategy_returns = strategy_returns[regime_mask]
            regime_market_returns = returns[regime_mask]
            
            if len(regime_strategy_returns) > 0:
                regime_return = (1 + regime_strategy_returns).prod() - 1
                regime_market_return = (1 + regime_market_returns).prod() - 1
                
                print(f"Regime {regime_id}:")
                print(f"  Strategy Return: {regime_return:.4%}")
                print(f"  Market Return: {regime_market_return:.4%}")
                print(f"  Alpha: {regime_return - regime_market_return:.4%}")
    
    return result_df

def run_enhanced_backtest_regime_specific():
    """
    Run enhanced backtest of SMA strategy with regime-specific parameter optimization.
    """
    print(f"Starting enhanced SMA backtest with regime-specific optimization for {CURRENCY}")
    start_time = time.time()
    
    try:
        # Initialize database connection
        db = DatabaseHandler()
        
        # Fetch complete data for the entire period
        print(f"Fetching data from {TRAINING_START} to {TESTING_END}")
        df = db.get_historical_data(CURRENCY, TRAINING_START, TESTING_END)
        if len(df) < 1000:
            print(f"Insufficient data for {CURRENCY} ({len(df)} data points). Exiting.")
            db.close()
            return None, None, None
        
        # Generate cross-validation splits for parameter optimization
        cv_config = STRATEGY_CONFIG['cross_validation']
        splits = generate_time_series_cv_splits(
            TRAINING_START, 
            TRAINING_END,
            n_splits=cv_config['n_splits'],
            min_train_size=cv_config['min_train_size'],
            step_forward=cv_config['step_forward']
        )
        
        # Optimize parameters using cross-validation - regime specific
        regime_params, cv_results, regime_distribution = optimize_parameters_by_regime(df, STRATEGY_CONFIG, splits)

        if not regime_params:
            print("Regime-specific parameter optimization failed. Exiting.")
            db.close()
            return None, None, None

        # Fetch test data
        print("Fetching test data...")
        try:
            test_df = df.loc[TESTING_START:TESTING_END].copy()
            print(f"Test data fetched: {len(test_df)} data points")
        except Exception as e:
            print(f"Error fetching test data: {e}")
            import traceback
            traceback.print_exc()
            db.close()
            return None, None, None

        # Apply enhanced strategy with regime-specific parameters
        print("Applying enhanced strategy with regime-specific parameters...")
        try:
            result_df = apply_enhanced_sma_strategy_regime_specific(test_df, regime_params, STRATEGY_CONFIG)
            print("Strategy applied successfully")
        except Exception as e:
            print(f"Error applying strategy: {e}")
            import traceback
            traceback.print_exc()
            db.close()
            return None, None, None
            
        # Calculate performance metrics
        metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
        
        # Calculate buy & hold metrics
        buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
        buy_hold_metrics = calculate_advanced_metrics(result_df['returns'], result_df['buy_hold_cumulative'])
        
        # Close database connection
        db.close()
        
        # Print test results
        print("\n===== Test Results =====")
        print(f"Total Return: {metrics['total_return']:.4%}")
        print(f"Annualized Return: {metrics['annualized_return']:.4%}")
        print(f"Volatility: {metrics['volatility']:.4%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"Win Rate: {metrics['win_rate']:.4%}")
        print(f"Gain-to-Pain Ratio: {metrics['gain_to_pain']:.4f}")
        print(f"Buy & Hold Return: {buy_hold_return:.4%}")
        print(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}")
        
        # Calculate performance by regime
        regimes = result_df['regime']
        strategy_returns = result_df['strategy_returns']
        market_returns = result_df['returns']
        
        print("\n===== Regime Performance =====")
        for regime_id in sorted(regime_params.keys()):
            regime_mask = (regimes == regime_id)
            if regime_mask.any():
                regime_strategy_returns = strategy_returns[regime_mask]
                regime_market_returns = market_returns[regime_mask]
                
                if len(regime_strategy_returns) > 0:
                    regime_return = (1 + regime_strategy_returns).prod() - 1
                    regime_market_return = (1 + regime_market_returns).prod() - 1
                    regime_alpha = regime_return - regime_market_return
                    
                    # Calculate Sharpe and Sortino ratios
                    regime_sharpe = calculate_sharpe_ratio(regime_strategy_returns)
                    regime_sortino = calculate_sortino_ratio(regime_strategy_returns)
                    
                    # Get parameter set used for this regime
                    regime_param_set = regime_params[regime_id]
                    short_window = regime_param_set.get('short_window', 13)
                    long_window = regime_param_set.get('long_window', 55)
                    
                    print(f"Regime {regime_id} (Using SMA {short_window}/{long_window}):")
                    print(f"  Data points: {regime_mask.sum()} ({regime_mask.mean()*100:.2f}%)")
                    print(f"  Strategy Return: {regime_return:.4%}")
                    print(f"  Market Return: {regime_market_return:.4%}")
                    print(f"  Alpha: {regime_alpha:.4%}")
                    print(f"  Sharpe: {regime_sharpe:.4f}")
                    print(f"  Sortino: {regime_sortino:.4f}")
        
        # Plot results - we need to modify the plotting function to show regime-specific parameters
        if PLOT_RESULTS:
            # Convert regime_params to a format plot_enhanced_results can use
            combined_params = {
                'regime_specific': True,  # Flag to indicate we're using regime-specific parameters
                'regime_params': regime_params
            }
            plot_enhanced_results_regime_specific(result_df, combined_params, metrics)
        
        # Save results
        if SAVE_RESULTS:
            save_enhanced_results_regime_specific(result_df, regime_params, metrics, cv_results)
        
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
        
        return result_df, regime_params, metrics
    
    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main entry point for the enhanced SMA strategy with regime-specific optimization."""
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Run enhanced backtest with regime-specific optimization
    print("Running enhanced SMA backtest with regime-specific parameter optimization...")
    run_enhanced_backtest_regime_specific()
    
if __name__ == "__main__":
    main()