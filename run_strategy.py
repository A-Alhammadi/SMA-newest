#!/usr/bin/env python
# run_strategy.py - Script to run the enhanced SMA strategy

import os
import sys
import time
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import our modules - only import what we actually use
# Try to import our modules - only import what we actually use
try:
    from purged_walk_forward import run_purged_walk_forward_optimization
    from visualization import plot_enhanced_results_with_regimes, plot_enhanced_results_regime_specific, analyze_regime_distributions
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure purged_walk_forward.py and all its dependencies are in the current directory.")
    sys.exit(1)

# Set up logging to file
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"strategy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Redirect stdout and stderr to log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

# Print start information
print("=" * 80)
print(f"Enhanced SMA Strategy with Purged Walk-Forward Optimization")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Logging to: {log_file}")
print("=" * 80)
print()

# Add these new functions to run_strategy.py:

def extract_best_regime_parameters(section_results):
    """Extract the best parameter set for each regime across all sections."""
    # Collect parameter sets and performance metrics for each regime
    regime_performances = {}
    
    for section in section_results:
        if 'regime_params' in section:
            for regime_id, params in section['regime_params'].items():
                if regime_id not in regime_performances:
                    regime_performances[regime_id] = []
                
                # Get relevant performance metric (prefer validation metrics if available)
                sharpe_ratio = 0.0
                
                # Check if validation metrics are stored in params
                if 'validation_regime_sharpe' in params:
                    sharpe_ratio = params['validation_regime_sharpe']
                elif 'validation_score' in params:
                    sharpe_ratio = params['validation_score']
                # If no validation metrics, try to use test metrics if available
                elif 'regime_metrics' in section and regime_id in section['regime_metrics']:
                    sharpe_ratio = section['regime_metrics'][regime_id].get('sharpe_ratio', 0)
                # Last resort: use overall section metrics
                elif 'metrics' in section:
                    sharpe_ratio = section['metrics'].get('sharpe_ratio', 0)
                
                # Store parameters and performance
                regime_performances[regime_id].append({
                    'params': params.copy(),  # Make a copy to avoid modifying original
                    'sharpe_ratio': sharpe_ratio,
                    'section': section.get('section_index', 0)
                })
    
    # Select best parameter set for each regime based on Sharpe ratio
    best_params = {}
    for regime_id, performances in regime_performances.items():
        if performances:
            # Sort by Sharpe ratio in descending order
            sorted_performances = sorted(
                performances, 
                key=lambda x: x['sharpe_ratio'], 
                reverse=True
            )
            
            # Take the parameters from the best performing section
            best_params[regime_id] = sorted_performances[0]['params'].copy()
            
            # Print info about best parameters for this regime
            print(f"Best parameters for regime {regime_id} from section {sorted_performances[0]['section']} "
                  f"with Sharpe ratio {sorted_performances[0]['sharpe_ratio']:.4f}")
            
            # Clean up any non-parameter fields that might be in the params dict
            for key in ['validation_regime_sharpe', 'validation_score', 'validation_regime_return']:
                if key in best_params[regime_id]:
                    del best_params[regime_id][key]
    
    return best_params

def evaluate_strategy_with_best_params(df, best_params, config):
    """Evaluate the strategy on the entire dataset using the best parameters."""
    # Apply strategy with best parameters
    from enhanced_sma import apply_enhanced_sma_strategy_regime_specific
    from performance_metrics import calculate_advanced_metrics
    
    # Print parameter summary
    print("\nBest parameters summary:")
    for regime_id, params in best_params.items():
        print(f"Regime {regime_id}:")
        for key in ['short_window', 'long_window', 'trend_strength_threshold']:
            if key in params:
                print(f"  {key}: {params[key]}")
    
    # Apply strategy with best parameters
    result_df = apply_enhanced_sma_strategy_regime_specific(df, best_params, config['STRATEGY_CONFIG'])
    
    # Calculate performance metrics
    metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
    
    return result_df, metrics

def evaluate_strategy_with_precomputed_regimes(df, best_params, config):
    """Evaluate the strategy on the entire dataset using the best parameters and precomputed regimes."""
    # Apply strategy with best parameters and precomputed regimes
    from purged_walk_forward import apply_enhanced_sma_strategy_with_precomputed_regimes
    from performance_metrics import calculate_advanced_metrics
    
    # Print parameter summary
    print("\nBest parameters summary for precomputed regimes:")
    for regime_id, params in best_params.items():
        print(f"Regime {regime_id}:")
        for key in ['short_window', 'long_window', 'trend_strength_threshold']:
            if key in params:
                print(f"  {key}: {params[key]}")
    
    # Apply strategy with best parameters and precomputed regimes
    result_df = apply_enhanced_sma_strategy_with_precomputed_regimes(df, best_params, config['STRATEGY_CONFIG'])
    
    # Calculate performance metrics
    metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
    
    return result_df, metrics

def save_best_regime_results(best_params, result_df, metrics, config):
    """Save best regime parameters and performance to a file."""
    from datetime import datetime
    import os
    
    # Get results directory
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    currency = config.get('CURRENCY', 'BTC/USD')
    
    # Create directory if needed
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_params_file = os.path.join(
        results_dir, 
        f'best_regime_parameters_{currency.replace("/", "_")}_{timestamp}.txt'
    )
    
    with open(best_params_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"BEST REGIME-SPECIFIC PARAMETERS AND PERFORMANCE\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write best parameters
        f.write("BEST PARAMETERS FOR EACH REGIME\n")
        f.write("-" * 50 + "\n\n")
        
        for regime_id, params in best_params.items():
            f.write(f"Regime {regime_id} Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Write overall performance
        f.write("PERFORMANCE USING BEST PARAMETERS\n")
        f.write("-" * 50 + "\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['total_return','annualized_return','volatility','max_drawdown','win_rate']:
                    f.write(f"{key}: {value:.4%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Buy & hold comparison
        buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
        f.write(f"\nBuy & Hold Return: {buy_hold_return:.4%}\n")
        f.write(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}\n")
        
        # Performance by regime
        f.write("\nPERFORMANCE BY REGIME\n")
        f.write("-" * 50 + "\n\n")
        
        regimes = result_df['regime']
        strategy_returns = result_df['strategy_returns']
        market_returns = result_df['returns']
        
        for regime_id in sorted(best_params.keys()):
            regime_mask = (regimes == regime_id)
            regime_count = regime_mask.sum()
            regime_pct = regime_count / len(result_df) * 100
            
            f.write(f"Regime {regime_id} ({regime_count} periods, {regime_pct:.2f}%):\n")
            
            if regime_mask.any():
                regime_strategy_returns = strategy_returns[regime_mask]
                regime_market_returns = market_returns[regime_mask]
                
                if len(regime_strategy_returns) > 0:
                    # Calculate returns
                    regime_strategy_cumulative = (1 + regime_strategy_returns).cumprod()
                    regime_market_cumulative = (1 + regime_market_returns).cumprod()
                    
                    regime_return = regime_strategy_cumulative.iloc[-1] - 1 if len(regime_strategy_cumulative) > 0 else 0
                    regime_market_return = regime_market_cumulative.iloc[-1] - 1 if len(regime_market_cumulative) > 0 else 0
                    
                    # Calculate Sharpe
                    from performance_metrics import calculate_sharpe_ratio
                    regime_sharpe = calculate_sharpe_ratio(regime_strategy_returns)
                    
                    f.write(f"  Return: {regime_return:.4%}\n")
                    f.write(f"  Market Return: {regime_market_return:.4%}\n")
                    f.write(f"  Alpha: {regime_return - regime_market_return:.4%}\n")
                    f.write(f"  Sharpe Ratio: {regime_sharpe:.4f}\n")
                    
                    # Parameters used
                    params = best_params.get(regime_id, {})
                    f.write(f"  Parameters: SMA({params.get('short_window', 'N/A')}/{params.get('long_window', 'N/A')}), "
                            f"Trend={params.get('trend_strength_threshold', 'N/A')}\n")
            f.write("\n")
    
    # Save results to CSV
    csv_file = os.path.join(
        results_dir, 
        f'best_regime_results_{currency.replace("/", "_")}_{timestamp}.csv'
    )
    result_df.to_csv(csv_file)
    
    # Save visualizations
    try:
        from visualization import plot_enhanced_results_regime_specific
        plot_enhanced_results_regime_specific(result_df, {'regime_params': best_params}, metrics)
        print(f"Visualization saved to {results_dir}")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return best_params_file

# Modified main try-except block:
try:
    start_time = time.time()
    
    print("Running regime-aware purged walk-forward optimization...")
    
    # Import database handler to fetch data
    from database import DatabaseHandler
    db = DatabaseHandler()
    
    # Get complete data
    from enhanced_config import TRAINING_START, TESTING_END, CURRENCY
    print(f"Fetching historical data for {CURRENCY} from {TRAINING_START} to {TESTING_END}...")
    full_df = db.get_historical_data(CURRENCY, TRAINING_START, TESTING_END)
    
    # Create complete config
    from enhanced_config import (STRATEGY_CONFIG, WALK_FORWARD_CONFIG, LEARNING_CONFIG)
    full_config = {
        'STRATEGY_CONFIG': STRATEGY_CONFIG,
        'WALK_FORWARD_CONFIG': WALK_FORWARD_CONFIG,
        'LEARNING_CONFIG': LEARNING_CONFIG,
        'CURRENCY': CURRENCY
    }
    
    # Run purged walk-forward optimization
    overall_results = run_purged_walk_forward_optimization(full_df, full_config)
    
    # Extract best parameters for each regime
    print("\nExtracting best parameters for each regime...")
    best_params = extract_best_regime_parameters(overall_results['section_results'])
    
    # Evaluate strategy with best parameters
    if 'precomputed_regimes' in overall_results and overall_results['precomputed_regimes']:
        # Add precomputed regimes to the full dataset for final evaluation
        print("\nPreparing data with precomputed regimes for final evaluation...")
        eval_df = full_df.copy()
        eval_df['precomputed_regime'] = overall_results['global_regimes']
        
        # Evaluate with precomputed regimes
        print("\nEvaluating strategy with best parameters on entire period using precomputed regimes...")
        result_df, metrics = evaluate_strategy_with_precomputed_regimes(eval_df, best_params, full_config)
    else:
        # Standard evaluation
        print("\nEvaluating strategy with best parameters on entire period...")
        result_df, metrics = evaluate_strategy_with_best_params(full_df, best_params, full_config)
    
    # Save results
    print("\nSaving best regime results...")
    best_params_file = save_best_regime_results(best_params, result_df, metrics, full_config)
    print(f"Best regime results saved to {best_params_file}")
    
    # Create an enhanced visualization with regime information
    print("\nCreating enhanced visualization...")
    if 'global_regimes' in overall_results:
        # Use precomputed regimes for visualization
        plot_enhanced_results_with_regimes(result_df, best_params, metrics, 
                                          overall_results['global_regimes'], full_config)
    else:
        # Standard visualization
        plot_enhanced_results_regime_specific(result_df, {'regime_params': best_params}, metrics)
    
    # Analyze regime transitions and distributions
    print("\nAnalyzing regime distributions and transitions...")
    analyze_regime_distributions(result_df, best_params, full_config)
    
    # Close database connection
    db.close()
    
    # Print overall results
    print("\nStrategy Summary:")
    print(f"  Total Return: {overall_results['overall_return']:.2%}")
    print(f"  Buy & Hold Return: {overall_results['overall_buy_hold']:.2%}")
    print(f"  Outperformance: {overall_results['overall_outperformance']:.2%}")
    print(f"  Sharpe Ratio: {overall_results['overall_sharpe']:.2f}")
    print(f"  Max Drawdown: {overall_results['overall_max_drawdown']:.2%}")
    print(f"  Sections: {overall_results['section_count']}")
    
    # Print best parameters results
    print("\nBest Parameters Strategy Summary:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Buy & Hold Return: {result_df['buy_hold_cumulative'].iloc[-1] - 1:.2%}")
    print(f"  Outperformance: {metrics['total_return'] - (result_df['buy_hold_cumulative'].iloc[-1] - 1):.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Print regime-specific performance
    print("\nRegime-Specific Performance:")
    regimes = sorted(best_params.keys())
    for regime_id in regimes:
        regime_mask = (result_df['regime'] == regime_id)
        if regime_mask.any():
            regime_returns = result_df.loc[regime_mask, 'strategy_returns']
            if len(regime_returns) > 0:
                regime_return = (1 + regime_returns).cumprod().iloc[-1] - 1
                print(f"  Regime {regime_id}: Return={regime_return:.2%}, "
                      f"Params: SMA({best_params[regime_id].get('short_window')}/{best_params[regime_id].get('long_window')})")
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\nTotal runtime: {duration:.2f} minutes")
    
except Exception as e:
    print(f"ERROR: Strategy execution failed with error: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Ensure database connections are closed
    try:
        from database import DatabaseHandler
        print("Shutting down database connection pool...")
        DatabaseHandler.shutdown_pool()
    except Exception as e:
        print(f"Error shutting down database pool: {e}")
    
print("\nLog file saved to:", log_file)
print("=" * 80)