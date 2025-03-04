#!/usr/bin/env python
# run_strategy.py - Script to run the enhanced SMA strategy

import os
import sys
import time
from datetime import datetime

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import our modules
try:
    from enhanced_sma import run_enhanced_backtest_regime_specific as run_enhanced_backtest
except ImportError as e:
    print(f"Error: Could not import enhanced_sma module: {e}")
    print("Please ensure enhanced_sma.py and all its dependencies are in the current directory.")
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
print(f"Enhanced SMA Strategy Runner (Regime-Specific)")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Logging to: {log_file}")
print("=" * 80)
print()

# Modified main try-except block:

# Modified main try-except block in run_strategy.py:

try:
    start_time = time.time()
    
    # Check if purged walk-forward is enabled
    from enhanced_config import WALK_FORWARD_CONFIG
    
    if WALK_FORWARD_CONFIG.get('enabled', False):
        print("Running purged walk-forward optimization...")
        
        # Import purged walk-forward module
        from purged_walk_forward import run_purged_walk_forward_optimization
        
        # Import database handler to fetch data
        from database import DatabaseHandler
        db = DatabaseHandler()
        
        # Get complete data
        from enhanced_config import TRAINING_START, TESTING_END, CURRENCY
        full_df = db.get_historical_data(CURRENCY, TRAINING_START, TESTING_END)
        
        # Create complete config
        from enhanced_config import (STRATEGY_CONFIG, WALK_FORWARD_CONFIG, LEARNING_CONFIG)
        full_config = {
            'STRATEGY_CONFIG': STRATEGY_CONFIG,
            'WALK_FORWARD_CONFIG': WALK_FORWARD_CONFIG,
            'LEARNING_CONFIG': LEARNING_CONFIG
        }
        
        # Run purged walk-forward optimization
        overall_results = run_purged_walk_forward_optimization(full_df, full_config)
        
        # Close database connection
        db.close()
        
        # Print overall results
        print("\nStrategy Summary:")
        print(f"  Total Return: {overall_results['overall_return']:.2%}")
        print(f"  Buy & Hold Return: {overall_results['overall_buy_hold']:.2%}")
        print(f"  Outperformance: {overall_results['overall_outperformance']:.2%}")
        print(f"  Sections: {overall_results['section_count']}")
    else:
        print("Running enhanced SMA backtest with regime-specific optimization...")
        from enhanced_sma import run_enhanced_backtest_regime_specific
        result_df, regime_params, metrics = run_enhanced_backtest_regime_specific()
        
        print("\nStrategy Summary:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
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