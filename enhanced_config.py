############################################################
#                   DATABASE SETTINGS                      #
############################################################

DB_CONFIG = {
    'dbname': 'cryptocurrencies',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',
    'port': '5432'
}

############################################################
#                   BACKTEST SETTINGS                      #
############################################################

# Data and Backtesting Settings
TRADING_FREQUENCY = "1H"  # Frequency of data (1H = hourly, 1D = daily)
TRAINING_START = "2020-01-01"
TRAINING_END = "2020-12-31"
TESTING_START = "2021-01-01"
TESTING_END = "2024-12-31"
CURRENCY = "BTC/USD"  # Base currency to analyze
INITIAL_CAPITAL = 10000
TRADING_FEE_PCT = 0.001  # Example: 0.1% trading fee per trade

############################################################
#               WALK-FORWARD OPTIMIZATION                  #
############################################################

WALK_FORWARD_CONFIG = {
    'enabled': True,                   # Enable walk-forward optimization
    'window_type': 'sliding',          # Window type: 'expanding' or 'sliding'
    'sliding_window_size': 365,        # Size of sliding window in days (for sliding window)
    'section_length_days': 365,        # NOT USED IN SLIDING Length of each walk-forward section in days
    'initial_training_days': 365,      # NOT USED IN SLIDING Initial training period length in days (for expanding window)
    'validation_days': 45,             # Validation period after each training period
    'test_days': 90,                   # Testing period after each validation
    'step_days': 90,                   # Days to step forward in each iteration
    'purge_days': 30,                  # Days to purge from training
    'embargo_days': 30,                # Days to wait after test period (embargo)
    'min_training_size': 365,          # Minimum required training data size
    'max_training_size': 1095,         # Maximum training data (for expanding window)
    'max_sections': 25,                # Maximum number of walk-forward sections to process
    'save_sections': True,             # Save results for each walk-forward section
    'min_regime_data_points': 50,      # Minimum data points required for a regime to be valid
}

############################################################
#                  LEARNING PARAMETERS                     #
############################################################

LEARNING_CONFIG = {
    'enabled': True,                  # Enable learning from previous periods
    'exploration_pct': 0.30,          # Percentage of parameter combinations for exploration (new random parameters)
    'exploitation_pct': 0.70,         # Percentage for exploitation (parameters from previous good results)
    'top_params_pct': 0.20,           # Percentage of top parameters to keep from previous sections
    'mutation_probability': 0.25,     # Probability of mutating a parameter when exploiting previous results
    'mutation_factor': 0.20,          # How much to mutate parameters (as a percentage of parameter range)
    'max_history_sections': 3,        # Maximum number of previous sections to learn from
    'param_consistency_weight': 0.30, # Weight given to parameter consistency vs. performance
}

############################################################
#                  ENHANCED STRATEGY SETTINGS              #
############################################################

# Strategy parameters for the enhanced SMA model
STRATEGY_CONFIG = {
    # Volatility calculation settings
    'volatility': {
        'methods': ['parkinson'],  # 'parkinson', 'standard', 'yang_zhang', 'garch' Added all methods
        'lookback_periods': [8, 13, 20, 34, 50, 80, 100, 120, 150, 200],  # More Fibonacci and round numbers
        'regime_smoothing': [2, 3, 5, 8, 10, 13, 21],  # Added more Fibonacci-based smoothing periods
        'min_history_multiplier': 5,
    },
    
    # Regime detection settings
    'regime_detection': {
        'methods': ['hmm'],  #'kmeans', 'quantile', 'hmm' Added more detection methods
        'n_regimes': [2],  # Testing different number of regimes
        'quantile_thresholds': [
            [0.33, 0.67],       # For 3 regimes (standard)
            [0.25, 0.5, 0.75],  # For 4 regimes (quartiles)
            [0.5],              # For 2 regimes (median)
            [0.2, 0.8],         # For 3 regimes (more extreme)
            [0.3, 0.7]          # For 3 regimes (moderate)
        ],
        'regime_stability_period': [12, 24, 36, 48, 72],  # More options for stability timing
        'regime_opt_out': {
            0: False,
            1: False,
            2: False
        },
        'regime_buy_hold': {
            0: False,
            1: False,
            2: False
        }
    },
    
    # SMA strategy settings
    'sma': {
        'short_windows': [3,  5, 8, 12, 21],  # Extended short windows including Fibonacci
        'long_windows': [21, 24, 34, 55, 72, 89, 120, 167, 200, 240, 300],  # More long windows
        'min_holding_period': [6, 12, 24],  # More holding period options (hours)
        'trend_filter_period': [89, 120, 167, 200, 240],  # More trend filter options
        'trend_strength_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # More threshold options
    },
    
    # Risk management settings
    'risk_management': {
        'target_volatility': [0.15, 0.2, 0.25, 0.3, 0.35],  # More volatility targets
        'max_position_size': [0.8, 1.0],  # Position size options
        'min_position_size': [0.05, 0.1, 0.2],  # Min position options
        'max_drawdown_exit': [0.1, 0.12, 0.15, 0.18, 0.2],  # More exit thresholds
        'profit_taking_threshold': [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35],  # More profit targets
        'trailing_stop_activation': [0.03, 0.05, 0.08, 0.1, 0.12, 0.15],  # More activation levels
        'trailing_stop_distance': [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1],  # More trailing distances
        'materiality_threshold': 0.05,
    },
    
    # Cross-validation settings - Modified for Walk-Forward compatibility
    'cross_validation': {
        'parameter_testing': {
            'method': 'optuna',  # Changed from 'greedy' to 'optuna'
            'n_trials': 20000,  # Number of Optuna trials per regime optimization
            'timeout': 1200,  # Maximum seconds per optimization (optional)
            'n_random_combinations': 50000,  # Still used for fallback if Optuna fails
            'max_combinations': 500000,  # Maximum number of combinations for fallback
            'optimize_risk_params': True,  # Whether to optimize risk params or use defaults
            'optimize_regime_params': True,  # Whether to optimize regime detection parameters
            'optimize_sma_params': True,  # Whether to optimize SMA parameters
            'advanced_mode': True,  # Set to True to enable all parameter combinations
            'early_stop_threshold': 1000,  # For fallback method
            'min_combinations': 200,  # Minimum combinations for fallback method
            'print_frequency': 50,  # How often to print progress during optimization
            'adaptive_sampling': True,  # Use adaptive sampling for walk-forward
            'use_simplified_scoring': True,  # Use simplified scoring (Sharpe + Returns)
            'optuna_settings': {
                'pruner': 'median',  # Pruner type: 'median', 'percentile', 'hyperband', 'none'
                'n_startup_trials': 10,  # Number of random trials before pruning begins
                'n_warmup_steps': 5,  # Number of steps before pruning can happen in a trial
                'show_progress_bar': True  # Show Optuna progress bar during optimization
            }
        }
    },
    
    # Parameter selection settings - Simplified for Walk-Forward
    'parameter_selection': {
        'sharpe_weight': 0.70,  # Higher weight for Sharpe ratio
        'return_weight': 0.30,  # Lower weight for returns
        'consistency_weight': 0.20,  # Weight for consistency across regimes
        'stability_weight': 0.25,  # Reduced weight for parameter stability vs. performance
        'sortino_weight': 0.0,  # Removed from scoring
        'calmar_weight': 0.0  # Removed from scoring
    },
    
    # Parameter sensitivity analysis settings - NEW
    'parameter_sensitivity': {
        'enabled': True,  # Enable parameter sensitivity analysis
        'top_n_parameters': 10,  # Number of top parameters to analyze in detail
        'importance_methods': ['correlation', 'random_forest', 'linear_regression'],  # Methods for calculating importance
        'create_visualizations': True,  # Create visualizations for parameter importance
        'save_trial_database': True,  # Save complete database of all trials
        'correlation_weight': 0.3,  # Weight for correlation-based importance
        'random_forest_weight': 0.5,  # Weight for random forest-based importance
        'linear_regression_weight': 0.2  # Weight for linear regression-based importance
    }
}

############################################################
#                   OUTPUT SETTINGS                        #
############################################################

SAVE_RESULTS = True
PLOT_RESULTS = True
RESULTS_DIR = "enhanced_sma_results"