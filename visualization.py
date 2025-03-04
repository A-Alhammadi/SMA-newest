# visualization.py
# Functions for visualizing backtest results

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

def get_safe_integer(value, default=3):
    """
    Convert a value to an integer safely, handling lists and other types.
    
    Parameters:
        value (any): Value to convert
        default (int): Default value if conversion fails
        
    Returns:
        int: Converted integer
    """
    if isinstance(value, list):
        return int(value[0]) if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
    
import matplotlib.colors as mcolors

# Convert regime colors from names to RGBA tuples
regime_colors = [mcolors.to_rgba(color) for color in ['green', 'gray', 'red', 'purple', 'orange']]

def plot_enhanced_results_regime_specific(df, params, metrics):
    """
    Plot enhanced backtest results with regime-specific parameter visualization.
    
    Parameters:
        df (DataFrame): Results DataFrame
        params (dict): Dictionary containing regime-specific parameters
        metrics (dict): Performance metrics
    """
    # Get config data
    try:
        from enhanced_config import (
            PLOT_RESULTS, RESULTS_DIR, CURRENCY, INITIAL_CAPITAL, STRATEGY_CONFIG
        )
    except ImportError:
        try:
            from config import (
                PLOT_RESULTS, RESULTS_DIR, CURRENCY, INITIAL_CAPITAL, STRATEGY_CONFIG
            )
        except ImportError:
            # Use defaults if import fails
            PLOT_RESULTS = True
            RESULTS_DIR = "enhanced_sma_results"
            CURRENCY = "BTC/USD"
            INITIAL_CAPITAL = 10000
            STRATEGY_CONFIG = {
                'regime_detection': {
                    'regime_opt_out': {},
                    'regime_buy_hold': {}
                }
            }
    
    if not PLOT_RESULTS:
        return
    
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Extract regime parameters
    regime_params = params.get('regime_params', {})
    
    # Check if there were any trades
    position_changes = df['managed_position'].diff().fillna(0).abs()
    num_trades = int((position_changes != 0).sum())
    
    # Gather exit statistics
    max_dd_exits = df['max_drawdown_exits'].max() if 'max_drawdown_exits' in df.columns else 0
    profit_exits = df['profit_taking_exits'].max() if 'profit_taking_exits' in df.columns else 0
    trail_exits = df['trailing_stop_exits'].max() if 'trailing_stop_exits' in df.columns else 0
    
    # Check if buy & hold was used
    buy_hold_used = 'buy_hold_mask' in df.columns and df['buy_hold_mask'].any()
    buy_hold_periods = df['buy_hold_mask'].sum() if buy_hold_used else 0
    buy_hold_pct = (buy_hold_periods / len(df)) * 100 if buy_hold_used else 0
    
    # Create figure with more subplots for regime-specific visualization
    fig, axs = plt.subplots(7, 1, figsize=(14, 28), 
                           gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1]})
    
    # -------------------------------------------------------------------------
    # Plot 1: Price and Performance
    # -------------------------------------------------------------------------
    ax1 = axs[0]
    ax1.set_title(f'Enhanced SMA Strategy for {CURRENCY} with Regime-Specific Parameters', fontsize=16)
    ax1.plot(df.index, df['close_price'], color='gray', alpha=0.6, label='Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['strategy_cumulative'] * INITIAL_CAPITAL, 'b-', label='Strategy')
    ax1_twin.plot(df.index, df['buy_hold_cumulative'] * INITIAL_CAPITAL, 'r--', label='Market Buy & Hold')
    
    # Highlight different regimes
    n_regimes = len(regime_params)
    regime_colors = ['green', 'gray', 'red', 'purple', 'orange']  # Add more colors if needed
    
    for i in range(n_regimes):
        regime_mask = df['regime'] == i
        if regime_mask.any():
            # Determine color (handle more regimes than we have colors)
            color_idx = i % len(regime_colors)
            
            # Add background shading for the regime
            for j in range(len(df) - 1):
                if regime_mask.iloc[j]:
                    ax1.axvspan(df.index[j], df.index[j+1], color=regime_colors[color_idx], alpha=0.05)
    
    # Highlight buy & hold periods if used
    if buy_hold_used:
        buy_hold_mask = df['buy_hold_mask']
        for i in range(len(df) - 1):
            if buy_hold_mask.iloc[i]:
                ax1.axvspan(df.index[i], df.index[i+1], color='gold', alpha=0.2)
    
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Portfolio Value ($)')
    
    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    if buy_hold_used:
        from matplotlib.patches import Patch
        buy_hold_patch = Patch(color='gold', alpha=0.2, label='Buy & Hold Periods')
        ax1.legend(lines1 + lines2 + [buy_hold_patch], labels1 + labels2 + ['Buy & Hold Periods'], loc='upper left')
    else:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 2: Volatility and Regimes with Parameter Labels
    # -------------------------------------------------------------------------
    ax2 = axs[1]
    ax2.set_title('Volatility and Regimes with Optimized Parameters', fontsize=14)
    ax2.plot(df.index, df['volatility'], 'b-', label='Volatility')
    
    # Color background by regime and show parameter info
    regime_colors = ['green', 'gray', 'red', 'purple', 'orange']
    
    for regime_id, params in regime_params.items():
        regime_mask = df['regime'] == regime_id
        if regime_mask.any():
            # Use default colors if we have more regimes than colors
            color_idx = min(regime_id, len(regime_colors) - 1)
            color = regime_colors[color_idx]
            
            # Extract key parameters for label
            short_window = params.get('short_window', 13)
            long_window = params.get('long_window', 55)
            trend_threshold = params.get('trend_strength_threshold', 0.3)
            
            # Create label with parameter info
            label = f'Regime {regime_id}: SMA({short_window}/{long_window}), Trend={trend_threshold:.2f}'
            
            # Check if this regime has special behavior
            if STRATEGY_CONFIG['regime_detection'].get('regime_opt_out', {}).get(regime_id, False):
                label += " (Opt-out)"
            elif STRATEGY_CONFIG['regime_detection'].get('regime_buy_hold', {}).get(regime_id, False):
                label += " (Buy & Hold)"
                
            ax2.fill_between(df.index, 0, df['volatility'].max(), where=regime_mask, 
                             color=color, alpha=0.2, label=label)
    
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 3: Trading Signals and Position
    # -------------------------------------------------------------------------
    ax3 = axs[2]
    ax3.set_title('Signals, Positions and Risk Management', fontsize=14)
    ax3.plot(df.index, df['raw_signal'], 'k--', alpha=0.5, label='Raw Signal')
    ax3.plot(df.index, df['filtered_signal'], 'g-', alpha=0.7, label='Filtered Signal')
    ax3.plot(df.index, df['managed_position'], 'b-', linewidth=1.5, label='Final Position')
    
    # Highlight different regimes
    for regime_id, params in regime_params.items():
        regime_mask = df['regime'] == regime_id
        if regime_mask.any():
            color_idx = min(regime_id, len(regime_colors) - 1)
            for i in range(len(df) - 1):
                if regime_mask.iloc[i]:
                    ax3.axvspan(df.index[i], df.index[i+1], color=regime_colors[color_idx], alpha=0.05)
    
    # Highlight buy & hold periods if used
    if buy_hold_used:
        buy_hold_mask = df['buy_hold_mask']
        for i in range(len(df) - 1):
            if buy_hold_mask.iloc[i]:
                ax3.axvspan(df.index[i], df.index[i+1], color='gold', alpha=0.2)
    
    # Highlight trend strength on the twin axis
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df.index, df['trend_strength'], 'r-', alpha=0.3, label='Trend Strength')
    
    # Add threshold lines
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    
    # Plot threshold lines for each regime
    for regime_id, params in regime_params.items():
        threshold = params.get('trend_strength_threshold', 0.3)
        if isinstance(threshold, list):
            threshold = threshold[0]
        
        # Only add lines for regimes that actually appear in the data
        if (df['regime'] == regime_id).any():
            ax3_twin.axhline(y=threshold, color=regime_colors[min(regime_id, len(regime_colors)-1)], 
                           linestyle='--', alpha=0.3, 
                           label=f'Threshold Regime {regime_id}: {threshold:.2f}')
    
    ax3.set_ylabel('Position')
    ax3_twin.set_ylabel('Trend Strength')
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    if buy_hold_used:
        from matplotlib.patches import Patch
        buy_hold_patch = Patch(color='gold', alpha=0.2, label='Buy & Hold Periods')
        ax3.legend(lines1 + lines2 + [buy_hold_patch], labels1 + labels2 + ['Buy & Hold Periods'], loc='upper left')
    else:
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 4: Drawdown
    # -------------------------------------------------------------------------
    ax4 = axs[3]
    drawdown = df['strategy_cumulative'] / df['strategy_cumulative'].cummax() - 1
    ax4.set_title(f'Drawdown (Max: {metrics["max_drawdown"]:.2%})', fontsize=14)
    ax4.fill_between(df.index, drawdown * 100, 0, color='red', alpha=0.3)

    # Highlight different regimes
    for regime_id, params in regime_params.items():
        regime_mask = df['regime'] == regime_id
        if regime_mask.any():
            color_idx = min(regime_id, len(regime_colors) - 1)
            for i in range(len(df) - 1):
                if regime_mask.iloc[i]:
                    ax4.axvspan(df.index[i], df.index[i+1], color=regime_colors[color_idx], alpha=0.05)

    # Highlight buy & hold periods in drawdown chart if used
    if buy_hold_used:
        buy_hold_mask = df['buy_hold_mask']
        for i in range(len(df) - 1):
            if buy_hold_mask.iloc[i]:
                ax4.axvspan(df.index[i], df.index[i+1], color='gold', alpha=0.2)

    ax4.set_ylabel('Drawdown (%)')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Plot stop loss lines for each regime
    for regime_id, params in regime_params.items():
        max_dd_exit = params.get('max_drawdown_exit', 0.15)
        if isinstance(max_dd_exit, list):
            max_dd_exit = max_dd_exit[0]
        
        # Only add lines for regimes that actually appear in the data
        if (df['regime'] == regime_id).any():
            ax4.axhline(
                y=max_dd_exit * 100,
                color=regime_colors[min(regime_id, len(regime_colors)-1)],
                linestyle='--',
                alpha=0.5,
                label=f'Stop Loss R{regime_id}: {max_dd_exit * 100:.0f}%'
            )
    
    ax4.legend(loc='lower left')
    
    # -------------------------------------------------------------------------
    # Plot 5: Moving Averages - Show all regime-specific MAs
    # -------------------------------------------------------------------------
    ax5 = axs[4]
    ax5.set_title(f'Moving Averages (Regime-Specific)', fontsize=14)
    
    # Use subset of data for clarity (last 30% of the data)
    start_idx = int(len(df) * 0.7)
    subset_idx = df.index[start_idx:]
    
    ax5.plot(subset_idx, df.loc[subset_idx, 'close_price'], color='gray', alpha=0.6, label='Price')
    
    # Plot short and long MAs for each regime
    for regime_id, params in regime_params.items():
        short_window = params.get('short_window', 13)
        long_window = params.get('long_window', 55)
        
        # Get regime mask for the subset
        regime_mask = df.loc[subset_idx, 'regime'] == regime_id
        
        # Only plot if this regime appears in the subset
        if regime_mask.any():
            color_idx = min(regime_id, len(regime_colors) - 1)
            color = regime_colors[color_idx]
            
            # Extract periods for this regime
            regime_idx = subset_idx[regime_mask]
            
            # Calculate MAs for visualization (just for the plot)
            short_ma = df.loc[subset_idx, 'close_price'].rolling(window=short_window).mean()
            long_ma = df.loc[subset_idx, 'close_price'].rolling(window=long_window).mean()
            
            # Plot MA lines only during this regime's periods
            ax5.plot(regime_idx, short_ma.loc[regime_idx], color=color, linestyle='-', alpha=0.8,
                   label=f'R{regime_id} Short MA ({short_window})')
            ax5.plot(regime_idx, long_ma.loc[regime_idx], color=color, linestyle='--', alpha=0.8,
                   label=f'R{regime_id} Long MA ({long_window})')
    
    # Color background by regime
    for i in range(start_idx, len(df) - 1):
        regime_id = df['regime'].iloc[i]
        color_idx = min(regime_id, len(regime_colors) - 1)
        ax5.axvspan(df.index[i], df.index[i+1], color=regime_colors[color_idx], alpha=0.05)
    
    # Highlight positions
    for i in range(start_idx, len(df) - 1):
        if buy_hold_used and df['buy_hold_mask'].iloc[i]:
            ax5.axvspan(df.index[i], df.index[i+1], color='gold', alpha=0.2)
        elif df['managed_position'].iloc[i] > 0:
            ax5.axvspan(df.index[i], df.index[i+1], color='green', alpha=0.1)
        elif df['managed_position'].iloc[i] < 0:
            ax5.axvspan(df.index[i], df.index[i+1], color='red', alpha=0.1)
    
    ax5.set_ylabel('Price & MAs')
    ax5.legend(loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 6: Regime-Specific Performance Comparison
    # -------------------------------------------------------------------------
    ax6 = axs[5]
    ax6.set_title('Regime-Specific Performance', fontsize=14)
    
    # Calculate performance by regime
    regime_returns = []
    regime_market_returns = []
    regime_alphas = []
    regime_sharpes = []
    regime_labels = []
    
    for regime_id in range(len(regime_params)):
        regime_mask = df['regime'] == regime_id
        if regime_mask.any():
            regime_strategy_returns = df.loc[regime_mask, 'strategy_returns']
            regime_market_returns_data = df.loc[regime_mask, 'returns']
            
            if len(regime_strategy_returns) > 0:
                regime_return = (1 + regime_strategy_returns).prod() - 1
                regime_market_return = (1 + regime_market_returns_data).prod() - 1
                regime_alpha = regime_return - regime_market_return
                
                from performance_metrics import calculate_sharpe_ratio
                regime_sharpe = calculate_sharpe_ratio(regime_strategy_returns)
                
                regime_returns.append(regime_return * 100)  # Convert to percentage
                regime_market_returns.append(regime_market_return * 100)
                regime_alphas.append(regime_alpha * 100)
                regime_sharpes.append(regime_sharpe)
                regime_labels.append(f'Regime {regime_id}')
    
    # Create bar chart with 4 grouped bars (strategy return, market return, alpha, sharpe)
    width = 0.2
    x = np.arange(len(regime_labels))
    
    ax6.bar(x - width*1.5, regime_returns, width, color='blue', alpha=0.7, label='Strategy Return')
    ax6.bar(x - width/2, regime_market_returns, width, color='red', alpha=0.7, label='Market Return')
    ax6.bar(x + width/2, regime_alphas, width, color='green', alpha=0.7, label='Alpha')
    ax6.bar(x + width*1.5, regime_sharpes, width, color='purple', alpha=0.7, label='Sharpe Ratio')
    
    ax6.set_ylabel('Percentage / Ratio')
    ax6.set_xticks(x)
    ax6.set_xticklabels(regime_labels)
    ax6.legend()
    
    # Add parameter info to the bars
    for i, regime_id in enumerate(range(len(regime_params))):
        if regime_id in regime_params:
            params = regime_params[regime_id]
            short_window = params.get('short_window', 13)
            long_window = params.get('long_window', 55)
            
            if regime_returns and i < len(regime_returns):
                ax6.text(i, regime_returns[i] + 1, f'SMA({short_window}/{long_window})', 
                        ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Add value labels to bars
    for i in range(len(regime_labels)):
        if i < len(regime_returns):
            ax6.text(i - width*1.5, regime_returns[i] + 0.5, f'{regime_returns[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_market_returns):
            ax6.text(i - width/2, regime_market_returns[i] + 0.5, f'{regime_market_returns[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_alphas):
            ax6.text(i + width/2, regime_alphas[i] + 0.5, f'{regime_alphas[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_sharpes):
            ax6.text(i + width*1.5, regime_sharpes[i] + 0.5, f'{regime_sharpes[i]:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # -------------------------------------------------------------------------
    # Plot 7: Parameters Summary by Regime
    # -------------------------------------------------------------------------
    ax7 = axs[6]
    ax7.set_title('Regime-Specific Parameter Summary', fontsize=14)
    ax7.axis('off')  # Turn off axis
    
    # Create parameter summary table
    table_data = []
    header = ['Regime', 'Short MA', 'Long MA', 'Trend Thresh', 'Target Vol', 'Max DD Exit', 'Profit Taking']
    table_data.append(header)
    
    for regime_id in range(len(regime_params)):
        if regime_id in regime_params:
            params = regime_params[regime_id]
            row = [
                f'Regime {regime_id}',
                str(params.get('short_window', 13)),
                str(params.get('long_window', 55)),
                f"{params.get('trend_strength_threshold', 0.3):.2f}",
                f"{params.get('target_vol', 0.15):.2f}",
                f"{params.get('max_drawdown_exit', 0.15):.2f}",
                f"{params.get('profit_taking_threshold', 0.05):.2f}"
            ]
            table_data.append(row)
    
    # Add a table to the plot
    table = ax7.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15],
        cellColours = [[(0.9, 0.9, 0.9)] * len(header)] + [[regime_colors[i % len(regime_colors)]] * len(header) for i in range(len(table_data)-1)]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Extended performance summary at the bottom
    regime_summary = " | ".join([
        f"R{regime_id}: SMA({params.get('short_window', 13)}/{params.get('long_window', 55)})"
        for regime_id, params in regime_params.items()
    ])
    
    plt.figtext(
        0.1, 0.01, 
        (
            f"Return: {metrics['total_return']:.2%} | Annual: {metrics['annualized_return']:.2%} | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | Sortino: {metrics['sortino_ratio']:.2f} | "
            f"Calmar: {metrics['calmar_ratio']:.2f} | MaxDD: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%} | Gain/Pain: {metrics['gain_to_pain']:.2f} | "
            f"Volatility: {metrics['volatility']:.2%} | "
            f"Buy & Hold: {df['buy_hold_cumulative'].iloc[-1] - 1:.2%} | "
            f"Alpha: {metrics['total_return'] - (df['buy_hold_cumulative'].iloc[-1] - 1):.2%}\n"
            f"Regime Parameters: {regime_summary}\n"
            + (f"\nWARNING: No trades executed during backtest period" if num_trades == 0 else "")
        ),
        ha='left', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Adjust bottom margin to fit the additional info
    
    # Save the figure
    plot_filename = os.path.join(RESULTS_DIR, f'enhanced_sma_regime_specific_results_{CURRENCY.replace("/", "_")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to {plot_filename}")

def save_enhanced_results_regime_specific(df, regime_params, metrics, cv_results):
    """
    Save enhanced backtest results with regime-specific parameters to files.
    
    Parameters:
        df (DataFrame): Results DataFrame
        regime_params (dict): Regime-specific optimized parameters
        metrics (dict): Performance metrics
        cv_results (list): Cross-validation results
    """
    # Get config data
    try:
        from enhanced_config import (
            SAVE_RESULTS, RESULTS_DIR, CURRENCY, TRADING_FREQUENCY, INITIAL_CAPITAL,
            TRADING_FEE_PCT, STRATEGY_CONFIG
        )
    except ImportError:
        try:
            from config import (
                SAVE_RESULTS, RESULTS_DIR, CURRENCY, TRADING_FREQUENCY, INITIAL_CAPITAL,
                TRADING_FEE_PCT, STRATEGY_CONFIG
            )
        except ImportError:
            # Use defaults if import fails
            SAVE_RESULTS = True
            RESULTS_DIR = "enhanced_sma_results"
            CURRENCY = "BTC/USD"
            TRADING_FREQUENCY = "1H"
            INITIAL_CAPITAL = 10000
            TRADING_FEE_PCT = 0.001
            STRATEGY_CONFIG = {
                'regime_detection': {
                    'regime_opt_out': {},
                    'regime_buy_hold': {}
                }
            }
    
    if not SAVE_RESULTS:
        return
    
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Calculate position changes and check if there are any trades
    position_changes = df['managed_position'].diff().fillna(0).abs()
    num_trades = int((position_changes != 0).sum())
    
    # Check if buy & hold was used
    buy_hold_used = 'buy_hold_mask' in df.columns and df['buy_hold_mask'].any()
    buy_hold_periods = df['buy_hold_mask'].sum() if buy_hold_used else 0
    
    # Prepare results file
    results_file = os.path.join(
        RESULTS_DIR,
        f'enhanced_sma_regime_results_{CURRENCY.replace("/", "_")}.txt'
    )
    
    with open(results_file, 'w') as f:
        f.write("===== ENHANCED SMA STRATEGY RESULTS (REGIME-SPECIFIC) =====\n\n")
        
        # Basic config info
        f.write("Strategy Configuration:\n")
        f.write(f"Trading Frequency: {TRADING_FREQUENCY}\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Trading Fee: {TRADING_FEE_PCT:.4%} per trade\n\n")
        
        # Regime-specific optimized parameters
        f.write("===== REGIME-SPECIFIC OPTIMIZED PARAMETERS =====\n\n")
        for regime_id, params in regime_params.items():
            f.write(f"Regime {regime_id} Parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Check if no trades
        if num_trades == 0:
            f.write("WARNING: No trades were executed during the backtest period.\n")
            f.write("This may be due to all regimes being set to opt-out or other restrictive settings.\n\n")
        
        # Buy & hold info, if used
        if buy_hold_used:
            f.write("Buy & Hold Usage:\n")
            f.write(f"Total periods using buy & hold: {buy_hold_periods} ({buy_hold_periods/len(df)*100:.2f}%)\n")
            
            # Performance during buy & hold
            if buy_hold_periods > 0:
                bh_returns = df.loc[df['buy_hold_mask'], 'strategy_returns']
                bh_return = (1 + bh_returns).prod() - 1 if len(bh_returns) > 0 else 0
                f.write(f"Buy & hold period return: {bh_return:.4%}\n\n")
        
        # Overall performance metrics
        f.write("===== OVERALL PERFORMANCE METRICS =====\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                # Print some as percentage
                if key in ['total_return','annualized_return','volatility','max_drawdown','win_rate']:
                    f.write(f"{key}: {value:.4%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Buy & hold comparison
        buy_hold_return = df['buy_hold_cumulative'].iloc[-1] - 1
        f.write(f"\nBuy & Hold Return: {buy_hold_return:.4%}\n")
        f.write(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}\n")
        
        # Trade stats
        f.write(f"\nNumber of Trades: {num_trades}\n")
        if num_trades > 0:
            trade_durations = []
            in_trade = False
            trade_start = None
            
            for i, (timestamp, position) in enumerate(df['managed_position'].items()):
                if not in_trade and position != 0:
                    in_trade = True
                    trade_start = timestamp
                elif in_trade and position == 0:
                    in_trade = False
                    if trade_start is not None:
                        duration = (timestamp - trade_start).total_seconds() / 3600  # hours
                        trade_durations.append(duration)
            
            if trade_durations:
                avg_duration = np.mean(trade_durations)
                f.write(f"Average Trade Duration: {avg_duration:.2f} hours\n")
        
        # Regime distribution and performance
        f.write("\n===== REGIME DISTRIBUTION AND PERFORMANCE =====\n\n")
        regimes = df['regime']
        strategy_returns = df['strategy_returns']
        market_returns = df['returns']
        
        for regime_id in sorted(regime_params.keys()):
            regime_mask = (regimes == regime_id)
            regime_count = regime_mask.sum()
            regime_pct = regime_count / len(df) * 100
            
            # Get parameters for this regime
            params = regime_params.get(regime_id, {})
            short_window = params.get('short_window', 13)
            long_window = params.get('long_window', 55)
            trend_threshold = params.get('trend_strength_threshold', 0.3)
            
            f.write(f"Regime {regime_id} ({regime_count} periods, {regime_pct:.2f}%):\n")
            f.write(f"  Parameters: SMA({short_window}/{long_window}), Trend Threshold={trend_threshold:.2f}\n")
            
            if regime_mask.any():
                regime_strategy_returns = strategy_returns[regime_mask]
                regime_market_returns = market_returns[regime_mask]
                
                if len(regime_strategy_returns) > 0:
                    regime_return = (1 + regime_strategy_returns).prod() - 1
                    regime_market_return = (1 + regime_market_returns).prod() - 1
                    regime_alpha = regime_return - regime_market_return
                    
                    # Calculate additional metrics
                    from performance_metrics import calculate_sharpe_ratio, calculate_sortino_ratio
                    regime_sharpe = calculate_sharpe_ratio(regime_strategy_returns)
                    regime_sortino = calculate_sortino_ratio(regime_strategy_returns)
                    
                    f.write(f"  Strategy Return: {regime_return:.4%}\n")
                    f.write(f"  Market Return: {regime_market_return:.4%}\n")
                    f.write(f"  Alpha: {regime_alpha:.4%}\n")
                    f.write(f"  Sharpe Ratio: {regime_sharpe:.4f}\n")
                    f.write(f"  Sortino Ratio: {regime_sortino:.4f}\n")
                    
                    # Calculate trades in this regime
                    regime_position_changes = df.loc[regime_mask, 'managed_position'].diff().fillna(0).abs()
                    regime_trades = int((regime_position_changes != 0).sum())
                    f.write(f"  Trades in regime: {regime_trades}\n")
            f.write("\n")
            
        # Save model parameters to pickle
        import joblib
        model_file = os.path.join(
            RESULTS_DIR,
            f'enhanced_sma_regime_model_{CURRENCY.replace("/", "_")}.pkl'
        )
        
        model_data = {
            'regime_params': regime_params,
            'config': STRATEGY_CONFIG,
            'metrics': metrics,
            'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(model_data, model_file)
        print(f"Model saved to {model_file}")
    
    print(f"Results saved to {results_file}")
    
    # Save DataFrame to CSV
    csv_file = os.path.join(
        RESULTS_DIR,
        f'enhanced_sma_regime_data_{CURRENCY.replace("/", "_")}.csv'
    )
    df.to_csv(csv_file)
    print(f"Data saved to {csv_file}")