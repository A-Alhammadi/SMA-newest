# visualization.py
# Functions for visualizing backtest results

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from volatility import calculate_volatility

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


def plot_regime_distribution(df, regimes, config, filename=None):
    """
    Plot the distribution of regimes over time with volatility overlay.
    
    Parameters:
        df (DataFrame): Price data
        regimes (Series): Regime classifications
        config (dict): Configuration
        filename (str, optional): Filename to save plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from datetime import datetime
    
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: Price with regime shading
    ax1 = axs[0]
    price = df['close']
    ax1.plot(price.index, price.values, 'k-', linewidth=1)
    ax1.set_title('Price with Regime Classification', fontsize=14)
    
    # Add colored background for each regime
    colors = ['#E0F8FF', '#FFE0E0', '#E0FFE0', '#E0E0FF', '#FFF0E0']
    
    # Identify regime change points
    changes = regimes.diff().fillna(0) != 0
    change_points = regimes.index[changes].tolist()
    change_points = [regimes.index[0]] + change_points + [regimes.index[-1]]
    
    # Color regions by regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(colors)
        ax1.axvspan(start, end, alpha=0.3, color=colors[color_idx])
    
    # Add regime labels
    unique_regimes = sorted(regimes.unique())
    for regime_id in unique_regimes:
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            color_idx = int(regime_id) % len(colors)
            label = f"Regime {regime_id}"
            # Add a small colored rectangle for the legend
            ax1.plot([], [], color=colors[color_idx], alpha=0.6, linewidth=10, label=label)
    
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Price', fontsize=12)
    
    # Plot 2: Volatility
    ax2 = axs[1]
    volatility = df['volatility'] if 'volatility' in df.columns else calculate_volatility(df)
    ax2.plot(volatility.index, volatility.values, 'b-')
    ax2.set_title('Volatility', fontsize=14)
    ax2.set_ylabel('Volatility', fontsize=12)
    
    # Plot 3: Regime distribution as heatmap
    ax3 = axs[2]
    
    # Create masked arrays for each regime
    regime_arrays = []
    for regime_id in unique_regimes:
        mask = np.where(regimes == regime_id, 1, np.nan)
        regime_arrays.append(mask)
    
    # Combine into a 2D array
    regime_matrix = np.array(regime_arrays)
    
    # Plot as heatmap
    im = ax3.imshow(regime_matrix, aspect='auto', cmap='viridis', 
                 extent=[mdates.date2num(regimes.index[0]), mdates.date2num(regimes.index[-1]), -0.5, len(unique_regimes)-0.5])
    
    # Set y-ticks for regimes
    ax3.set_yticks(range(len(unique_regimes)))
    ax3.set_yticklabels([f'Regime {r}' for r in unique_regimes])
    
    # Format x-axis as dates
    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    ax3.set_title('Regime Distribution', fontsize=14)
    ax3.set_xlabel('Date', fontsize=12)
    
    # Add colorbar
    plt.colorbar(im, ax=ax3, orientation='vertical', label='Regime')
    
    # Calculate regime statistics
    regime_counts = regimes.value_counts()
    regime_pcts = regime_counts / len(regimes) * 100
    
    # Add regime statistics to bottom of plot
    stats_str = "Regime Distribution:\n"
    for regime_id in unique_regimes:
        if regime_id in regime_counts:
            stats_str += f"Regime {regime_id}: {regime_counts[regime_id]} points ({regime_pcts[regime_id]:.1f}%)\n"
    
    plt.figtext(0.01, 0.01, stats_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Format x-axis dates
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Regime distribution plot saved to {filename}")
    
    return fig

def plot_regime_aware_walk_forward_results(combined_results, section_results, global_regimes, config):
    """
    Plot results from regime-aware purged walk-forward optimization.
    
    Parameters:
        combined_results (DataFrame): Combined equity curve
        section_results (list): Results from each section
        global_regimes (Series): Precomputed regimes for the entire dataset
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
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Set window type string for title
    window_type_str = "Sliding Window" if window_type == 'sliding' else "Expanding Window"
    
    # Plot equity curve
    ax1 = axs[0]
    ax1.set_title(f'Regime-Aware Purged Walk-Forward Results ({window_type_str}) for {currency}', fontsize=16)
    
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
    
    # Highlight regimes if available
    if global_regimes is not None:
        # Add regime coloring to the background
        regime_colors = ['#E0F8FF', '#FFE0E0', '#E0FFE0', '#E0E0FF', '#FFF0E0']
        
        # Get regimes for the indices in combined_results
        regimes_subset = global_regimes.loc[combined_results.index]
        
        # Find regime change points
        changes = regimes_subset.diff().fillna(0) != 0
        change_indices = np.where(changes)[0]
        
        # Add starting point if needed
        if len(change_indices) == 0 or change_indices[0] != 0:
            change_indices = np.insert(change_indices, 0, 0)
        
        # Convert indices to dates
        change_points = [combined_results.index[i] for i in change_indices]
        
        # Add ending point if needed
        if change_points[-1] != combined_results.index[-1]:
            change_points.append(combined_results.index[-1])
        
        # Color regions by regime
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i+1]
            
            # Get the regime for this period
            if i < len(change_indices):
                regime = regimes_subset.iloc[change_indices[i]]
                # Use the color corresponding to this regime
                color_idx = int(regime) % len(regime_colors)
                ax1.axvspan(start, end, alpha=0.1, color=regime_colors[color_idx])
    
    # Format y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}x'))
    
    # Add legend
    from matplotlib.patches import Patch
    purge_patch = Patch(color='yellow', alpha=0.15, label='Purge Zone')
    embargo_patch = Patch(color='orange', alpha=0.15, label='Embargo Zone')
    handles, labels = ax1.get_legend_handles_labels()
    handles.extend([purge_patch, embargo_patch])
    labels.extend(['Purge Zone', 'Embargo Zone'])
    
    # Add regime patches to legend if regimes are available
    if global_regimes is not None:
        unique_regimes = sorted(global_regimes.unique())
        for regime_id in unique_regimes:
            color_idx = int(regime_id) % len(regime_colors)
            regime_patch = Patch(color=regime_colors[color_idx], alpha=0.3, label=f'Regime {regime_id}')
            handles.append(regime_patch)
            labels.append(f'Regime {regime_id}')
    
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
    
    # Plot regime distribution if available
    if global_regimes is not None:
        ax3 = axs[2]
        ax3.set_title('Regime Distribution', fontsize=14)
        
        # Get regimes for the indices in combined_results
        regimes_subset = global_regimes.loc[combined_results.index]
        
        # Create a colormap for regimes
        from matplotlib.colors import ListedColormap
        regime_cmap = ListedColormap(regime_colors[:len(np.unique(regimes_subset))])
        
        # Plot regimes as a filled step plot
        unique_regimes = sorted(np.unique(regimes_subset))
        
        # Create a step plot
        for i, regime_id in enumerate(unique_regimes):
            regime_mask = (regimes_subset == regime_id)
            ax3.fill_between(combined_results.index, 0, 1, where=regime_mask, 
                            color=regime_colors[i % len(regime_colors)], 
                            alpha=0.7, step='post', label=f'Regime {regime_id}')
        
        # Add section boundaries
        for test_start in section_boundaries:
            ax3.axvline(x=test_start, color='gray', linestyle='--', alpha=0.7)
        
        # Customize y-axis
        ax3.set_yticks([])
        ax3.set_yticklabels([])
        
        # Add legend
        ax3.legend(loc='upper right')
        
        # Annotate with regime statistics
        regime_counts = regimes_subset.value_counts()
        regime_pcts = regime_counts / len(regimes_subset) * 100
        
        # Create stats text
        stats_text = "Regime Distribution:\n"
        for regime_id in unique_regimes:
            if regime_id in regime_counts:
                stats_text += f"Regime {regime_id}: {regime_counts[regime_id]} ({regime_pcts[regime_id]:.1f}%)\n"
        
        ax3.text(0.02, 0.05, stats_text, transform=ax3.transAxes,
                bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        # Plot section-wise performance
        ax4 = axs[3]
        ax4.set_title('Performance by Section', fontsize=14)
        
        # Extract section performance metrics
        section_indices = []
        section_returns = []
        section_sharpes = []
        section_market_returns = []
        
        for i, section in enumerate(section_results):
            metrics = section.get('metrics', {})
            section_indices.append(i + 1)
            section_returns.append(metrics.get('total_return', 0) * 100)  # Convert to percentage
            section_sharpes.append(metrics.get('sharpe_ratio', 0))
            section_market_returns.append(section.get('buy_hold_return', 0) * 100)  # Convert to percentage
        
        # Plot as grouped bar chart
        width = 0.25
        x = np.array(section_indices)
        
        ax4.bar(x - width, section_returns, width, color='blue', label='Strategy Return (%)')
        ax4.bar(x, section_market_returns, width, color='red', label='Market Return (%)')
        ax4.bar(x + width, section_sharpes, width, color='green', label='Sharpe Ratio')
        
        # Add labels
        for i, (ret, mkt, sharpe) in enumerate(zip(section_returns, section_market_returns, section_sharpes)):
            ax4.annotate(f"{ret:.1f}%", xy=(x[i] - width, ret + 1), ha='center', fontsize=8)
            ax4.annotate(f"{mkt:.1f}%", xy=(x[i], mkt + 1), ha='center', fontsize=8)
            ax4.annotate(f"{sharpe:.2f}", xy=(x[i] + width, sharpe + 0.1), ha='center', fontsize=8)
        
        # Customize x-axis
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'S{idx}' for idx in section_indices])
        
        # Add legend and grid
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Label axes
        ax4.set_ylabel('Performance Metrics', fontsize=12)
        ax4.set_xlabel('Section', fontsize=12)
    else:
        # If no regimes, use the existing drawdown plot for the remaining space
        ax3 = axs[2]
        ax3.set_visible(False)
        ax4 = axs[3]
        ax4.set_visible(False)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    for ax in axs:
        if ax.get_visible():
            ax.xaxis.set_major_formatter(date_format)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(results_dir, f'regime_aware_walk_forward_results_{currency.replace("/", "_")}_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Regime-aware walk-forward results plot saved to {filename}")
    
    # Close figure
    plt.close(fig)

def analyze_regime_distributions(df, params, config):
    """
    Analyze and visualize regime distributions, transitions, and performance.
    
    Parameters:
        df (DataFrame): Results DataFrame
        params (dict): Parameters for each regime
        config (dict): Configuration
    """
    # Get config settings
    results_dir = config.get('STRATEGY_CONFIG', {}).get('RESULTS_DIR', 'enhanced_sma_results')
    currency = config.get('CURRENCY', 'BTC/USD')
    
    # Create directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Get regime time series
    regimes = df['regime'] if 'regime' in df.columns else df['precomputed_regime']
    
    # Calculate regime distribution
    regime_counts = regimes.value_counts()
    regime_pcts = regime_counts / len(regimes) * 100
    
    # Print regime distribution
    print("\nRegime Distribution:")
    for regime_id, count in regime_counts.items():
        print(f"Regime {regime_id}: {count} periods ({regime_pcts[regime_id]:.2f}%)")
    
    # Calculate regime transitions
    transitions = pd.crosstab(regimes.shift(), regimes, normalize='index') * 100
    
    # Print transition matrix
    print("\nRegime Transition Matrix (%):")
    print(transitions)
    
    # Calculate average regime duration
    durations = []
    current_regime = regimes.iloc[0]
    current_duration = 1
    
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current_regime:
            current_duration += 1
        else:
            durations.append((current_regime, current_duration))
            current_regime = regimes.iloc[i]
            current_duration = 1
    
    # Add the last regime
    durations.append((current_regime, current_duration))
    
    # Create a DataFrame with durations
    duration_df = pd.DataFrame(durations, columns=['regime', 'duration'])
    
    # Calculate statistics by regime
    avg_durations = duration_df.groupby('regime')['duration'].mean()
    max_durations = duration_df.groupby('regime')['duration'].max()
    
    # Print duration statistics
    print("\nRegime Duration Statistics (periods):")
    for regime_id in sorted(avg_durations.index):
        print(f"Regime {regime_id}: Avg={avg_durations[regime_id]:.2f}, Max={max_durations[regime_id]}")
    
    # Calculate performance by regime
    performance = []
    
    for regime_id in sorted(regimes.unique()):
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            regime_returns = df.loc[regime_mask, 'strategy_returns']
            market_returns = df.loc[regime_mask, 'returns']
            
            if len(regime_returns) > 0:
                # Calculate metrics
                regime_return = (1 + regime_returns).prod() - 1
                market_return = (1 + market_returns).prod() - 1
                
                # Calculate sharpe ratio if possible
                try:
                    from performance_metrics import calculate_sharpe_ratio
                    sharpe = calculate_sharpe_ratio(regime_returns)
                except:
                    sharpe = 0
                
                # Store performance data
                performance.append({
                    'regime': regime_id,
                    'return': regime_return * 100,  # Convert to percentage
                    'market_return': market_return * 100,  # Convert to percentage
                    'alpha': (regime_return - market_return) * 100,  # Convert to percentage
                    'sharpe': sharpe,
                    'count': regime_mask.sum(),
                    'percentage': regime_mask.sum() / len(regimes) * 100
                })
    
    # Create performance DataFrame
    performance_df = pd.DataFrame(performance)
    
    # Print performance by regime
    print("\nPerformance by Regime:")
    for _, row in performance_df.iterrows():
        print(f"Regime {int(row['regime'])}: Return={row['return']:.2f}%, "
              f"Market={row['market_return']:.2f}%, Alpha={row['alpha']:.2f}%, "
              f"Sharpe={row['sharpe']:.2f}")
    
    # Create visualization of regime distribution and transitions
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Regime Distribution (Pie Chart)
    ax1 = axs[0, 0]
    ax1.set_title('Regime Distribution', fontsize=14)
    
    # Create colors for regimes
    regime_colors = ['#4285F4', '#EA4335', '#34A853', '#FBBC05', '#8334E3']
    
    # Create labels with regime parameters
    labels = []
    for regime_id in sorted(regime_counts.index):
        if regime_id in params:
            short_window = params[regime_id].get('short_window', 'N/A')
            long_window = params[regime_id].get('long_window', 'N/A')
            labels.append(f"R{regime_id}: {regime_pcts[regime_id]:.1f}%\nSMA({short_window}/{long_window})")
        else:
            labels.append(f"Regime {regime_id}: {regime_pcts[regime_id]:.1f}%")
    
    # Create pie chart
    wedges, _, autotexts = ax1.pie(
        regime_counts, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=[regime_colors[int(r) % len(regime_colors)] for r in regime_counts.index]
    )
    
    # Enhance text properties
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    # Plot 2: Transition Matrix (Heatmap)
    ax2 = axs[0, 1]
    ax2.set_title('Regime Transition Matrix (%)', fontsize=14)
    
    # Create heatmap
    im = ax2.imshow(transitions, cmap='YlGnBu', vmin=0, vmax=100)
    
    # Add labels
    ax2.set_xticks(np.arange(len(transitions.columns)))
    ax2.set_yticks(np.arange(len(transitions.index)))
    ax2.set_xticklabels([f'R{r}' for r in transitions.columns])
    ax2.set_yticklabels([f'R{r}' for r in transitions.index])
    
    # Add values to cells
    for i in range(len(transitions.index)):
        for j in range(len(transitions.columns)):
            text = ax2.text(j, i, f"{transitions.iloc[i, j]:.1f}%",
                          ha="center", va="center", color="black")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Transition Probability (%)')
    
    # Customize axes
    ax2.set_xlabel('To Regime')
    ax2.set_ylabel('From Regime')
    
    # Plot 3: Regime Duration (Bar Chart)
    ax3 = axs[1, 0]
    ax3.set_title('Average Regime Duration', fontsize=14)
    
    # Create bar chart
    bars = ax3.bar(
        [f'R{r}' for r in avg_durations.index],
        avg_durations.values,
        color=[regime_colors[int(r) % len(regime_colors)] for r in avg_durations.index]
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}",
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Customize axes
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Average Duration (periods)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance by Regime (Multi-Bar Chart)
    ax4 = axs[1, 1]
    ax4.set_title('Performance by Regime', fontsize=14)
    
    if not performance_df.empty:
        # Create x positions
        x = np.arange(len(performance_df))
        width = 0.2
        
        # Create grouped bar chart
        ax4.bar(x - width*1.5, performance_df['return'], width, color='blue', label='Strategy Return (%)')
        ax4.bar(x - width/2, performance_df['market_return'], width, color='red', label='Market Return (%)')
        ax4.bar(x + width/2, performance_df['alpha'], width, color='green', label='Alpha (%)')
        ax4.bar(x + width*1.5, performance_df['sharpe'], width, color='purple', label='Sharpe Ratio')
        
        # Add value labels
        for i, (idx, row) in enumerate(performance_df.iterrows()):
            ax4.annotate(f"{row['return']:.1f}%",
                        xy=(i - width*1.5, row['return'] + 0.5),
                        ha='center', fontsize=8)
            ax4.annotate(f"{row['market_return']:.1f}%",
                        xy=(i - width/2, row['market_return'] + 0.5),
                        ha='center', fontsize=8)
            ax4.annotate(f"{row['alpha']:.1f}%",
                        xy=(i + width/2, row['alpha'] + 0.5),
                        ha='center', fontsize=8)
            ax4.annotate(f"{row['sharpe']:.2f}",
                        xy=(i + width*1.5, row['sharpe'] + 0.1),
                        ha='center', fontsize=8)
        
        # Customize x-axis
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'R{int(r)}' for r in performance_df['regime']])
        
        # Add legend and grid
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Label axes
        ax4.set_ylabel('Performance Metrics')
        ax4.set_xlabel('Regime')
    else:
        ax4.text(0.5, 0.5, "No performance data available", 
                ha='center', va='center', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(results_dir, f'regime_analysis_{currency.replace("/", "_")}_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Create a text report
    report_filename = os.path.join(results_dir, f'regime_analysis_{currency.replace("/", "_")}_{timestamp}.txt')
    
    with open(report_filename, 'w') as f:
        f.write("===== REGIME ANALYSIS REPORT =====\n\n")
        f.write(f"Currency: {currency}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("REGIME DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for regime_id, count in regime_counts.items():
            f.write(f"Regime {regime_id}: {count} periods ({regime_pcts[regime_id]:.2f}%)\n")
        
        f.write("\nREGIME TRANSITION MATRIX (%):\n")
        f.write("-" * 40 + "\n")
        f.write(transitions.to_string() + "\n\n")
        
        f.write("REGIME DURATION STATISTICS (periods):\n")
        f.write("-" * 40 + "\n")
        for regime_id in sorted(avg_durations.index):
            f.write(f"Regime {regime_id}: Avg={avg_durations[regime_id]:.2f}, Max={max_durations[regime_id]}\n")
        
        f.write("\nPERFORMANCE BY REGIME:\n")
        f.write("-" * 40 + "\n")
        for _, row in performance_df.iterrows():
            f.write(f"Regime {int(row['regime'])}:\n")
            f.write(f"  - Strategy Return: {row['return']:.2f}%\n")
            f.write(f"  - Market Return: {row['market_return']:.2f}%\n")
            f.write(f"  - Alpha: {row['alpha']:.2f}%\n")
            f.write(f"  - Sharpe Ratio: {row['sharpe']:.2f}\n")
            
            # Add parameter info if available
            if int(row['regime']) in params:
                regime_params = params[int(row['regime'])]
                f.write("  - Parameters:\n")
                for key, value in regime_params.items():
                    f.write(f"    * {key}: {value}\n")
            f.write("\n")
        
        f.write("\nREGIME PARAMETER SUMMARY:\n")
        f.write("-" * 40 + "\n")
        for regime_id, regime_params in params.items():
            f.write(f"Regime {regime_id} Parameters:\n")
            # Extract key parameters
            short_window = regime_params.get('short_window', 'N/A')
            long_window = regime_params.get('long_window', 'N/A')
            trend_threshold = regime_params.get('trend_strength_threshold', 'N/A')
            target_vol = regime_params.get('target_vol', 'N/A')
            max_dd_exit = regime_params.get('max_drawdown_exit', 'N/A')
            
            f.write(f"  - SMA Parameters: Short={short_window}, Long={long_window}, Trend={trend_threshold}\n")
            f.write(f"  - Risk Parameters: Target Vol={target_vol}, Max DD={max_dd_exit}\n\n")
    
    print(f"Regime analysis saved to {filename} and {report_filename}")
    
    return fig

def plot_enhanced_results_with_regimes(result_df, params, metrics, global_regimes=None, config=None):
    """
    Plot enhanced results with regime information.
    Updated to include detailed regime visualization and global regime overlay.
    
    Parameters:
        result_df (DataFrame): Results DataFrame
        params (dict): Parameters for each regime
        metrics (dict): Performance metrics
        global_regimes (Series, optional): Precomputed regimes for the entire dataset
        config (dict, optional): Configuration
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
    regime_params = params if isinstance(params, dict) else params.get('regime_params', {})
    
    # Check if regime data exists
    if 'regime' not in result_df.columns and global_regimes is None:
        print("No regime data found. Using standard visualization.")
        plot_enhanced_results_regime_specific(result_df, {'regime_params': regime_params}, metrics)
        return
    
    # Use global regimes if provided, otherwise use the regime column from the dataframe
    if global_regimes is not None:
        regimes = global_regimes.loc[result_df.index]
        result_df['regime'] = regimes
    else:
        regimes = result_df['regime']
    
    # Check if there were any trades
    position_changes = result_df['managed_position'].diff().fillna(0).abs() if 'managed_position' in result_df.columns else 0
    num_trades = int((position_changes != 0).sum()) if isinstance(position_changes, pd.Series) else 0
    
    # Create figure with subplots for regime-aware visualization
    fig, axs = plt.subplots(7, 1, figsize=(14, 28), 
                           gridspec_kw={'height_ratios': [2, 1, 1, 1, 1, 1, 1]})
    
    # -------------------------------------------------------------------------
    # Plot 1: Price and Performance with Regime Background
    # -------------------------------------------------------------------------
    ax1 = axs[0]
    ax1.set_title(f'Enhanced Strategy for {CURRENCY} with Regime-Aware Parameters', fontsize=16)
    
    # Plot price
    ax1.plot(result_df.index, result_df['close_price'] if 'close_price' in result_df.columns else result_df['close'], 
            color='gray', alpha=0.6, label='Price')
    
    # Create twin axis for portfolio value
    ax1_twin = ax1.twinx()
    
    # Plot strategy and buy & hold equity curves
    ax1_twin.plot(result_df.index, result_df['strategy_cumulative'] * INITIAL_CAPITAL, 'b-', label='Strategy')
    ax1_twin.plot(result_df.index, result_df['buy_hold_cumulative'] * INITIAL_CAPITAL, 'r--', label='Market Buy & Hold')
    
    # Color background by regime
    regime_colors = ['#E0F8FF', '#FFE0E0', '#E0FFE0', '#E0E0FF', '#FFF0E0']
    
    # Identify regime change points
    changes = regimes.diff().fillna(0) != 0
    change_points = regimes.index[changes].tolist()
    change_points = [regimes.index[0]] + change_points + [regimes.index[-1]]
    
    # Color regions by regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(regime_colors)
        ax1.axvspan(start, end, alpha=0.2, color=regime_colors[color_idx])
    
    # Add regime labels
    unique_regimes = sorted(regimes.unique())
    for regime_id in unique_regimes:
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            color_idx = int(regime_id) % len(regime_colors)
            label = f"Regime {regime_id}"
            # Add a small colored rectangle for the legend
            ax1.plot([], [], color=regime_colors[color_idx], alpha=0.6, linewidth=10, label=label)
    
    # Customize axes labels
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Portfolio Value ($)')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 2: Volatility, Regimes, and Parameters
    # -------------------------------------------------------------------------
    ax2 = axs[1]
    ax2.set_title('Volatility and Regime-Specific Parameters', fontsize=14)
    
    # Plot volatility
    if 'volatility' in result_df.columns:
        ax2.plot(result_df.index, result_df['volatility'], 'b-', label='Volatility')
    
    # Color background by regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Extract key parameters for this regime
        regime_id = int(regime)
        if regime_id in regime_params:
            params_str = regime_params_to_str(regime_params[regime_id])
        else:
            params_str = f"Regime {regime_id}"
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(regime_colors)
        
        # Add colored background
        ax2.axvspan(start, end, alpha=0.2, color=regime_colors[color_idx])
        
        # Add parameter text in the middle of each regime span
        mid_point = start + (end - start) / 2
        
        # Only add text for sufficiently long regimes
        if (end - start).total_seconds() > (result_df.index[-1] - result_df.index[0]).total_seconds() * 0.05:
            ax2.text(mid_point, ax2.get_ylim()[1] * 0.9, params_str, 
                   ha='center', va='top', fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 3: Trading Signals and Position
    # -------------------------------------------------------------------------
    ax3 = axs[2]
    ax3.set_title('Signals, Positions and Regime-Specific Parameters', fontsize=14)
    
    # Plot signals if available
    if 'raw_signal' in result_df.columns:
        ax3.plot(result_df.index, result_df['raw_signal'], 'k--', alpha=0.5, label='Raw Signal')
    
    if 'filtered_signal' in result_df.columns:
        ax3.plot(result_df.index, result_df['filtered_signal'], 'g-', alpha=0.7, label='Filtered Signal')
    
    if 'managed_position' in result_df.columns:
        ax3.plot(result_df.index, result_df['managed_position'], 'b-', linewidth=1.5, label='Final Position')
    
    # Color background by regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(regime_colors)
        ax3.axvspan(start, end, alpha=0.2, color=regime_colors[color_idx])
    
    # Add threshold lines for each regime
    for regime_id in unique_regimes:
        if regime_id in regime_params:
            threshold = regime_params[regime_id].get('trend_strength_threshold', 0.3)
            if isinstance(threshold, list):
                threshold = threshold[0]
            
            # Calculate the average position for this regime
            regime_mask = (regimes == regime_id)
            if regime_mask.any() and 'managed_position' in result_df.columns:
                avg_pos = result_df.loc[regime_mask, 'managed_position'].mean()
                
                # Add horizontal line at this level
                color_idx = int(regime_id) % len(regime_colors)
                color = regime_colors[color_idx]
                
                ax3.axhline(y=avg_pos, color=color, linestyle='--', alpha=0.5,
                          label=f'Avg. Position R{regime_id}: {avg_pos:.2f}')
    
    # Reference lines at 0, 1, -1
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    
    ax3.set_ylabel('Position')
    ax3.legend(loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 4: Regime-Specific Performance
    # -------------------------------------------------------------------------
    ax4 = axs[3]
    ax4.set_title('Regime-Specific Performance', fontsize=14)
    
    # Calculate performance by regime
    regime_returns = []
    regime_market_returns = []
    regime_alphas = []
    regime_sharpes = []
    regime_labels = []
    
    for regime_id in unique_regimes:
        regime_mask = (regimes == regime_id)
        if regime_mask.any():
            regime_strategy_returns = result_df.loc[regime_mask, 'strategy_returns']
            regime_market_returns_data = result_df.loc[regime_mask, 'returns']
            
            if len(regime_strategy_returns) > 0:
                regime_return = (1 + regime_strategy_returns).prod() - 1
                regime_market_return = (1 + regime_market_returns_data).prod() - 1
                regime_alpha = regime_return - regime_market_return
                
                try:
                    from performance_metrics import calculate_sharpe_ratio
                    regime_sharpe = calculate_sharpe_ratio(regime_strategy_returns)
                except:
                    regime_sharpe = 0
                
                regime_returns.append(regime_return * 100)  # Convert to percentage
                regime_market_returns.append(regime_market_return * 100)
                regime_alphas.append(regime_alpha * 100)
                regime_sharpes.append(regime_sharpe)
                regime_labels.append(f'Regime {regime_id}')
    
    # Create bar chart with 4 grouped bars (strategy return, market return, alpha, sharpe)
    width = 0.2
    x = np.arange(len(regime_labels))
    
    ax4.bar(x - width*1.5, regime_returns, width, color='blue', alpha=0.7, label='Strategy Return (%)')
    ax4.bar(x - width/2, regime_market_returns, width, color='red', alpha=0.7, label='Market Return (%)')
    ax4.bar(x + width/2, regime_alphas, width, color='green', alpha=0.7, label='Alpha (%)')
    ax4.bar(x + width*1.5, regime_sharpes, width, color='purple', alpha=0.7, label='Sharpe Ratio')
    
    # Add value labels
    for i in range(len(regime_labels)):
        if i < len(regime_returns):
            ax4.text(i - width*1.5, regime_returns[i] + 0.5, f'{regime_returns[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_market_returns):
            ax4.text(i - width/2, regime_market_returns[i] + 0.5, f'{regime_market_returns[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_alphas):
            ax4.text(i + width/2, regime_alphas[i] + 0.5, f'{regime_alphas[i]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        if i < len(regime_sharpes):
            ax4.text(i + width*1.5, regime_sharpes[i] + 0.5, f'{regime_sharpes[i]:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    # Customize x-axis
    ax4.set_xticks(x)
    ax4.set_xticklabels(regime_labels)
    
    # Add parameter info to the bars
    for i, regime_id in enumerate(unique_regimes):
        if regime_id in regime_params and i < len(x):
            params = regime_params[regime_id]
            short_window = params.get('short_window', 13)
            long_window = params.get('long_window', 55)
            
            if i < len(regime_returns):
                ax4.text(i, max(regime_returns[i], regime_market_returns[i], regime_alphas[i], regime_sharpes[i]) + 2, 
                        f'SMA({short_window}/{long_window})', 
                        ha='center', va='bottom', rotation=0, fontsize=8)
    
    ax4.set_ylabel('Performance Metrics')
    ax4.legend(loc='upper left')
    
    # -------------------------------------------------------------------------
    # Plot 5: Drawdown and Risk Management
    # -------------------------------------------------------------------------
    ax5 = axs[4]
    ax5.set_title(f'Drawdown (Max: {metrics["max_drawdown"]:.2%})', fontsize=14)
    
    # Calculate drawdown
    drawdown = result_df['strategy_cumulative'] / result_df['strategy_cumulative'].cummax() - 1
    
    # Plot drawdown
    ax5.fill_between(result_df.index, drawdown * 100, 0, color='red', alpha=0.3)
    
    # Color background by regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(regime_colors)
        ax5.axvspan(start, end, alpha=0.1, color=regime_colors[color_idx])
    
    # Add stop loss lines for each regime
    for regime_id in unique_regimes:
        if regime_id in regime_params:
            max_dd_exit = regime_params[regime_id].get('max_drawdown_exit', 0.15)
            if isinstance(max_dd_exit, list):
                max_dd_exit = max_dd_exit[0]
            
            # Add horizontal line at this level
            color_idx = int(regime_id) % len(regime_colors)
            color = regime_colors[color_idx]
            
            ax5.axhline(y=max_dd_exit * 100 * -1, color=color, linestyle='--', alpha=0.5,
                      label=f'Stop Loss R{regime_id}: {max_dd_exit * 100:.0f}%')
    
    ax5.set_ylabel('Drawdown (%)')
    ax5.legend(loc='lower left')
    
    # -------------------------------------------------------------------------
    # Plot 6: Regime Transitions
    # -------------------------------------------------------------------------
    ax6 = axs[5]
    ax6.set_title('Regime Transitions and Durations', fontsize=14)
    
    # Plot regimes over time
    ax6.step(regimes.index, regimes, where='post', color='blue', alpha=0.7)
    
    # Add colored background for each regime
    for i in range(len(change_points) - 1):
        start = change_points[i]
        end = change_points[i+1]
        regime = regimes.loc[start]
        
        # Use the color corresponding to this regime
        color_idx = int(regime) % len(regime_colors)
        ax6.axvspan(start, end, alpha=0.2, color=regime_colors[color_idx])
    
    # Calculate regime durations
    durations = []
    current_regime = regimes.iloc[0]
    current_start = regimes.index[0]
    
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime:
            duration = (regimes.index[i] - current_start).total_seconds() / 3600  # hours
            durations.append({
                'regime': current_regime,
                'start': current_start,
                'end': regimes.index[i],
                'duration': duration
            })
            current_regime = regimes.iloc[i]
            current_start = regimes.index[i]
    
    # Add the last regime
    duration = (regimes.index[-1] - current_start).total_seconds() / 3600  # hours
    durations.append({
        'regime': current_regime,
        'start': current_start,
        'end': regimes.index[-1],
        'duration': duration
    })
    
    # Add duration labels for longer regimes
    for dur in durations:
        # Only add text for sufficiently long regimes (more than 5% of the total time)
        total_duration = (regimes.index[-1] - regimes.index[0]).total_seconds() / 3600
        if dur['duration'] > total_duration * 0.05:
            mid_point = dur['start'] + (dur['end'] - dur['start']) / 2
            ax6.text(mid_point, dur['regime'], f"{dur['duration']:.1f}h", 
                   ha='center', va='bottom', fontsize=8)
    
    # Customize y-axis
    ax6.set_yticks(unique_regimes)
    ax6.set_yticklabels([f'Regime {r}' for r in unique_regimes])
    
    ax6.set_ylabel('Regime')
    
    # -------------------------------------------------------------------------
    # Plot 7: Parameter Summary
    # -------------------------------------------------------------------------
    ax7 = axs[6]
    ax7.set_title('Regime-Specific Parameter Summary', fontsize=14)
    ax7.axis('off')  # Turn off axis
    
    # Create parameter summary table
    table_data = []
    header = ['Regime', 'Short MA', 'Long MA', 'Trend Thresh', 'Target Vol', 'Max DD Exit', 'Profit Taking']
    table_data.append(header)
    
    for regime_id in unique_regimes:
        if regime_id in regime_params:
            params = regime_params[regime_id]
            row = [
                f'Regime {regime_id}',
                str(params.get('short_window', 'N/A')),
                str(params.get('long_window', 'N/A')),
                f"{params.get('trend_strength_threshold', 'N/A')}",
                f"{params.get('target_vol', 'N/A')}",
                f"{params.get('max_drawdown_exit', 'N/A')}",
                f"{params.get('profit_taking_threshold', 'N/A')}"
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
        f"R{regime_id}: SMA({params.get('short_window', 'N/A')}/{params.get('long_window', 'N/A')})"
        for regime_id, params in regime_params.items()
    ])
    
    plt.figtext(
        0.1, 0.01, 
        (
            f"Return: {metrics['total_return']:.2%} | Annual: {metrics.get('annualized_return', 0):.2%} | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Sortino: {metrics.get('sortino_ratio', 0):.2f} | "
            f"Calmar: {metrics.get('calmar_ratio', 0):.2f} | MaxDD: {metrics.get('max_drawdown', 0):.2%}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.2%} | Gain/Pain: {metrics.get('gain_to_pain', 0):.2f} | "
            f"Volatility: {metrics.get('volatility', 0):.2%} | "
            f"Buy & Hold: {result_df['buy_hold_cumulative'].iloc[-1] - 1:.2%} | "
            f"Alpha: {metrics['total_return'] - (result_df['buy_hold_cumulative'].iloc[-1] - 1):.2%}\n"
            f"Regime Parameters: {regime_summary}\n"
            + (f"\nWARNING: No trades executed during backtest period" if num_trades == 0 else "")
        ),
        ha='left', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Adjust bottom margin to fit the additional info
    
    # Save the figure
    plot_filename = os.path.join(RESULTS_DIR, f'enhanced_regime_aware_results_{CURRENCY.replace("/", "_")}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Regime-aware results plot saved to {plot_filename}")

def regime_params_to_str(params):
    """
    Convert regime parameters to a short string representation.
    
    Parameters:
        params (dict): Regime parameters
        
    Returns:
        str: Short string representation
    """
    short_window = params.get('short_window', 'N/A')
    long_window = params.get('long_window', 'N/A')
    trend_threshold = params.get('trend_strength_threshold', 'N/A')
    
    return f"SMA({short_window}/{long_window}), Trend={trend_threshold}"

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