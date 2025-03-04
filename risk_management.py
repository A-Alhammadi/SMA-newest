# risk_management.py
# Functions for position sizing, risk management, and trade execution

import pandas as pd
import numpy as np
from numba import njit

def calculate_adaptive_position_size(volatility, target_vol=0.15, max_size=1.0, min_size=0.1):
    """
    Scale position size inversely with volatility using a continuous function.
    
    Parameters:
        volatility (Series): Volatility series
        target_vol (float): Target annualized volatility
        max_size (float): Maximum position size
        min_size (float): Minimum position size
        
    Returns:
        Series: Position size scaling factor
    """
    # Avoid division by zero
    safe_volatility = volatility.replace(0, volatility.median())
    
    # Calculate position scale based on volatility
    position_scale = target_vol / safe_volatility
    
    # Apply limits
    position_scale = position_scale.clip(lower=min_size, upper=max_size)
    
    return position_scale

def apply_rebalancing_schedule(position_scale, timestamp, frequency="daily", materiality_threshold=0.05):
    """
    Apply a rebalancing schedule to position sizes to reduce trading frequency.
    
    Parameters:
        position_scale (Series): Original position scale series
        timestamp (DatetimeIndex): Timestamps for the series
        frequency (str): Rebalancing frequency - "daily", "weekly", or "hourly"
        materiality_threshold (float): Only rebalance if position change exceeds this percentage
        
    Returns:
        Series: Rebalanced position scale series
    """
    # Create a copy to avoid modifying the original
    rebalanced_scale = position_scale.copy()
    
    # Initialize with the first position
    current_position = position_scale.iloc[0]
    last_rebalance_time = None
    
    # Go through each timestamp
    for i, (ts, new_position) in enumerate(zip(timestamp, position_scale)):
        # Skip the first position as it's already set
        if i == 0:
            last_rebalance_time = ts
            continue
        
        # Determine if it's time to rebalance
        rebalance_time = False
        
        if frequency == "daily":
            # Rebalance at the beginning of each day
            if ts.date() != last_rebalance_time.date():
                rebalance_time = True
        elif frequency == "weekly":
            # Rebalance at the beginning of each week
            if ts.isocalendar()[1] != last_rebalance_time.isocalendar()[1]:
                rebalance_time = True
        elif frequency == "hourly":
            # Rebalance every N hours
            rebalance_hours = 4  # Adjust as needed
            hour_diff = (ts - last_rebalance_time).total_seconds() / 3600
            if hour_diff >= rebalance_hours:
                rebalance_time = True
        
        # Special case: if BOTH current and new positions are zero, don't consider it a change
        if current_position == 0 and new_position == 0:
            material_change = False
        # Special case: if going from zero to non-zero, always rebalance
        elif current_position == 0 and new_position != 0:
            material_change = True
        # Special case: if going from non-zero to zero, always rebalance
        elif current_position != 0 and new_position == 0:
            material_change = True
        # Normal case: calculate percentage change
        else:
            position_change_pct = abs(new_position - current_position) / current_position
            material_change = position_change_pct > materiality_threshold
        
        # Rebalance if it's time and the change is material
        if rebalance_time and material_change:
            current_position = new_position
            last_rebalance_time = ts
        else:
            # Keep the previous position
            rebalanced_scale.iloc[i] = current_position
    
    return rebalanced_scale

def calculate_adaptive_position_size_with_schedule(volatility, regime, timestamp, 
                                                  target_vol=0.15, max_size=1.0, min_size=0.1,
                                                  rebalance_frequency="daily", 
                                                  materiality_threshold=0.05,
                                                  regime_opt_out=None,
                                                  regime_buy_hold=None):
    """
    Scale position size inversely with volatility using a scheduled rebalancing approach.
    Position sizing is regime-aware and can:
    - Opt out of trading in specified regimes
    - Switch to buy & hold in specified regimes
    
    Parameters:
        volatility (Series): Volatility series
        regime (Series): Regime classifications
        timestamp (DatetimeIndex): Timestamps for the series
        target_vol (float): Target annualized volatility
        max_size (float): Maximum position size
        min_size (float): Minimum position size
        rebalance_frequency (str): How often to rebalance - "daily", "weekly", or "hourly"
        materiality_threshold (float): Only rebalance if position size change exceeds this percentage
        regime_opt_out (dict): Dictionary specifying which regimes to opt out from trading (True = opt out)
        regime_buy_hold (dict): Dictionary specifying which regimes to switch to buy & hold (True = buy & hold)
        
    Returns:
        Series: Position size scaling factor and buy & hold mask
    """
    # Print debugging info
    print("\nPOSITION SIZING DEBUG:")
    print(f"Target volatility: {target_vol}")
    print(f"Min size: {min_size}, Max size: {max_size}")
    print(f"Materiality threshold: {materiality_threshold}")
    
    # Create a buy & hold mask (True where buy & hold should be applied)
    buy_hold_mask = pd.Series(False, index=regime.index)
    
    # Apply buy & hold for specific regimes if specified
    if regime_buy_hold is not None:
        for r, use_buy_hold in regime_buy_hold.items():
            if use_buy_hold:
                # Mark this regime for buy & hold
                buy_hold_mask[regime == r] = True
                regime_count = (regime == r).sum()
                print(f"Applied buy & hold for regime {r}: {regime_count} periods affected")
    
    # Avoid division by zero
    safe_volatility = volatility.replace(0, volatility.median())
    
    # Calculate base position scale based on volatility
    position_scale = target_vol / safe_volatility
    
    # Create regime adjustment factors - reduce position in higher volatility regimes
    # but don't completely eliminate trading unless specified in regime_opt_out
    regime_factors = pd.Series(1.0, index=regime.index)
    regime_factors[regime == 1] = 0.8  # Normal volatility: 80% normal position size
    regime_factors[regime == 2] = 0.5  # High volatility: 50% normal position size
    
    # Debug regime factors
    print("\nREGIME ADJUSTMENT FACTORS:")
    for r in range(3):  # Assuming 3 regimes
        regime_count = (regime == r).sum()
        if regime_count > 0:
            avg_factor = regime_factors[regime == r].mean()
            print(f"Regime {r}: Average factor = {avg_factor:.2f}")
    
    # Apply opt-out for specific regimes if specified
    if regime_opt_out is not None:
        # Debug before opt-out
        zero_before = (position_scale == 0).sum()
        print(f"\nZero positions before opt-out: {zero_before}")
        
        for r, opt_out in regime_opt_out.items():
            if opt_out:
                # Set position size to 0 for opted-out regimes
                regime_factors[regime == r] = 0.0
                regime_count = (regime == r).sum()
                print(f"Applied opt-out for regime {r}: {regime_count} periods affected")
    
    # Apply regime adjustment
    position_scale = position_scale * regime_factors
    
    # Debug after regime adjustment
    zero_after_regime = (position_scale == 0).sum()
    print(f"Zero positions after regime adjustment: {zero_after_regime}")
    
    # Apply limits
    position_scale = position_scale.clip(lower=min_size, upper=max_size)
    
    # Debug after clipping
    zero_after_clip = (position_scale == 0).sum()
    print(f"Zero positions after clipping: {zero_after_clip}")
    
    # Set position size to 0 for opted-out regimes (need to do this after clipping)
    if regime_opt_out is not None:
        for r, opt_out in regime_opt_out.items():
            if opt_out:
                position_scale[regime == r] = 0.0
        
        # Debug after final opt-out
        zero_final = (position_scale == 0).sum()
        print(f"Zero positions after final opt-out: {zero_final}")
    
    # Apply scheduled rebalancing to reduce trading frequency
    rebalanced_scale = apply_rebalancing_schedule(position_scale, timestamp, 
                                                 rebalance_frequency, 
                                                 materiality_threshold)
    
    # Debug rebalancing effect
    zero_after_rebalance = (rebalanced_scale == 0).sum()
    print(f"Zero positions after rebalancing: {zero_after_rebalance}")
    
    # Print buy & hold info
    if regime_buy_hold is not None:
        buy_hold_count = buy_hold_mask.sum()
        print(f"\nBuy & Hold strategy applied to {buy_hold_count} periods ({buy_hold_count/len(regime)*100:.2f}%)")
    
    # Final position size stats
    print(f"\nFinal position sizing stats:")
    print(f"Min: {rebalanced_scale.min()}")
    print(f"Max: {rebalanced_scale.max()}")
    print(f"Mean: {rebalanced_scale.mean():.4f}")
    print(f"Zero positions: {(rebalanced_scale == 0).sum()} out of {len(rebalanced_scale)}")
    
    return rebalanced_scale, buy_hold_mask

def apply_trailing_stop(df, position, returns, activation_threshold=0.03, stop_distance=0.02):
    """
    Apply trailing stop loss to protect profits.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        position (Series): Position series (-1, 0, 1)
        returns (Series): Returns series
        activation_threshold (float): Profit threshold to activate trailing stop
        stop_distance (float): Distance to maintain trailing stop
        
    Returns:
        Series: Position with trailing stop applied
    """
    modified_position = position.copy()
    
    # Calculate cumulative returns for each trade
    trade_returns = pd.Series(0.0, index=position.index)
    running_return = 0.0
    entry_price = None
    
    for i in range(len(position)):
        # New position
        if i > 0 and position.iloc[i] != 0 and position.iloc[i-1] == 0:
            entry_price = df['close_price'].iloc[i]
            running_return = 0.0
        
        # Maintaining position
        elif i > 0 and position.iloc[i] != 0 and position.iloc[i] == position.iloc[i-1]:
            running_return = (running_return + 1) * (1 + returns.iloc[i]) - 1
        
        # Closed position
        elif i > 0 and position.iloc[i] == 0 and position.iloc[i-1] != 0:
            running_return = 0.0
            entry_price = None
        
        trade_returns.iloc[i] = running_return
    
    # Track highest return achieved during each trade
    highest_return = trade_returns.copy()
    for i in range(1, len(trade_returns)):
        if position.iloc[i] == position.iloc[i-1] and position.iloc[i] != 0:
            highest_return.iloc[i] = max(highest_return.iloc[i-1], trade_returns.iloc[i])
    
    # Apply trailing stop
    for i in range(1, len(position)):
        # Check if in a position and trailing stop is activated
        if position.iloc[i] != 0 and highest_return.iloc[i] >= activation_threshold:
            # Calculate drawdown from highest point
            drawdown = (trade_returns.iloc[i] - highest_return.iloc[i])
            
            # Close position if trailing stop is hit
            if drawdown < -stop_distance:
                modified_position.iloc[i] = 0
    
    return modified_position

def apply_stop_loss(position, returns, trade_returns, max_drawdown=0.15):
    """
    Apply maximum drawdown stop loss.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        returns (Series): Returns series
        trade_returns (Series): Cumulative returns for each trade
        max_drawdown (float): Maximum allowed drawdown
        
    Returns:
        Series: Position with stop loss applied
    """
    modified_position = position.copy()
    
    # Track highest return achieved during each trade
    highest_return = pd.Series(0.0, index=position.index)
    
    for i in range(1, len(position)):
        if position.iloc[i] == position.iloc[i-1] and position.iloc[i] != 0:
            highest_return.iloc[i] = max(highest_return.iloc[i-1], trade_returns.iloc[i])
        else:
            highest_return.iloc[i] = trade_returns.iloc[i]
    
    # Apply stop loss
    for i in range(1, len(position)):
        if position.iloc[i] != 0:
            # Calculate drawdown from highest point
            drawdown = (trade_returns.iloc[i] - highest_return.iloc[i])
            
            # Close position if drawdown exceeds maximum
            if drawdown < -max_drawdown:
                modified_position.iloc[i] = 0
    
    return modified_position

def apply_profit_taking(position, trade_returns, profit_threshold=0.05):
    """
    Apply profit taking rule.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        trade_returns (Series): Cumulative returns for each trade
        profit_threshold (float): Profit threshold to take profits
        
    Returns:
        Series: Position with profit taking applied
    """
    modified_position = position.copy()
    
    # Apply profit taking
    for i in range(1, len(position)):
        if position.iloc[i] != 0 and trade_returns.iloc[i] >= profit_threshold:
            modified_position.iloc[i] = 0
    
    return modified_position

def apply_unified_risk_management(df, position, returns, volatility, regimes, config):
    """
    Apply a unified risk management approach that prioritizes exit conditions.
    
    Priority order:
    1. Maximum drawdown stop loss (prevent catastrophic losses)
    2. Profit taking (secure profits at target)
    3. Trailing stop (protect profits while allowing upside)
    
    Parameters:
        df (DataFrame): DataFrame with price data
        position (Series): Position series (-1, 0, 1)
        returns (Series): Returns series
        volatility (Series): Volatility series
        regimes (Series): Regime classifications
        config (dict): Risk management configuration
        
    Returns:
        Series: Position with risk management applied
        dict: Statistics about which exit conditions were triggered
    """
    # Create copy of position for modification
    managed_position = position.copy()
    
    # Track trade stats
    trade_stats = {
        'max_drawdown_exits': 0,
        'profit_taking_exits': 0,
        'trailing_stop_exits': 0,
        'total_trades': 0
    }
    
    # Track trade information
    in_position = False
    entry_price = None
    entry_time = None
    highest_price = None
    lowest_price = None
    trade_return = 0.0
    trailing_stop_activated = False
    
    # Set regime-specific risk parameters
    regime_risk_params = {
        0: {  # Low volatility regime - more conservative profit taking
            'profit_mult': 0.8,  # 80% of standard profit taking
            'trailing_mult': 1.2  # 120% of standard trailing stop distance
        },
        1: {  # Medium volatility regime - standard parameters
            'profit_mult': 1.0,
            'trailing_mult': 1.0
        },
        2: {  # High volatility regime - tighter risk management
            'profit_mult': 1.2,  # 120% of standard profit taking (higher target)
            'trailing_mult': 0.8  # 80% of standard trailing stop distance (tighter)
        }
    }
    
    # Process each bar
    for i in range(1, len(position)):
        current_price = df['close_price'].iloc[i]
        current_regime = regimes.iloc[i]
        
        # Skip if no position
        if position.iloc[i] == 0:
            in_position = False
            continue
        
        # Get regime-specific parameters
        regime_params = regime_risk_params.get(current_regime, regime_risk_params[1])
        
        # New position initiation
        if not in_position or position.iloc[i] != position.iloc[i-1]:
            in_position = True
            entry_price = current_price
            entry_time = df.index[i]
            highest_price = current_price
            lowest_price = current_price
            trade_return = 0.0
            trailing_stop_activated = False
            trade_stats['total_trades'] += 1
            continue
        
        # Update highest/lowest price if in a position
        highest_price = max(highest_price, current_price)
        lowest_price = min(lowest_price, current_price)
        
        # Calculate trade return based on position direction
        if position.iloc[i] > 0:  # Long position
            price_change = (current_price / entry_price) - 1
        else:  # Short position
            price_change = 1 - (current_price / entry_price)
        
        # Update trade return
        trade_return = price_change * abs(position.iloc[i])  # Adjust for position size
        
        # 1. Check maximum drawdown - highest priority
        drawdown_from_peak = 0
        if position.iloc[i] > 0:  # Long position
            drawdown_from_peak = (highest_price - current_price) / highest_price
        else:  # Short position
            drawdown_from_peak = (current_price - lowest_price) / lowest_price if lowest_price > 0 else 0
        
        max_drawdown_threshold = config['max_drawdown_exit']
        if drawdown_from_peak > max_drawdown_threshold:
            managed_position.iloc[i] = 0
            trade_stats['max_drawdown_exits'] += 1
            in_position = False
            continue
        
        # 2. Check profit taking threshold
        profit_threshold = config['profit_taking_threshold'] * regime_params['profit_mult']
        if trade_return > profit_threshold:
            managed_position.iloc[i] = 0
            trade_stats['profit_taking_exits'] += 1
            in_position = False
            continue
        
        # 3. Check trailing stop
        trailing_activation = config['trailing_stop_activation']
        if trade_return > trailing_activation and not trailing_stop_activated:
            trailing_stop_activated = True
        
        if trailing_stop_activated:
            # Calculate dynamic trailing stop distance based on volatility and regime
            base_distance = config['trailing_stop_distance']
            vol_factor = min(2.0, max(0.5, volatility.iloc[i] / volatility.mean()))
            regime_factor = regime_params['trailing_mult']
            dynamic_distance = base_distance * vol_factor * regime_factor
            
            # Calculate price levels for trailing stop
            if position.iloc[i] > 0:  # Long position
                stop_level = highest_price * (1 - dynamic_distance)
                if current_price < stop_level:
                    managed_position.iloc[i] = 0
                    trade_stats['trailing_stop_exits'] += 1
                    in_position = False
            else:  # Short position
                stop_level = lowest_price * (1 + dynamic_distance)
                if current_price > stop_level:
                    managed_position.iloc[i] = 0
                    trade_stats['trailing_stop_exits'] += 1
                    in_position = False
    
    return managed_position, trade_stats

@njit(fastmath=True)
def _calculate_trade_returns(positions, returns, n):
    """
    Numba-optimized function to calculate trade returns.
    
    Parameters:
        positions: NumPy array of position sizes
        returns: NumPy array of price returns
        n: Length of arrays
        
    Returns:
        NumPy array of trade returns
    """
    trade_returns = np.zeros(n)
    
    for i in range(1, n):
        if positions[i] != 0:
            if positions[i] == positions[i-1]:
                # Continuing the same position
                trade_returns[i] = (1 + trade_returns[i-1]) * (1 + returns[i] * positions[i-1]) - 1
            else:
                # New position
                trade_returns[i] = returns[i] * positions[i]
    
    return trade_returns

def calculate_trade_returns(sized_position, returns):
    """
    Calculate the cumulative returns for each trade using Numba optimization.
    
    Parameters:
        sized_position (Series): Position series with sizing applied
        returns (Series): Asset returns series
        
    Returns:
        Series: Trade returns series
    """
    # Extract arrays for faster processing
    positions_array = sized_position.values
    returns_array = returns.values
    n = len(positions_array)
    
    # Use Numba-optimized function
    trade_returns_array = _calculate_trade_returns(positions_array, returns_array, n)
    
    # Convert back to pandas Series
    trade_returns = pd.Series(trade_returns_array, index=sized_position.index)
    
    return trade_returns