# signals.py
# Functions for signal generation, trend calculation, and filtering

import numpy as np
import pandas as pd

def calculate_trend_strength(df, window=200):
    """
    Calculate trend strength indicator.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        window (int): Window for trend calculation
        
    Returns:
        Series: Trend strength indicator
    """
    # Calculate moving average
    ma = df['close_price'].rolling(window=window).mean()
    
    # Calculate distance from MA normalized by volatility
    price_distance = (df['close_price'] - ma).abs()
    price_volatility = df['close_price'].rolling(window=window).std()
    
    # Calculate trend strength as normalized distance from MA
    trend_strength = price_distance / price_volatility
    
    # Determine trend direction (positive for uptrend, negative for downtrend)
    trend_direction = np.sign(df['close_price'] - ma)
    
    # Combine strength and direction
    directional_strength = trend_strength * trend_direction
    
    return directional_strength.fillna(0)

def calculate_momentum(df, window=20):
    """
    Calculate price momentum.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        window (int): Window for momentum calculation
        
    Returns:
        Series: Momentum indicator
    """
    # Calculate momentum as percentage change over window
    momentum = df['close_price'].pct_change(periods=window).fillna(0)
    
    # Normalize momentum by dividing by volatility over same period
    volatility = df['close_price'].pct_change().rolling(window=window).std().fillna(0)
    
    # Avoid division by zero
    volatility = volatility.replace(0, volatility.median())
    
    # Calculate normalized momentum
    normalized_momentum = momentum / volatility
    
    return normalized_momentum

def filter_signals(signal, trend_strength, momentum, min_trend_strength=0.3):
    """
    Filter trading signals based on trend strength and momentum.
    Vectorized implementation.
    
    Parameters:
        signal (Series): Raw signal series (-1, 0, 1)
        trend_strength (Series): Trend strength indicator
        momentum (Series): Momentum indicator
        min_trend_strength (float): Minimum trend strength to generate a signal
        
    Returns:
        Series: Filtered signal (-1, 0, 1)
    """
    # Initialize filtered signal
    filtered_signal = pd.Series(0, index=signal.index)
    
    # Take long positions only in uptrend with sufficient strength (vectorized)
    long_condition = (signal > 0) & (trend_strength > min_trend_strength) & (momentum > 0)
    filtered_signal[long_condition] = 1
    
    # Take short positions only in downtrend with sufficient strength (vectorized)
    short_condition = (signal < 0) & (trend_strength < -min_trend_strength) & (momentum < 0)
    filtered_signal[short_condition] = -1
    
    return filtered_signal

def apply_min_holding_period(position, min_holding_hours=24):
    """
    Apply minimum holding period to reduce overtrading.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        min_holding_hours (int): Minimum holding period in hours
        
    Returns:
        Series: Position with minimum holding period applied
    """
    if min_holding_hours <= 1:
        return position
    
    modified_position = position.copy()
    
    # Track last trade and holding period
    last_trade_time = None
    last_position = 0
    
    for i, (timestamp, current_position) in enumerate(position.items()):
        # Position change detected
        if current_position != last_position and current_position != 0:
            # Check if minimum holding period has passed
            if last_trade_time is not None:
                hours_since_last_trade = (timestamp - last_trade_time).total_seconds() / 3600
                
                if hours_since_last_trade < min_holding_hours:
                    # Reject this trade, keep previous position
                    modified_position.iloc[i] = last_position
                    continue
            
            # Update last trade time and position
            last_trade_time = timestamp
            last_position = current_position
        
        # Position closed (moved to neutral)
        elif current_position == 0 and last_position != 0:
            # Update position
            last_position = 0
    
    return modified_position