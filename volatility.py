# volatility.py
# Functions for calculating various volatility metrics

import numpy as np
import pandas as pd
from numba import njit
# Handle different versions of numba - fastmath might be a decorator not an import
try:
    from numba import fastmath
    has_fastmath_import = True
except ImportError:
    has_fastmath_import = False

from arch import arch_model

# Global cache for volatility calculations to avoid redundant computations
volatility_cache = {}

@njit(fastmath=True)
def _calculate_parkinson_var_safe(high_vals, low_vals, window):
    """Numba-optimized helper for Parkinson volatility calculation with safety checks"""
    n = len(high_vals)
    factor = 1.0 / (4.0 * np.log(2.0))
    result = np.zeros(n)
    
    # Calculate log(high/low)^2 with safety checks
    log_hl_squared = np.zeros(n)
    for i in range(n):
        # Ensure high is > low and both are positive
        if high_vals[i] > low_vals[i] and high_vals[i] > 0 and low_vals[i] > 0:
            log_hl_squared[i] = np.log(high_vals[i] / low_vals[i])**2
        elif i > 0:  # Use previous value if current is invalid
            log_hl_squared[i] = log_hl_squared[i-1]
    
    # Calculate rolling mean with a safety check for zero divide
    for i in range(n):
        if i >= window:
            total = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if j >= 0 and j < n and not np.isnan(log_hl_squared[j]) and not np.isinf(log_hl_squared[j]):
                    total += log_hl_squared[j]
                    count += 1
            if count > 0:
                result[i] = factor * (total / count)
    
    return result

def calculate_parkinson_volatility(df, window=20):
    """
    Calculate Parkinson volatility using high-low range.
    Optimized with Numba and improved error handling.
    """
    # Check if high/low data is available
    if 'high_price' not in df.columns or 'low_price' not in df.columns:
        print("Warning: high_price or low_price not found. Falling back to standard deviation.")
        return calculate_standard_volatility(df, window)
    
    try:
        # Extract numpy arrays for faster processing
        high_vals = df['high_price'].values
        low_vals = df['low_price'].values
        
        # Check for zero or negative values that will cause problems
        valid_mask = (high_vals > 0) & (low_vals > 0) & (high_vals >= low_vals)
        if not valid_mask.all():
            print(f"Warning: Found {(~valid_mask).sum()} invalid high/low values. Correcting them.")
            
            # Fix invalid values
            for i in range(len(high_vals)):
                if not valid_mask[i]:
                    # Set to previous valid values or reasonable defaults
                    if i > 0:
                        high_vals[i] = high_vals[i-1]
                        low_vals[i] = low_vals[i-1]
                    else:
                        # Use close price with small spread for first element if needed
                        close_val = df['close_price'].iloc[0]
                        high_vals[i] = close_val * 1.001
                        low_vals[i] = close_val * 0.999
        
        # Use numba-optimized function with error handling
        parkinsons_var = _calculate_parkinson_var_safe(high_vals, low_vals, window)
        
        # Convert variance to standard deviation (volatility)
        parkinsons_vol = np.sqrt(np.maximum(parkinsons_var, 0))  # Ensure non-negative
        
        # Create series with same index
        result = pd.Series(parkinsons_vol, index=df.index)
        
        # Get trading frequency from global configuration if available
        try:
            from enhanced_config import TRADING_FREQUENCY
        except ImportError:
            try:
                from config import TRADING_FREQUENCY
            except ImportError:
                TRADING_FREQUENCY = "1H"  # Default fallback
        
        # Annualize the volatility based on trading frequency
        if TRADING_FREQUENCY == "1H":
            result = result * np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            result = result * np.sqrt(365)
        
        # Fill NaN values
        result = result.bfill().ffill()
        
        return result
        
    except Exception as e:
        print(f"Error in Parkinson volatility calculation: {e}")
        print("Falling back to standard volatility")
        return calculate_standard_volatility(df, window)

@njit(fastmath=True)
def _calculate_rolling_std(values, window):
    """Numba-optimized rolling standard deviation"""
    n = len(values)
    result = np.zeros(n)
    
    for i in range(n):
        if i >= window:
            # Calculate mean
            mean = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if j >= 0:
                    mean += values[j]
                    count += 1
            if count > 0:
                mean /= count
                
            # Calculate variance
            var = 0.0
            count = 0
            for j in range(i - window + 1, i + 1):
                if j >= 0:
                    var += (values[j] - mean)**2
                    count += 1
            if count > 1:
                var /= (count - 1)
                result[i] = np.sqrt(var)
    
    return result

def calculate_standard_volatility(df, window=20):
    """
    Calculate standard volatility using close price returns.
    Optimized with Numba.
    """
    # Calculate returns
    returns = df['close_price'].pct_change().fillna(0).values
    
    # Use numba-optimized function
    volatility = _calculate_rolling_std(returns, window)
    
    # Create series with same index
    result = pd.Series(volatility, index=df.index)
    
    # Get trading frequency from global configuration if available
    try:
        from enhanced_config import TRADING_FREQUENCY
    except ImportError:
        try:
            from config import TRADING_FREQUENCY
        except ImportError:
            TRADING_FREQUENCY = "1H"  # Default fallback
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        result = result * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        result = result * np.sqrt(365)
    
    return result.fillna(method='bfill').fillna(method='ffill')

def calculate_yang_zhang_volatility(df, window=20):
    """
    Calculate Yang-Zhang volatility which combines overnight and intraday volatility.
    
    Parameters:
        df (DataFrame): DataFrame with open, high, low, close columns
        window (int): Rolling window for volatility calculation
        
    Returns:
        Series: Yang-Zhang volatility (annualized)
    """
    # Check if necessary data is available
    required_cols = ['open_price', 'high_price', 'low_price', 'close_price']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Required columns for Yang-Zhang volatility not found. Using Parkinson.")
        return calculate_parkinson_volatility(df, window)
    
    # Calculate overnight returns (close to open)
    close_to_open = np.log(df['open_price'] / df['close_price'].shift(1))
    
    # Calculate open to close returns
    open_to_close = np.log(df['close_price'] / df['open_price'])
    
    # Calculate Rogers-Satchell volatility components
    rs_vol = (np.log(df['high_price'] / df['close_price']) * 
              np.log(df['high_price'] / df['open_price']) + 
              np.log(df['low_price'] / df['close_price']) * 
              np.log(df['low_price'] / df['open_price']))
    
    # Calculate the different variance components
    close_to_open_var = close_to_open.rolling(window=window).var()
    open_to_close_var = open_to_close.rolling(window=window).var()
    rs_var = rs_vol.rolling(window=window).mean()
    
    # Combine components with weights
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yang_zhang_var = close_to_open_var + k * open_to_close_var + (1 - k) * rs_var
    
    # Convert to volatility (standard deviation)
    yang_zhang_vol = np.sqrt(yang_zhang_var)
    
    # Get trading frequency from global configuration if available
    try:
        from enhanced_config import TRADING_FREQUENCY
    except ImportError:
        try:
            from config import TRADING_FREQUENCY
        except ImportError:
            TRADING_FREQUENCY = "1H"  # Default fallback
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        yang_zhang_vol = yang_zhang_vol * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        yang_zhang_vol = yang_zhang_vol * np.sqrt(365)
    
    return yang_zhang_vol.fillna(method='bfill').fillna(method='ffill')

def calculate_garch_volatility(df, forecast_horizon=1):
    """
    Calculate volatility using a GARCH(1,1) model.
    
    Parameters:
        df (DataFrame): DataFrame with close_price column
        forecast_horizon (int): Forecast horizon for volatility prediction
        
    Returns:
        Series: GARCH volatility forecast (annualized)
    """
    # Calculate returns
    returns = 100 * df['close_price'].pct_change().fillna(0)
    
    try:
        # Initialize GARCH model (reduced complexity for speed)
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        
        # Fit the model with a fixed window to avoid recalculation for entire series
        # This uses a rolling estimation instead of expanding to improve performance
        window_size = min(1000, len(returns))
        
        # Create Series to store forecasted volatility
        forecasted_vol = pd.Series(index=returns.index, dtype=float)
        
        # For periods with enough data, estimate GARCH and forecast
        for i in range(window_size, len(returns), 100):  # Update every 100 steps for efficiency
            end_loc = min(i + 100, len(returns))
            try:
                # Get subseries for estimation
                subseries = returns.iloc[max(0, i-window_size):i]
                
                # Fit model
                res = model.fit(disp='off', show_warning=False, update_freq=0)
                
                # Forecast volatility
                forecast = res.forecast(horizon=forecast_horizon)
                conditional_vol = np.sqrt(forecast.variance.iloc[-1].values[0])
                
                # Assign forecasted volatility to next periods
                for j in range(i, min(end_loc, len(returns))):
                    if j < len(forecasted_vol):
                        forecasted_vol.iloc[j] = conditional_vol
            except:
                # Fall back to standard deviation on failure
                forecasted_vol.iloc[i:end_loc] = returns.iloc[max(0, i-window_size):i].std()
        
        # Fill any missing values with standard deviation
        forecasted_vol = forecasted_vol.fillna(returns.rolling(window=20).std())
        
        # Get trading frequency from global configuration if available
        try:
            from enhanced_config import TRADING_FREQUENCY
        except ImportError:
            try:
                from config import TRADING_FREQUENCY
            except ImportError:
                TRADING_FREQUENCY = "1H"  # Default fallback
        
        # Annualize volatility based on trading frequency
        if TRADING_FREQUENCY == "1H":
            annualized_vol = forecasted_vol * np.sqrt(24 * 365) / 100
        elif TRADING_FREQUENCY == "1D":
            annualized_vol = forecasted_vol * np.sqrt(365) / 100
        else:
            annualized_vol = forecasted_vol / 100
        
        return annualized_vol
    except Exception as e:
        print(f"GARCH estimation failed: {e}")
        print("Falling back to standard volatility")
        return calculate_standard_volatility(df)

def calculate_volatility(df, method='parkinson', window=20):
    """
    Calculate volatility using the specified method with robust error handling.
    Uses caching to avoid recalculating for the same parameters.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        method (str): Method to use ('parkinson', 'garch', 'yang_zhang', or 'standard')
        window (int): Rolling window for volatility calculation
        
    Returns:
        Series: Volatility series
    """
    # Create a cache key
    cache_key = f"{method}_{window}_{hash(tuple(df['close_price']))}"
    
    # Check if result is in cache
    if cache_key in volatility_cache:
        return volatility_cache[cache_key]
    
    try:
        # Calculate volatility based on method
        if method == 'parkinson':
            result = calculate_parkinson_volatility(df, window)
        elif method == 'garch':
            result = calculate_garch_volatility(df)
        elif method == 'yang_zhang':
            result = calculate_yang_zhang_volatility(df, window)
        else:  # Use standard volatility as default
            result = calculate_standard_volatility(df, window)
        
        # Validate result - replace infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with forward fill then backward fill
        result = result.ffill().bfill()
        
        # Check for remaining NaNs and replace with median if any exist
        if result.isna().any():
            median_val = result.median()
            # If median is NaN, use a small default value
            if pd.isna(median_val):
                median_val = 0.01  # 1% annualized volatility as a fallback
            result = result.fillna(median_val)
        
        # Apply reasonable bounds to volatility
        min_vol = 0.001  # 0.1% annualized
        max_vol = 5.0    # 500% annualized
        result = result.clip(lower=min_vol, upper=max_vol)
        
        # Store in cache
        volatility_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"Error calculating {method} volatility: {e}")
        print("Falling back to standard volatility")
        try:
            # Use standard volatility as fallback
            result = calculate_standard_volatility(df, window)
            
            # Apply safety checks as above
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.ffill().bfill()
                        
            if result.isna().any():
                median_val = result.median()
                if pd.isna(median_val):
                    median_val = 0.01
                result = result.fillna(median_val)
            
            result = result.clip(lower=0.001, upper=5.0)
            return result
        except Exception as ex:
            print(f"Error in fallback volatility calculation: {ex}")
            # Return a constant volatility as last resort
            return pd.Series(0.2, index=df.index)  # 20% constant volatility

def clear_volatility_cache():
    """Clear the volatility cache to free memory"""
    global volatility_cache
    volatility_cache = {}
    print("Volatility cache cleared")