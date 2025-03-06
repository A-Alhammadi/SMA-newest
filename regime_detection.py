# regime_detection.py
# Functions for detecting market regimes based on volatility

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from hmmlearn import hmm
from statsmodels.nonparametric.kde import KDEUnivariate

def detect_regimes_hmm(volatility_series, n_regimes=3, smoothing_period=5, stability_period=48, verbose=False, max_attempts=5):
    """
    Detect volatility regimes using Hidden Markov Models with improved stability and robustness.
    """
    # Prepare data for HMM - robust handling of edge cases
    vol_values = volatility_series.replace([np.inf, -np.inf], np.nan)
    
    # Ensure we have positive values for log transform
    min_positive = vol_values[vol_values > 0].min() if (vol_values > 0).any() else 1e-6
    vol_values = vol_values.fillna(min_positive).clip(lower=min_positive)
    
    # Use log volatility to better capture distribution characteristics
    log_vol = np.log(vol_values)
    
    # Standardize the log volatility for better HMM performance
    log_vol_std = (log_vol - log_vol.mean()) / log_vol.std()
    X = log_vol_std.values.reshape(-1, 1)
    
    # List of configurations to try in order of preference
    hmm_configs = [
        {"covariance_type": "full", "n_iter": 1000, "tol": 0.01, "random_state": 42},
        {"covariance_type": "diag", "n_iter": 1000, "tol": 0.01, "random_state": 0},
        {"covariance_type": "spherical", "n_iter": 1000, "tol": 0.05, "random_state": 1},
        {"covariance_type": "tied", "n_iter": 1500, "tol": 0.05, "random_state": 2},
        {"covariance_type": "full", "n_iter": 2000, "tol": 0.1, "random_state": 3}
    ]
    
    best_model = None
    best_score = float('-inf')
    best_hidden_states = None
    
    # Try different HMM configurations
    for config_idx, config in enumerate(hmm_configs):
        if verbose:
            print(f"Trying HMM configuration {config_idx+1}/{len(hmm_configs)}: {config['covariance_type']} covariance")
        
        # Try multiple random initializations for this configuration
        for attempt in range(max_attempts):
            try:
                # Set different random seed for each attempt
                random_seed = config["random_state"] + attempt if "random_state" in config else None
                
                # Initialize HMM model with current configuration
                model = hmm.GaussianHMM(
                    n_components=n_regimes, 
                    covariance_type=config["covariance_type"], 
                    n_iter=config["n_iter"],
                    tol=config["tol"],
                    random_state=random_seed,
                    verbose=False
                )
                
                # Fit model
                model.fit(X)
                
                # Check if model converged
                if not model.monitor_.converged:
                    if verbose:
                        print(f"  Attempt {attempt+1}: HMM did not converge")
                    continue
                
                # Calculate score
                score = model.score(X)
                
                # If this is the best model so far, store it
                if score > best_score:
                    best_score = score
                    best_model = model
                    # Predict hidden states
                    best_hidden_states = model.predict(X)
                    
                    if verbose:
                        print(f"  Attempt {attempt+1}: New best model with score {score:.2f}")
                
                # Early stopping if we found a good model
                if best_score > -0.5 and attempt > 2:
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"  Attempt {attempt+1}: HMM estimation failed: {str(e)}")
        
        # If we found a good model with this configuration, we can stop
        if best_model is not None and best_score > -0.5:
            if verbose:
                print(f"Found satisfactory model with {config['covariance_type']} covariance, score: {best_score:.2f}")
            break
    
    # If we found a model, use it
    if best_model is not None and best_hidden_states is not None:
        # Map states by volatility level (0=low, n_regimes-1=high)
        state_vol_means = {}
        for state in range(n_regimes):
            if np.any(best_hidden_states == state):
                state_vol_means[state] = np.mean(volatility_series[best_hidden_states == state])
        
        # Sort states by mean volatility
        sorted_states = sorted(state_vol_means.items(), key=lambda x: x[1])
        state_mapping = {old_state: new_state for new_state, (old_state, _) in enumerate(sorted_states)}
        
        # Apply mapping to reorder states from low to high volatility
        mapped_states = np.array([state_mapping.get(state, state) for state in best_hidden_states])
        
        # Create Series with regime labels
        regimes = pd.Series(mapped_states, index=volatility_series.index)
        
        # Apply smoothing to prevent frequent regime transitions
        if smoothing_period > 1:
            regimes = regimes.rolling(window=smoothing_period, center=True).median().ffill().bfill()
            regimes = regimes.round().astype(int)
        
        # Apply stability constraint if specified
        if stability_period > 1:
            regimes = apply_regime_stability_constraint(regimes, stability_period)
        
        return regimes
    
    # If all HMM attempts failed, fall back to K-means
    print("All HMM attempts failed. Falling back to K-means for regime detection")
    return detect_regimes_kmeans(volatility_series, n_regimes, smoothing_period)

def apply_regime_stability_constraint(regimes, stability_period):
    """
    Apply stability constraint to prevent rapid regime transitions.
    """
    stable_regimes = regimes.copy()
    
    # Get unique regimes
    unique_regimes = sorted(regimes.unique())
    current_regime = regimes.iloc[0]
    regime_counters = {r: 0 for r in unique_regimes}
    regime_counters[current_regime] = 1
    
    # Process each time step
    for i in range(1, len(regimes)):
        new_regime = regimes.iloc[i]
        
        if new_regime == current_regime:
            # Still in the same regime
            regime_counters[current_regime] += 1
        else:
            # Potential regime change
            regime_counters[new_regime] = 1
            
            # Only change regime if it has been stable for long enough
            if regime_counters[current_regime] >= stability_period:
                # Reset counter for previous regime
                regime_counters[current_regime] = 0
                current_regime = new_regime
            else:
                # Not enough consecutive periods, maintain previous regime
                stable_regimes.iloc[i] = current_regime
                regime_counters[current_regime] += 1
    
    return stable_regimes

def detect_regimes_kmeans(volatility_series, n_regimes=3, smoothing_period=5):
    """
    Detect volatility regimes using K-means clustering with robust error handling.
    
    Parameters:
        volatility_series (Series): Volatility time series
        n_regimes (int): Number of regimes to identify
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0 to n_regimes-1)
    """
    try:
        # Prepare data for clustering - safely handle extremes and infinities
        vol_values = volatility_series.replace([np.inf, -np.inf], np.nan)
        vol_values = vol_values.fillna(vol_values.median())
        
        # Make sure we have valid values throughout
        if vol_values.isna().any() or (vol_values <= 0).any():
            # Handle problematic values
            min_vol = vol_values[vol_values > 0].min() if (vol_values > 0).any() else 0.001
            vol_values = vol_values.clip(lower=min_vol)
            vol_values = vol_values.fillna(vol_values.median())
        
        # Use log volatility to better capture distribution characteristics
        X = np.log(vol_values).values.reshape(-1, 1)
        
        # Handle any remaining infinities or NaNs
        X = np.nan_to_num(X, nan=np.log(vol_values.median()), 
                          posinf=np.log(vol_values.quantile(0.99)), 
                          neginf=np.log(vol_values.quantile(0.01)))
        
        # Train KMeans model
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Sort regimes by volatility level (0=low, n_regimes-1=high)
        cluster_centers = kmeans.cluster_centers_.flatten()
        sorting_indices = np.argsort(cluster_centers)
        
        # Map original labels to sorted labels
        regime_map = {sorting_indices[i]: i for i in range(n_regimes)}
        sorted_labels = np.array([regime_map[label] for label in labels])
        
        # Create Series with regime labels
        regimes = pd.Series(sorted_labels, index=volatility_series.index)
        
        # Apply smoothing to prevent frequent regime transitions
        if smoothing_period > 1:
            regimes = regimes.rolling(window=smoothing_period, center=True).median().ffill().bfill()
            regimes = regimes.round().astype(int)
        
        return regimes
        
    except Exception as e:
        print(f"Error in regime detection: {e}")
        print("Falling back to simple quantile-based regime detection")
        
        # Simple fallback using quantiles
        try:
            # Create a simple 3-regime classification using quantiles
            vol_values = volatility_series.replace([np.inf, -np.inf], np.nan)
            vol_values = vol_values.fillna(vol_values.median())
            
            # Create regimes based on quantiles
            low_threshold = vol_values.quantile(0.33)
            high_threshold = vol_values.quantile(0.67)
            
            regimes = pd.Series(1, index=volatility_series.index)  # Default to middle regime
            regimes[vol_values <= low_threshold] = 0
            regimes[vol_values >= high_threshold] = 2
            
            # Apply smoothing if needed
            if smoothing_period > 1:
                regimes = regimes.rolling(window=smoothing_period, center=True).median().ffill().bfill()
                regimes = regimes.round().astype(int)
            
            return regimes
            
        except Exception as ex:
            print(f"Error in fallback regime detection: {ex}")
            # Last resort: return all zeros (single regime)
            return pd.Series(0, index=volatility_series.index)

def detect_regimes_kde(volatility_series, n_regimes=3, smoothing_period=5):
    """
    Detect volatility regimes using Kernel Density Estimation.
    
    Parameters:
        volatility_series (Series): Volatility time series
        n_regimes (int): Number of regimes to identify
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0 to n_regimes-1)
    """
    # Use log volatility to better capture distribution characteristics
    log_vol = np.log(volatility_series.replace(0, np.nan).fillna(volatility_series.min()))
    
    # Fit KDE
    kde = KDEUnivariate(log_vol.values)
    kde.fit()
    
    # Find local minima in the density to identify regime boundaries
    density = kde.density
    x_grid = kde.support
    
    # Use first and second derivatives to find local minima
    d_density = np.gradient(density)
    dd_density = np.gradient(d_density)
    
    # Find points where first derivative is close to zero and second derivative is positive
    # These are local minima which represent boundaries between regimes
    local_min_indices = []
    for i in range(1, len(d_density) - 1):
        if abs(d_density[i]) < 0.01 and dd_density[i] > 0:
            local_min_indices.append(i)
    
    # If we cannot find enough local minima, use quantiles instead
    if len(local_min_indices) < n_regimes - 1:
        print(f"Not enough local minima found. Using quantiles instead.")
        thresholds = [np.quantile(log_vol, q) for q in np.linspace(0, 1, n_regimes+1)[1:-1]]
    else:
        # Sort and select top n_regimes-1 boundaries
        local_min_indices.sort(key=lambda i: density[i])
        boundary_indices = local_min_indices[:n_regimes-1]
        boundary_indices.sort()  # Sort by x value for consistent thresholds
        thresholds = [x_grid[i] for i in boundary_indices]
    
    # Apply thresholds to classify regimes
    regimes = pd.Series(0, index=volatility_series.index)
    for i, threshold in enumerate(sorted(thresholds), 1):
        regimes[log_vol > threshold] = i
    
    # Apply smoothing to prevent frequent regime transitions
    if smoothing_period > 1:
        regimes = regimes.rolling(window=smoothing_period, center=True).median().ffill().bfill()
        regimes = regimes.round().astype(int)
    
    return regimes

def detect_regimes_quantile(volatility_series, quantile_thresholds=[0.33, 0.67], smoothing_period=5):
    """
    Detect volatility regimes using empirical quantiles.
    
    Parameters:
        volatility_series (Series): Volatility time series
        quantile_thresholds (list): List of quantile thresholds
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0, 1, 2, etc.)
    """
    n_regimes = len(quantile_thresholds) + 1
    
    # Calculate expanding quantiles to avoid lookahead bias
    expanding_thresholds = []
    min_periods = 500  # Minimum data points for reliable quantile estimation
    
    for q in quantile_thresholds:
        threshold_series = volatility_series.expanding(min_periods=min_periods).quantile(q)
        threshold_series = threshold_series.fillna(volatility_series.median())  # Fill early periods
        expanding_thresholds.append(threshold_series)
    
    # Initialize regime series
    regimes = pd.Series(0, index=volatility_series.index)
    
    # Classify regimes based on thresholds
    for i, threshold_series in enumerate(expanding_thresholds, 1):
        mask = volatility_series > threshold_series
        regimes[mask] = i
    
    # Apply smoothing to prevent frequent regime transitions
    if smoothing_period > 1:
        regimes = regimes.rolling(window=smoothing_period, center=True).median().ffill().bfill()
        regimes = regimes.round().astype(int)
    
    return regimes

def detect_volatility_regimes(df, volatility, method='kmeans', n_regimes=3, 
                             quantile_thresholds=[0.33, 0.67], smoothing_period=5,
                             stability_period=48, verbose=False):
    """
    Detect volatility regimes using the specified method and apply stability constraints.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        volatility (Series): Volatility series
        method (str): Method for regime detection ('kmeans', 'kde', 'quantile', or 'hmm')
        n_regimes (int): Number of regimes to identify (for kmeans/kde/hmm)
        quantile_thresholds (list): Quantile thresholds (for quantile method)
        smoothing_period (int): Period for smoothing regime transitions
        stability_period (int): Hours required before confirming regime change
        verbose (bool): Whether to print detailed progress messages
        
    Returns:
        Series: Regime classifications
    """
    # Detect regimes using selected method
    if method == 'kmeans':
        regimes = detect_regimes_kmeans(volatility, n_regimes, smoothing_period)
    elif method == 'kde':
        regimes = detect_regimes_kde(volatility, n_regimes, smoothing_period)
    elif method == 'hmm':
        regimes = detect_regimes_hmm(volatility, n_regimes, smoothing_period, verbose)
    else:  # Default to quantile method
        regimes = detect_regimes_quantile(volatility, quantile_thresholds, smoothing_period)
    
    # Apply stability constraint to prevent rapid regime transitions
    if stability_period > 1:
        stable_regimes = regimes.copy()
        
        # Track consecutive periods in the same regime
        last_regime = regimes.iloc[0]
        consecutive_periods = 0
        
        for i in range(len(regimes)):
            current_regime = regimes.iloc[i]
            
            if current_regime == last_regime:
                consecutive_periods += 1
            else:
                # Only change regime if it has persisted for stability_period
                if consecutive_periods >= stability_period:
                    # Confirmed regime change
                    last_regime = current_regime
                    consecutive_periods = 0
                else:
                    # Reject regime change, maintain previous regime
                    stable_regimes.iloc[i] = last_regime
                    consecutive_periods += 1
        
        return stable_regimes
    
    return regimes