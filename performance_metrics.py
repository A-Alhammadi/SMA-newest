# performance_metrics.py
# Functions for performance and risk metrics calculation

import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=None):
    """
    Calculate Sharpe ratio.
    
    Parameters:
        returns (Series): Returns series
        risk_free_rate (float): Risk-free rate
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Sharpe ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        try:
            from enhanced_config import TRADING_FREQUENCY
        except ImportError:
            try:
                from config import TRADING_FREQUENCY
            except ImportError:
                TRADING_FREQUENCY = "1H"  # Default
        
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    # Remove outliers
    clean_returns = returns.clip(returns.quantile(0.01), returns.quantile(0.99))
    
    mean_return = clean_returns.mean()
    std_return = clean_returns.std()
    
    if std_return == 0:
        return 0
    
    sharpe = ((mean_return - risk_free_rate) / std_return) * annualization_factor
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=None):
    """
    Calculate Sortino ratio (using only downside deviation).
    
    Parameters:
        returns (Series): Returns series
        risk_free_rate (float): Risk-free rate
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Sortino ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        try:
            from enhanced_config import TRADING_FREQUENCY
        except ImportError:
            try:
                from config import TRADING_FREQUENCY
            except ImportError:
                TRADING_FREQUENCY = "1H"  # Default
        
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    # Calculate downside returns (returns below target, typically 0)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return 0
    
    # Calculate Sortino ratio
    sortino = ((returns.mean() - risk_free_rate) / downside_deviation) * annualization_factor
    
    return sortino

def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve.
    
    Parameters:
        equity_curve (Series): Equity curve
        
    Returns:
        float: Maximum drawdown
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve / running_max) - 1
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def calculate_calmar_ratio(returns, max_drawdown, annualization_factor=None):
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Parameters:
        returns (Series): Returns series
        max_drawdown (float): Maximum drawdown (positive number)
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Calmar ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        try:
            from enhanced_config import TRADING_FREQUENCY
        except ImportError:
            try:
                from config import TRADING_FREQUENCY
            except ImportError:
                TRADING_FREQUENCY = "1H"  # Default
        
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    if max_drawdown == 0:
        return np.inf  # No drawdown
    
    # Convert max_drawdown to positive value if needed
    abs_drawdown = abs(max_drawdown)
    
    # Calculate annualized return
    annual_return = returns.mean() * annualization_factor
    
    calmar = annual_return / abs_drawdown
    
    return calmar

def calculate_advanced_metrics(strategy_returns, equity_curve):
    """
    Calculate advanced performance metrics for the strategy.
    
    Parameters:
        strategy_returns (Series): Strategy returns series
        equity_curve (Series): Strategy equity curve
        
    Returns:
        dict: Performance metrics
    """
    # Basic metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    annualized_return = ((1 + total_return) ** (365 / (equity_curve.index[-1] - equity_curve.index[0]).days)) - 1
    
    # Risk metrics
    volatility = strategy_returns.std() * np.sqrt(252)  # Annualized
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(strategy_returns)
    sortino = calculate_sortino_ratio(strategy_returns)
    calmar = calculate_calmar_ratio(strategy_returns, max_dd)
    
    # Efficiency metrics - handle case with no trades
    non_zero_returns = strategy_returns[strategy_returns != 0]
    if len(non_zero_returns) > 0:
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(non_zero_returns)
        gain_sum = strategy_returns[strategy_returns > 0].sum()
        loss_sum = abs(strategy_returns[strategy_returns < 0].sum())
        gain_to_pain = gain_sum / loss_sum if loss_sum > 0 else np.inf
    else:
        # No trades case
        win_rate = 0.0
        gain_to_pain = 0.0
    
    # Return dictionary of metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'gain_to_pain': gain_to_pain
    }
    
    return metrics