#!/usr/bin/env python
# analyze_parameters.py - Script to analyze parameter sensitivity from existing models

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze parameter sensitivity from existing models')
    parser.add_argument('--results_dir', type=str, default='enhanced_sma_results',
                        help='Directory containing saved model results')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Specific model file to analyze (if None, analyzes all models in results_dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (defaults to results_dir)')
    parser.add_argument('--n_top_params', type=int, default=10,
                        help='Number of top parameters to include in detailed analysis')
    
    return parser.parse_args()

def load_model_files(results_dir, model_file=None):
    """Load model files for analysis."""
    models = []
    
    # If specific model file is provided
    if model_file is not None:
        model_path = os.path.join(results_dir, model_file) if not os.path.isabs(model_file) else model_file
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                models.append((model_path, model_data))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
        else:
            print(f"Model file not found: {model_path}")
    # Otherwise, load all model files from the directory
    else:
        model_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
        for model_file in model_files:
            model_path = os.path.join(results_dir, model_file)
            try:
                model_data = joblib.load(model_path)
                models.append((model_path, model_data))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
    
    return models

def extract_parameters_and_performance(models):
    """
    Extract parameters and performance metrics from model files.
    Updated to handle different model file formats.
    
    Parameters:
        models (list): List of tuples (model_path, model_data)
        
    Returns:
        pd.DataFrame: DataFrame with parameters and performance metrics
    """
    all_data = []
    
    for model_path, model_data in models:
        try:
            model_name = os.path.basename(model_path)
            print(f"\nExtracting data from {model_name}")
            
            # Debug - print top-level keys in model data
            print(f"Top-level keys: {list(model_data.keys())}")
            
            # Handle different possible model formats
            
            # Format 1: Section result file
            if 'regime_params' in model_data:
                # This is a section result file that contains regime parameters directly
                regime_params = model_data['regime_params']
                
                print(f"Found {len(regime_params)} regimes in section file")
                
                # Get metrics if available
                metrics = model_data.get('metrics', {})
                
                # For each regime in this section
                for regime_id, params in regime_params.items():
                    # Create entry with metadata and performance
                    entry = {
                        'model_name': model_name,
                        'section': model_data.get('section_index', 0),
                        'regime': regime_id,
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'sortino_ratio': metrics.get('sortino_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'calmar_ratio': metrics.get('calmar_ratio', 0),
                        'outperformance': model_data.get('outperformance', 0)
                    }
                    
                    # Add regime-specific metrics if available
                    if 'regime_metrics' in model_data and regime_id in model_data['regime_metrics']:
                        regime_metrics = model_data['regime_metrics'][regime_id]
                        entry.update({
                            f'regime_{k}': v for k, v in regime_metrics.items()
                        })
                    
                    # Add all parameters
                    for param_name, param_value in params.items():
                        # Skip non-parameter items
                        if param_name in ['validation_score', 'validation_regime_return', 
                                          'validation_regime_sharpe']:
                            entry[param_name] = param_value
                            continue
                        
                        # Handle different parameter types
                        if isinstance(param_value, (int, float, str, bool)):
                            entry[param_name] = param_value
                        else:
                            try:
                                # Try to convert to string
                                entry[param_name] = str(param_value)
                            except:
                                # Skip parameters that can't be converted
                                pass
                    
                    all_data.append(entry)
            
            # Format 2: Combined results file with section_results
            elif 'section_results' in model_data:
                print(f"Found combined results file with {len(model_data['section_results'])} sections")
                
                for section_idx, section in enumerate(model_data['section_results']):
                    if 'regime_params' in section:
                        regime_params = section['regime_params']
                        print(f"  Section {section_idx}: Found {len(regime_params)} regimes")
                        
                        # Performance metrics for this section
                        metrics = section.get('metrics', {})
                        
                        for regime_id, params in regime_params.items():
                            # Create entry with metadata and performance
                            entry = {
                                'model_name': model_name,
                                'section': section_idx,
                                'regime': regime_id,
                                'total_return': metrics.get('total_return', 0),
                                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                                'sortino_ratio': metrics.get('sortino_ratio', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'calmar_ratio': metrics.get('calmar_ratio', 0),
                                'outperformance': section.get('outperformance', 0)
                            }
                            
                            # Add regime-specific metrics if available
                            if 'regime_metrics' in section and regime_id in section['regime_metrics']:
                                regime_metrics = section['regime_metrics'][regime_id]
                                entry.update({
                                    f'regime_{k}': v for k, v in regime_metrics.items()
                                })
                            
                            # Add all parameters
                            for param_name, param_value in params.items():
                                # Handle different parameter types
                                if isinstance(param_value, (int, float, str, bool)):
                                    entry[param_name] = param_value
                                else:
                                    try:
                                        # Try to convert to string
                                        entry[param_name] = str(param_value)
                                    except:
                                        # Skip parameters that can't be converted
                                        pass
                            
                            all_data.append(entry)
            
            # Format 3: Final model with parameter_importance
            elif 'overall_results' in model_data:
                print("Found final model with overall results")
                overall_results = model_data['overall_results']
                
                if 'section_results' in overall_results:
                    print(f"  Processing {len(overall_results['section_results'])} sections from overall results")
                    for section_idx, section in enumerate(overall_results['section_results']):
                        if 'regime_params' in section:
                            regime_params = section['regime_params']
                            
                            # Performance metrics for this section
                            metrics = section.get('metrics', {})
                            
                            for regime_id, params in regime_params.items():
                                # Create entry with metadata and performance
                                entry = {
                                    'model_name': model_name,
                                    'section': section_idx,
                                    'regime': regime_id,
                                    'total_return': metrics.get('total_return', 0),
                                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                                    'sortino_ratio': metrics.get('sortino_ratio', 0),
                                    'max_drawdown': metrics.get('max_drawdown', 0),
                                    'calmar_ratio': metrics.get('calmar_ratio', 0),
                                    'outperformance': section.get('outperformance', 0)
                                }
                                
                                # Add all parameters
                                for param_name, param_value in params.items():
                                    # Handle different parameter types
                                    if isinstance(param_value, (int, float, str, bool)):
                                        entry[param_name] = param_value
                                    else:
                                        try:
                                            # Try to convert to string
                                            entry[param_name] = str(param_value)
                                        except:
                                            # Skip parameters that can't be converted
                                            pass
                                
                                all_data.append(entry)
            else:
                print(f"Unknown model format for {model_name}. Available keys: {list(model_data.keys())}")
        
        except Exception as e:
            print(f"Error extracting data from {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert to DataFrame
    if not all_data:
        print("No parameter data found in model files")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"\nSuccessfully extracted data for {len(df)} parameter sets")
    print(f"Columns: {list(df.columns)}")
    
    return df

def analyze_parameter_sensitivity(params_df, output_dir, n_top_params=10):
    """
    Analyze parameter sensitivity and create visualizations.
    
    Parameters:
        params_df (pd.DataFrame): DataFrame with parameters and performance metrics
        output_dir (str): Directory to save analysis results
        n_top_params (int): Number of top parameters to include in detailed analysis
    """
    if params_df is None or len(params_df) == 0:
        print("No parameter data to analyze")
        return
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nAnalyzing parameter sensitivity with {len(params_df)} data points...")
    
    # Identify parameter columns and metric columns
    meta_columns = ['model_name', 'section', 'regime']
    metric_columns = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                      'calmar_ratio', 'outperformance']
    param_columns = [col for col in params_df.columns 
                    if col not in meta_columns and col not in metric_columns]
    
    print(f"Found {len(param_columns)} parameter columns and {len(metric_columns)} metric columns")
    
    # Save the full parameter database
    params_csv = os.path.join(output_dir, f"parameter_database_{timestamp}.csv")
    params_df.to_csv(params_csv, index=False)
    print(f"Parameter database saved to {params_csv}")
    
    # Calculate parameter importance using various methods
    try:
        # Prepare data for analysis
        numeric_params = params_df[param_columns].select_dtypes(include=['number'])
        if len(numeric_params.columns) == 0:
            print("No numeric parameters found for analysis")
            return
        
        # Target metric for importance analysis
        target_metric = 'sharpe_ratio'
        if target_metric not in params_df.columns:
            target_metric = 'total_return'
        
        y = params_df[target_metric]
        
        # Fill NaNs with column means
        X = numeric_params.fillna(numeric_params.mean())
        
        # Correlation-based importance
        correlations = {}
        for param in X.columns:
            corr = abs(X[param].corr(y))
            if not np.isnan(corr):
                correlations[param] = corr
        
        # Normalize correlations
        total_corr = sum(correlations.values())
        if total_corr > 0:
            for param in correlations:
                correlations[param] /= total_corr
        
        # Random Forest importance
        rf_importances = {}
        try:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train RandomForest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            
            # Get feature importances
            feature_importances = rf.feature_importances_
            
            # Map back to parameter names
            for name, importance in zip(X.columns, feature_importances):
                rf_importances[name] = importance
        except Exception as e:
            print(f"RandomForest importance calculation failed: {e}")
        
        # Linear regression importance (normalized absolute coefficients)
        lr_importances = {}
        try:
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Train linear regression
            lr = LinearRegression()
            lr.fit(X_scaled, y)
            
            # Get coefficients
            for name, coef in zip(X.columns, lr.coef_):
                lr_importances[name] = abs(coef)
            
            # Normalize
            total_coef = sum(lr_importances.values())
            if total_coef > 0:
                for param in lr_importances:
                    lr_importances[param] /= total_coef
        except Exception as e:
            print(f"Linear regression importance calculation failed: {e}")
        
        # Combine importance measures
        combined_importance = {}
        for param in set(list(correlations.keys()) + list(rf_importances.keys()) + list(lr_importances.keys())):
            corr_imp = correlations.get(param, 0)
            rf_imp = rf_importances.get(param, 0)
            lr_imp = lr_importances.get(param, 0)
            
            # Weighted combination
            combined_importance[param] = 0.3 * corr_imp + 0.5 * rf_imp + 0.2 * lr_imp
        
        # Sort by importance
        sorted_importance = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create parameter importance plot
        plt.figure(figsize=(14, 10))
        params = [x[0] for x in sorted_importance]
        importances = [x[1] for x in sorted_importance]
        
        # Plot
        sns.barplot(x=importances, y=params, palette='viridis')
        plt.title('Parameter Importance Analysis', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Parameter', fontsize=12)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"parameter_importance_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter importance plot saved to {filename}")
        
        # Create parameter correlation heatmap
        plt.figure(figsize=(16, 14))
        sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Parameter Correlation Matrix', fontsize=16)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"parameter_correlation_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter correlation matrix saved to {filename}")
        
        # Create detailed plots for top parameters
        top_params = [x[0] for x in sorted_importance[:n_top_params]]
        print(f"\nTop {len(top_params)} most important parameters:")
        for param, importance in sorted_importance[:n_top_params]:
            print(f"  {param}: {importance:.4f}")
        
        # Create individual scatter plots for top parameters
        for param in top_params:
            if param in X.columns:
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot
                sns.scatterplot(x=params_df[param], y=params_df[target_metric])
                
                # Try to add regression line
                try:
                    sns.regplot(x=params_df[param], y=params_df[target_metric], 
                               scatter=False, color='red')
                except Exception:
                    pass
                
                plt.title(f'Impact of {param} on {target_metric}', fontsize=16)
                plt.xlabel(param, fontsize=12)
                plt.ylabel(target_metric, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = os.path.join(output_dir, f"param_impact_{param}_{timestamp}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Parameter impact plot saved for {param}")
        
        # Create a boxplot showing parameter distributions for top/bottom performers
        for param in top_params:
            if param in X.columns:
                plt.figure(figsize=(12, 8))
                
                # Split into top/bottom performers
                median_performance = params_df[target_metric].median()
                params_df['performance_group'] = 'Average'
                params_df.loc[params_df[target_metric] > params_df[target_metric].quantile(0.75), 'performance_group'] = 'Top Performers'
                params_df.loc[params_df[target_metric] < params_df[target_metric].quantile(0.25), 'performance_group'] = 'Bottom Performers'
                
                # Create boxplot
                sns.boxplot(x='performance_group', y=param, data=params_df, 
                           order=['Bottom Performers', 'Average', 'Top Performers'])
                
                plt.title(f'Distribution of {param} by Performance Group', fontsize=16)
                plt.xlabel('Performance Group', fontsize=12)
                plt.ylabel(param, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = os.path.join(output_dir, f"param_distribution_{param}_{timestamp}.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Parameter distribution plot saved for {param}")
        
        # Create text summary of findings
        with open(os.path.join(output_dir, f"parameter_sensitivity_summary_{timestamp}.txt"), 'w') as f:
            f.write("PARAMETER SENSITIVITY ANALYSIS SUMMARY\n")
            f.write("=====================================\n\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of data points analyzed: {len(params_df)}\n")
            f.write(f"Target performance metric: {target_metric}\n\n")
            
            f.write("PARAMETER IMPORTANCE RANKING\n")
            f.write("---------------------------\n")
            for i, (param, importance) in enumerate(sorted_importance, 1):
                f.write(f"{i}. {param}: {importance:.4f}\n")
            
            f.write("\n\nTOP PARAMETERS ANALYSIS\n")
            f.write("----------------------\n")
            for param, importance in sorted_importance[:n_top_params]:
                f.write(f"\nParameter: {param} (Importance: {importance:.4f})\n")
                
                # Calculate optimal range
                if param in X.columns:
                    # Get top 25% performers
                    top_performers = params_df[params_df[target_metric] > params_df[target_metric].quantile(0.75)]
                    
                    # Calculate parameter stats in top performers
                    param_min = top_performers[param].min()
                    param_max = top_performers[param].max()
                    param_mean = top_performers[param].mean()
                    param_median = top_performers[param].median()
                    
                    f.write(f"  Optimal range: {param_min:.4f} to {param_max:.4f}\n")
                    f.write(f"  Mean value in top performers: {param_mean:.4f}\n")
                    f.write(f"  Median value in top performers: {param_median:.4f}\n")
                    
                    # Calculate correlation with performance
                    corr = params_df[param].corr(params_df[target_metric])
                    f.write(f"  Correlation with {target_metric}: {corr:.4f}\n")
                    
                    # Interpretation
                    if corr > 0.3:
                        f.write("  Interpretation: Strong positive relationship - higher values generally lead to better performance\n")
                    elif corr < -0.3:
                        f.write("  Interpretation: Strong negative relationship - lower values generally lead to better performance\n")
                    elif abs(corr) > 0.1:
                        f.write("  Interpretation: Moderate relationship with performance\n")
                    else:
                        f.write("  Interpretation: Weak linear relationship - may have nonlinear effects or interact with other parameters\n")
            
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("--------------\n")
            f.write("Based on the analysis, consider focusing on these parameters:\n")
            for param, importance in sorted_importance[:min(5, n_top_params)]:
                f.write(f"- {param}: High importance score of {importance:.4f}\n")
            
            f.write("\nParameters with low importance that could potentially be simplified or fixed:\n")
            for param, importance in sorted_importance[-min(5, n_top_params):]:
                f.write(f"- {param}: Low importance score of {importance:.4f}\n")
        
        print(f"Parameter sensitivity summary saved to {os.path.join(output_dir, f'parameter_sensitivity_summary_{timestamp}.txt')}")
        
    except Exception as e:
        print(f"Error in parameter sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run parameter sensitivity analysis."""
    args = parse_arguments()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir is not None else args.results_dir
    
    # Load model files
    models = load_model_files(args.results_dir, args.model_file)
    
    if not models:
        print("No model files found or loaded. Exiting.")
        return
    
    # Extract parameters and performance metrics
    params_df = extract_parameters_and_performance(models)
    
    # Analyze parameter sensitivity
    analyze_parameter_sensitivity(params_df, output_dir, args.n_top_params)
    
    print("\nParameter sensitivity analysis complete.")

if __name__ == "__main__":
    main()