import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal, norm

def generate_samples(n_samples, is_bivariate=True, seed=42):
    """Generate synthetic spatial data samples."""
    np.random.seed(seed)
    
    # Generate grid points
    x = np.linspace(0, 1, int(np.sqrt(n_samples)))
    y = np.linspace(0, 1, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, y)
    locations = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Generate covariates
    n_covariates = 5
    covariates = np.random.normal(0, 1, (n_samples, n_covariates))
    
    # Define mean functions
    mean1 = lambda x, y: 2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    mean2 = lambda x, y: np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
    # Generate spatial random effects
    if is_bivariate:
        # Bivariate case
        cov_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        spatial_effects = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            dist_matrix = np.exp(-3 * np.sum((locations - locations[i])**2, axis=1))
            mv_normal = multivariate_normal(mean=[0, 0], cov=cov_matrix)
            spatial_effects[i] = mv_normal.rvs()
            spatial_effects[i] *= dist_matrix[i]
        
        # Combine mean function, spatial effects, and covariate effects
        var1 = mean1(locations[:,0], locations[:,1]) + \
               0.3 * np.sum(covariates, axis=1) + \
               spatial_effects[:,0] + \
               np.random.normal(0, 0.1, n_samples)
        
        var2 = mean2(locations[:,0], locations[:,1]) + \
               0.3 * np.sum(covariates, axis=1) + \
               spatial_effects[:,1] + \
               np.random.normal(0, 0.1, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': locations[:,0],
            'y': locations[:,1],
            'var1': var1,
            'var2': var2
        })
        
        # Add covariates
        for i in range(n_covariates):
            df[f'cov{i+1}'] = covariates[:,i]
            
    else:
        # Univariate case
        spatial_effects = np.zeros(n_samples)
        
        for i in range(n_samples):
            dist_matrix = np.exp(-3 * np.sum((locations - locations[i])**2, axis=1))
            spatial_effects[i] = norm.rvs(loc=0, scale=1)
            spatial_effects[i] *= dist_matrix[i]
        
        # Combine mean function, spatial effects, and covariate effects
        var1 = mean1(locations[:,0], locations[:,1]) + \
               0.3 * np.sum(covariates, axis=1) + \
               spatial_effects + \
               np.random.normal(0, 0.1, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': locations[:,0],
            'y': locations[:,1],
            'var1': var1
        })
        
        # Add covariates
        for i in range(n_covariates):
            df[f'cov{i+1}'] = covariates[:,i]
    
    return df

[Previous functions remain the same...]

def main():
    # Generate samples
    n_samples = 1200
    
    # Bivariate case
    df_biv = generate_samples(n_samples, is_bivariate=True)
    train_biv, test_biv = train_test_split(df_biv, test_size=0.2, random_state=42)
    
    # Save bivariate data
    train_biv.to_csv("synthetic_bivariate_train.csv", index=False)
    test_biv.to_csv("synthetic_bivariate_test.csv", index=False)
    
    # Run bivariate simulation
    predictions_biv, ci_lower_biv, ci_upper_biv, y_test_biv = run_simulation(
        "synthetic_bivariate",
        is_bivariate=True
    )
    plot_results(predictions_biv, ci_lower_biv, ci_upper_biv, y_test_biv, is_bivariate=True)
    
    # Univariate case
    df_uni = generate_samples(n_samples, is_bivariate=False)
    train_uni, test_uni = train_test_split(df_uni, test_size=0.2, random_state=42)
    
    # Save univariate data
    train_uni.to_csv("synthetic_univariate_train.csv", index=False)
    test_uni.to_csv("synthetic_univariate_test.csv", index=False)
    
    # Run univariate simulation
    predictions_uni, ci_lower_uni, ci_upper_uni, y_test_uni = run_simulation(
        "synthetic_univariate",
        is_bivariate=False
    )
    plot_results(predictions_uni, ci_lower_uni, ci_upper_uni, y_test_uni, is_bivariate=False)

if __name__ == "__main__":
    main()
