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

import tensorflow as tf
from keras.layers import Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

def generate_non_stationary_univariate(locations, num_basis, knots_1d):
    K = sum(num_basis)
    # Generate random coefficients
    a = np.random.uniform(-2.5, 2.5, K)
    b = np.random.uniform(-2.5, 2.5, K)
    
    # Get basis functions
    phi = get_basis_functions(locations, num_basis, knots_1d)
    
    # Generate field using the formula from PDF
    var1 = np.zeros(len(locations))
    for j in range(K-1):
        var1 += a[j] * phi[:,j]**1.5 - b[j] * np.sqrt(phi[:,j] * phi[:,j+1])
        
    return var1

def generate_non_stationary_bivariate(locations, num_basis, knots_1d):
    K = sum(num_basis)
    # Generate random coefficients
    a = np.random.uniform(-2.5, 2.5, K)
    b = np.random.uniform(-2.5, 2.5, K)
    c = np.random.uniform(-2.5, 2.5, K)
    d = np.random.uniform(-2.5, 2.5, K)
    f = np.random.uniform(-2.5, 2.5, K)
    
    # Get basis functions
    phi = get_basis_functions(locations, num_basis, knots_1d)
    
    # Generate fields using formulas from PDF
    var1 = np.zeros(len(locations))
    var2 = np.zeros(len(locations))
    
    for j in range(K//2):
        # Z1(s) formula
        var1 += (a[j] * phi[:,2*j]**1.5 + 
                c[j] * phi[:,2*j-1] - 
                b[j] * np.sqrt(phi[:,2*j] * phi[:,2*j-1]))
        
        # Z2(s) formula
        var2 += (d[j] * phi[:,2*j] - 
                f[j] * phi[:,2*j-1]**1.5)
    
    return var1, var2

def create_enhanced_model(input_dim, output_dim, dropout_rate=0.2):
    """
    Creates an enhanced neural network with advanced architecture features.
    """
    model = Sequential([
        # First block with larger units
        Dense(256, input_dim=input_dim, kernel_initializer='he_normal', 
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        
        # Second block with moderate units
        Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate),
        
        # Third block
        Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(dropout_rate/2),  # Lower dropout in deeper layers
        
        # Output layer
        Dense(output_dim, activation='linear', kernel_initializer='he_normal')
    ])
    
    return model

def fit_enhanced_ensemble(n_members, X_train, y_train, val_split=0.2):
    """
    Creates and trains an enhanced ensemble of neural networks with better regularization
    and training strategies.
    """
    ensemble = []
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            min_delta=1e-4
        )
    ]
    
    for i in range(n_members):
        # Create model with random dropout rate
        dropout_rate = np.random.uniform(0.1, 0.3)
        model = create_enhanced_model(input_dim, output_dim, dropout_rate)
        
        # Use different optimizers for diversity
        if i % 3 == 0:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
        elif i % 3 == 1:
            optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        else:
            optimizer = keras.optimizers.Adamax(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MAE
            metrics=['mae', 'mse']
        )
        
        # Random subsampling with replacement
        idx = np.random.choice(len(X_train), size=int(0.8*len(X_train)), replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        
        # Train with validation split
        model.fit(
            X_boot, y_boot,
            epochs=1000,  # Increased epochs since we have early stopping
            batch_size=32,  # Smaller batch size for better generalization
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )
        
        ensemble.append(model)
        
    return ensemble

def predict_with_enhanced_pi(ensemble, X, n_samples=100):
    """
    Enhanced prediction with uncertainty estimation using MC Dropout
    """
    all_predictions = []
    
    for model in ensemble:
        # Multiple forward passes with dropout enabled
        model.layers[3].rate = 0.1  # Set dropout rate for prediction
        temp_pred = np.array([model.predict(X, verbose=0) for _ in range(n_samples)])
        all_predictions.append(temp_pred)
    
    # Combine predictions from all models and MC samples
    all_predictions = np.array(all_predictions)  # (n_models, n_samples, n_points, n_outputs)
    
    # Calculate mean and variance
    mean_pred = np.mean(all_predictions, axis=(0,1))  # Average over models and MC samples
    var_pred = np.var(all_predictions, axis=(0,1))    # Variance over models and MC samples
    
    if mean_pred.shape[1] == 2:  # Bivariate case
        return mean_pred[:,0], var_pred[:,0], mean_pred[:,1], var_pred[:,1]
    else:  # Univariate case
        return mean_pred[:,0], var_pred[:,0]

def generate_covariance_matrix(locations, theta=3.0):
    """Generate spatial covariance matrix."""
    n = len(locations)
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = np.sum((locations[i] - locations[j])**2)
            cov_matrix[i,j] = np.exp(-theta * dist)
    return cov_matrix

def generate_samples(n_samples, is_bivariate=True, seed=42):
    """Generate synthetic spatial data samples."""
    np.random.seed(seed)
    
    # Generate grid points
    n_per_side = int(np.sqrt(n_samples))
    x = np.linspace(0, 1, n_per_side)
    y = np.linspace(0, 1, n_per_side)
    xx, yy = np.meshgrid(x, y)
    locations = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Generate covariates
    n_covariates = 5
    covariates = np.random.normal(0, 1, (len(locations), n_covariates))
    
    # Define mean functions
    mean1 = lambda x, y: 2 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    mean2 = lambda x, y: np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
    # Generate spatial correlation matrix
    spatial_cov = generate_covariance_matrix(locations)
    
    num_basis = [2**2, 3**2, 5**2]  # as specified in PDF
    knots_1d = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    phi = get_basis_functions(locations, num_basis, knots_1d)
        
    if is_bivariate:
        # Generate coefficients
        K = sum(num_basis)
        a = np.random.uniform(-2.5, 2.5, K)
        b = np.random.uniform(-2.5, 2.5, K)
        c = np.random.uniform(-2.5, 2.5, K)
        d = np.random.uniform(-2.5, 2.5, K)
        f = np.random.uniform(-2.5, 2.5, K)
            
        # Generate fields
        var1 = np.zeros(len(locations))
        var2 = np.zeros(len(locations))
            
        for j in range(K//2):
            if j > 0:  # Avoid index 0
                var1 += (a[j] * phi[:,2*j]**1.5 + 
                        c[j] * phi[:,2*j-1] - 
                        b[j] * np.sqrt(np.maximum(phi[:,2*j] * phi[:,2*j-1], 0)))
                var2 += (d[j] * phi[:,2*j] - 
                        f[j] * phi[:,2*j-1]**1.5)
        df = pd.DataFrame({
            'x': locations[:,0],
            'y': locations[:,1],
            'var1': var1,
            'var2': var2
        })

    else:
        # Generate coefficients
        K = sum(num_basis)
        a = np.random.uniform(-2.5, 2.5, K-1)
        b = np.random.uniform(-2.5, 2.5, K-1)
            
        # Generate field
        var1 = np.zeros(len(locations))
        for j in range(K-1):
            var1 += (a[j] * phi[:,j]**1.5 - 
                    b[j] * np.sqrt(np.maximum(phi[:,j] * phi[:,j+1], 0)))
        df = pd.DataFrame({
            'x': locations[:,0],
            'y': locations[:,1],
            'var1': var1
        })
    # Add covariates
    for i in range(n_covariates):
        df[f'cov{i+1}'] = covariates[:,i]
        
    return df
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

def mse(y_pred, y_true):
    return np.mean((y_pred-y_true)**2)

def get_basis_functions(s, num_basis, knots_1d):
    N = len(s)
    K = 0
    phi = np.zeros((N, sum(num_basis)))
    
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_s1, knots_s2 = np.meshgrid(knots_1d[res], knots_1d[res])
        knots = np.column_stack((knots_s1.flatten(), knots_s2.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(s-knots[i,:], axis=1)/theta
            for j in range(len(d)):
                if 0 <= d[j] <= 1:
                    phi[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
        K += num_basis[res]
    return phi

def fit_ensemble(n_members, X_train, y_train, base_model):
    ensemble = []
    for i in range(n_members):
        model = Sequential()
        model.add(Dense(100, input_dim=X_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='mae')
        
        # Randomly sample 80% of training data
        idx = np.random.choice(len(X_train), int(0.8*len(X_train)), replace=False)
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        
        model.fit(X_boot, y_boot, epochs=500, batch_size=64, verbose=0)
        ensemble.append(model)
    return ensemble

def predict_with_pi(ensemble, X):
    predictions = np.array([model.predict(X) for model in ensemble])
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0)
    
    if predictions.shape[2] == 2:  # Bivariate case
        return mean_pred[:,0], var_pred[:,0], mean_pred[:,1], var_pred[:,1]
    else:  # Univariate case
        return mean_pred[:,0], var_pred[:,0]

def run_simulation(data_path, is_bivariate=True):
    # Load and prepare data
    df_train = pd.read_csv(data_path + "-train.csv")
    df_test = pd.read_csv(data_path + "-test.csv")
    
    # Prepare covariates
    covariates_train = np.array(df_train[["cov1", "cov2", "cov3", "cov4", "cov5"]])
    covariates_test = np.array(df_test[["cov1", "cov2", "cov3", "cov4", "cov5"]])
    
    scaler = MinMaxScaler()
    covariates_train = scaler.fit_transform(covariates_train)
    covariates_test = scaler.transform(covariates_test)
    
    # Prepare spatial coordinates
    s_train = np.vstack((df_train["x"], df_train["y"])).T
    s_test = np.vstack((df_test["x"], df_test["y"])).T
    
    # Prepare target variables
    if is_bivariate:
        y_train = np.array(df_train[["var1", "var2"]])
        y_test = np.array(df_test[["var1", "var2"]])
        variance_vars = [np.var(df_train["var1"]), np.var(df_train["var2"])]
        means = [np.mean(df_train["var1"]), np.mean(df_train["var2"])]
    else:
        y_train = np.array(df_train["var1"]).reshape(-1, 1)
        y_test = np.array(df_test["var1"]).reshape(-1, 1)
        variance_vars = [np.var(df_train["var1"])]
        means = [np.mean(df_train["var1"])]
    
    # Standardize target variables
    for i in range(len(variance_vars)):
        y_train[:,i] = (y_train[:,i] - means[i])/np.sqrt(variance_vars[i])
    
    # Generate basis functions
    num_basis = [3**2, 7**2, 11**2]
    knots_1d = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    phi_train = get_basis_functions(s_train, num_basis, knots_1d)
    phi_test = get_basis_functions(s_test, num_basis, knots_1d)
    
    # Combine basis functions with covariates
    phi_train = np.hstack((covariates_train, phi_train))
    phi_test = np.hstack((covariates_test, phi_test))
    
    # Split data for ensemble
    s_train_ensemble, s_train_mse, X_train_ensemble, X_train_mse, y_train_ensemble, y_train_mse = train_test_split(
        s_train, phi_train, y_train, test_size=0.1, random_state=42
    )
    
    # # Create and train ensemble
    # base_model = Sequential([
    #     Dense(100, input_dim=phi_train.shape[1], kernel_initializer='he_uniform', activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(100, activation='relu'),
    #     Dense(y_train.shape[1], activation='linear')
    # ])
    
    # ensemble = fit_ensemble(20, X_train_ensemble, y_train_ensemble, base_model)
    
    # Replace the original ensemble creation with:
    ensemble = fit_enhanced_ensemble(20, phi_train, y_train, val_split=0.2)
    

    # Generate predictions with uncertainty
    if is_bivariate:
        mean_vec1, var_vec1, mean_vec2, var_vec2 = predict_with_enhanced_pi(ensemble, phi_test)
        predictions = np.column_stack([
            mean_vec1 * np.sqrt(variance_vars[0]) + means[0],
            mean_vec2 * np.sqrt(variance_vars[1]) + means[1]
        ])
        
        ci_lower = np.column_stack([
            mean_vec1 * np.sqrt(variance_vars[0]) + means[0] - 1.96 * np.sqrt(var_vec1) * np.sqrt(variance_vars[0]),
            mean_vec2 * np.sqrt(variance_vars[1]) + means[1] - 1.96 * np.sqrt(var_vec2) * np.sqrt(variance_vars[1])
        ])
        ci_upper = np.column_stack([
            mean_vec1 * np.sqrt(variance_vars[0]) + means[0] + 1.96 * np.sqrt(var_vec1) * np.sqrt(variance_vars[0]),
            mean_vec2 * np.sqrt(variance_vars[1]) + means[1] + 1.96 * np.sqrt(var_vec2) * np.sqrt(variance_vars[1])
        ])
    else:
        mean_vec, var_vec = predict_with_enhanced_pi(ensemble, phi_test)
        predictions = mean_vec * np.sqrt(variance_vars[0]) + means[0]
        
        ci_lower = mean_vec * np.sqrt(variance_vars[0]) + means[0] - 1.96 * np.sqrt(var_vec) * np.sqrt(variance_vars[0])
        ci_upper = mean_vec * np.sqrt(variance_vars[0]) + means[0] + 1.96 * np.sqrt(var_vec) * np.sqrt(variance_vars[0])
    
    # Calculate MSE for reporting
    mse_values = []
    if is_bivariate:
        mse_values = [mse(predictions[:,0], y_test[:,0]), mse(predictions[:,1], y_test[:,1])]
    else:
        mse_values = [mse(predictions, y_test)]

    # Calculate additional metrics
    interval_lengths = ci_upper - ci_lower
    point_mses = (predictions - y_test) ** 2  # MSE for each point
    coverage = np.logical_and(y_test >= ci_lower, y_test <= ci_upper)

    if is_bivariate:
        # Ensure all arrays have the same shape by reshaping y_test if needed
        y_test_reshaped = y_test if y_test.shape == predictions.shape else y_test.reshape(predictions.shape)
    
        metrics_df = pd.DataFrame({
            'point_mse_var1': point_mses[:,0].ravel(),
            'point_mse_var2': point_mses[:,1].ravel(),
            'interval_length_var1': interval_lengths[:,0].ravel(),
            'interval_length_var2': interval_lengths[:,1].ravel(),
            'coverage_var1': coverage[:,0].ravel(),
            'coverage_var2': coverage[:,1].ravel(),
            'true_value_var1': y_test_reshaped[:,0].ravel(),
            'true_value_var2': y_test_reshaped[:,1].ravel(),
            'predicted_value_var1': predictions[:,0].ravel(),
            'predicted_value_var2': predictions[:,1].ravel()
        })
    else:
        # Ensure all arrays are properly shaped
        predictions_flat = predictions.ravel()
        y_test_flat = y_test.ravel()
        ci_lower_flat = ci_lower.ravel()
        ci_upper_flat = ci_upper.ravel()
    
        metrics_df = pd.DataFrame({
            'point_mse': (predictions_flat - y_test_flat) ** 2,
            'interval_length': (ci_upper_flat - ci_lower_flat),
            'coverage': np.logical_and(y_test_flat >= ci_lower_flat, 
                                 y_test_flat <= ci_upper_flat),
            'true_value': y_test_flat,
            'predicted_value': predictions_flat
        })

    # Save metrics
    metrics_filename = f"{data_path}-prediction_metrics_nonstationary.csv"
    metrics_df.to_csv(metrics_filename, index=False)

    # Calculate and save summary statistics
    if is_bivariate:
        summary_metrics = {
            'average_coverage_var1': np.mean(metrics_df['coverage_var1']),
            'average_coverage_var2': np.mean(metrics_df['coverage_var2']),
            'average_interval_length_var1': np.mean(metrics_df['interval_length_var1']),
            'average_interval_length_var2': np.mean(metrics_df['interval_length_var2']),
            'average_point_mse_var1': np.mean(metrics_df['point_mse_var1']),
            'average_point_mse_var2': np.mean(metrics_df['point_mse_var2'])
        }
    else:
        summary_metrics = {
            'average_coverage': np.mean(metrics_df['coverage']),
            'average_interval_length': np.mean(metrics_df['interval_length']),
            'average_point_mse': np.mean(metrics_df['point_mse'])
        }

    summary_filename = f"{data_path}-summary_metrics_nonstationary.csv"
    pd.DataFrame([summary_metrics]).to_csv(summary_filename, index=False)
    return predictions, ci_lower, ci_upper, y_test, mse_values, metrics_df, summary_metrics

def plot_results(predictions, ci_lower, ci_upper, y_test, is_bivariate=True):

    if is_bivariate:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Variable 1
        ax1.scatter(range(len(y_test)), y_test[:,0], alpha=0.5, label='True')
        ax1.plot(predictions[:,0], 'r-', label='Predicted')
        ax1.fill_between(range(len(y_test)), ci_lower[:,0], ci_upper[:,0], color='r', alpha=0.2)
        ax1.set_title('Variable 1')
        ax1.legend()
        
        # Variable 2
        ax2.scatter(range(len(y_test)), y_test[:,1], alpha=0.5, label='True')
        ax2.plot(predictions[:,1], 'r-', label='Predicted')
        ax2.fill_between(range(len(y_test)), ci_lower[:,1], ci_upper[:,1], color='r', alpha=0.2)
        ax2.set_title('Variable 2')
        ax2.legend()
        
    else:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, alpha=0.5, label='True')
        plt.plot(predictions, 'r-', label='Predicted')
        plt.fill_between(range(len(y_test)), ci_lower, ci_upper, color='r', alpha=0.2)
        plt.title('Univariate Prediction with Confidence Intervals')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    n_samples = 1156  # 34x34 grid
    n_simulations = 50
    
    # Create directories
    os.makedirs("synthetic_data/synthetic_univariate", exist_ok=True)
    os.makedirs("synthetic_data/synthetic_bivariate", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize MSE storage
    mse_results_biv = {'var1': [], 'var2': []}
    mse_results_uni = []
    
    for sim in range(n_simulations):
        print(f"\nSimulation {sim + 1} of {n_simulations}")
        
        # Bivariate case
        print("Generating bivariate samples...")
        df_biv = generate_samples(n_samples, is_bivariate=True, seed=sim)
        train_biv, test_biv = train_test_split(df_biv, test_size=0.2, random_state=42)
        
        # Save bivariate data
        train_biv.to_csv(f"synthetic_data/synthetic_bivariate/2d_nonstationary_{sim+1}-train.csv", index=False)
        test_biv.to_csv(f"synthetic_data/synthetic_bivariate/2d_nonstationary_{sim+1}-test.csv", index=False)
        
        print("Running bivariate simulation...")
        predictions_biv, ci_lower_biv, ci_upper_biv, y_test_biv, mse_biv, _, _ = run_simulation(
            f"synthetic_data/synthetic_bivariate/2d_nonstationary_{sim+1}",
            is_bivariate=True
        )
        # plot_results(predictions_biv, ci_lower_biv, ci_upper_biv, y_test_biv, is_bivariate=True)
        mse_results_biv['var1'].append(mse_biv[0])
        mse_results_biv['var2'].append(mse_biv[1])
        
        # Univariate case
        print("Generating univariate samples...")
        df_uni = generate_samples(n_samples, is_bivariate=False, seed=sim)
        train_uni, test_uni = train_test_split(df_uni, test_size=0.2, random_state=42)
        
        # Save univariate data
        train_uni.to_csv(f"synthetic_data/synthetic_univariate/2d_nonstationary_{sim+1}-train.csv", index=False)
        test_uni.to_csv(f"synthetic_data/synthetic_univariate/2d_nonstationary_{sim+1}-test.csv", index=False)
        
        print("Running univariate simulation...")
        predictions_uni, ci_lower_uni, ci_upper_uni, y_test_uni, mse_uni, _, _ = run_simulation(
            f"synthetic_data/synthetic_univariate/2d_nonstationary_{sim+1}",
            is_bivariate=False
        )
        # plot_results(predictions_uni, ci_lower_uni, ci_upper_uni, y_test_uni, is_bivariate=False)
        mse_results_uni.append(mse_uni[0])
    
    # Save MSE results
    pd.DataFrame(mse_results_biv).to_csv("results/bivariate_mse_nonstationary.csv", index=False)
    pd.DataFrame({'mse': mse_results_uni}).to_csv("results/univariate_mse_nonstationary.csv", index=False)
    
    # Print average MSE
    print("\nAverage MSE:")
    print(f"Bivariate - Var1: {np.mean(mse_results_biv['var1']):.4f}")
    print(f"Bivariate - Var2: {np.mean(mse_results_biv['var2']):.4f}")
    print(f"Univariate: {np.mean(mse_results_uni):.4f}")

if __name__ == "__main__":
    main()