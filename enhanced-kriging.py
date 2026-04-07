def generate_samples_enhanced(n_samples, field_type='stationary_gaussian', seed=4):
    """Generate synthetic spatial data samples with various field types.
    
    Parameters:
        n_samples: int, number of samples to generate
        field_type: str, one of ['stationary_gaussian', 'nonstationary_variance', 
                                'student_t', 'mixture_gaussian', 'matern']
        seed: int, random seed
    """
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
    
    if field_type == 'nonstationary_variance':
        # Non-stationary variance field where variance increases with x coordinate
        spatial_cov = generate_covariance_matrix(locations)
        variance_field = np.exp(2 * locations[:, 0]).reshape(-1, 1)  # Variance increases exponentially with x
        spatial_effects = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov) * np.sqrt(variance_field).ravel()
        
    elif field_type == 'student_t':
        # Student-t random field with 4 degrees of freedom
        spatial_cov = generate_covariance_matrix(locations)
        chi2_samples = np.random.chisquare(df=4, size=len(locations))
        gaussian_field = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov)
        spatial_effects = gaussian_field / np.sqrt(chi2_samples/4)  # Student-t field
        
    elif field_type == 'mixture_gaussian':
        # Mixture of two Gaussian processes
        spatial_cov1 = generate_covariance_matrix(locations, theta=2.0)  # Shorter range
        spatial_cov2 = generate_covariance_matrix(locations, theta=0.5)  # Longer range
        mixing_weights = np.random.beta(2, 2, size=len(locations))  # Random mixing weights
        
        gp1 = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov1)
        gp2 = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov2)
        spatial_effects = mixing_weights * gp1 + (1 - mixing_weights) * gp2
        
    elif field_type == 'matern':
        # Matérn covariance function with varying smoothness
        def matern_cov(h, nu=1.5, rho=0.3):
            h = np.maximum(h, 1e-10)  # Avoid division by zero
            sqrt_2nu_h = np.sqrt(2 * nu) * h / rho
            return (2**(1-nu)/scipy.special.gamma(nu)) * (sqrt_2nu_h**nu) * scipy.special.kv(nu, sqrt_2nu_h)
        
        n = len(locations)
        spatial_cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = np.sqrt(np.sum((locations[i] - locations[j])**2))
                spatial_cov[i,j] = matern_cov(dist)
                
        spatial_effects = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov)
    
    else:  # Default to stationary Gaussian
        spatial_cov = generate_covariance_matrix(locations)
        spatial_effects = np.random.multivariate_normal(np.zeros(len(locations)), spatial_cov)
    
    # Combine components
    var1 = (mean1(locations[:,0], locations[:,1]) + 
            0.3 * np.sum(covariates, axis=1) + 
            spatial_effects + 
            np.random.normal(0, 0.1, len(locations)))
    
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

def main():
    n_samples = 1156  # 34x34 grid
    n_simulations = 50
    
    # Define field types to simulate
    field_types = ['stationary_gaussian', 'nonstationary_variance', 
                  'student_t', 'mixture_gaussian', 'matern']
    
    # Create directories
    for field_type in field_types:
        os.makedirs(f"synthetic_data/synthetic_{field_type}", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize MSE storage for each field type
    mse_results = {field_type: [] for field_type in field_types}
    
    for sim in range(n_simulations):
        print(f"\nSimulation {sim + 1} of {n_simulations}")
        
        for field_type in field_types:
            print(f"Generating {field_type} samples...")
            df = generate_samples_enhanced(n_samples, field_type=field_type, seed=sim)
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save data
            train.to_csv(f"synthetic_data/synthetic_{field_type}/2d_nongaussian_{sim+1}-train.csv", 
                        index=False)
            test.to_csv(f"synthetic_data/synthetic_{field_type}/2d_nongaussian_{sim+1}-test.csv", 
                       index=False)
            
            print(f"Running {field_type} simulation...")
            predictions, ci_lower, ci_upper, y_test, mse_vals = run_simulation(
                f"synthetic_data/synthetic_{field_type}/2d_nongaussian_{sim+1}",
                is_bivariate=False
            )
            plot_results(predictions, ci_lower, ci_upper, y_test, is_bivariate=False)
            mse_results[field_type].append(mse_vals[0])
    
    # Save MSE results
    pd.DataFrame(mse_results).to_csv("results/field_types_mse.csv", index=False)
    
    # Print average MSE for each field type
    print("\nAverage MSE by field type:")
    for field_type in field_types:
        print(f"{field_type}: {np.mean(mse_results[field_type]):.4f}")
