import tensorflow as tf
from keras.layers import Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

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
