import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import math
import os
import joblib
import datetime  # Make sure datetime is imported directly, not just datetime.datetime
from tkinter import Tk, filedialog
import optuna
import csv
from tensorflow.keras.regularizers import l1_l2
import sys
import subprocess

# Try to import docx - install if not available
try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    print("python-docx package not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    # Import again after installation
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

# Custom layer for feature weighting
class FeatureWeightingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Create a trainable weight for each feature
        self.feature_weights = self.add_weight(
            name='feature_weights',
            shape=(input_shape[-1],),
            initializer='ones',
            constraint=tf.keras.constraints.NonNeg(),  # Ensures weights are non-negative
            trainable=True
        )
        super(FeatureWeightingLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Apply weights to features (element-wise multiplication)
        return inputs * self.feature_weights
        
    def get_config(self):
        config = super(FeatureWeightingLayer, self).get_config()
        return config

class TemporalFusionTransformer:
    def __init__(self, 
                 input_seq_length=16,
                 forecast_horizons=4,
                 hidden_size=96,
                 attention_heads=3,
                 dropout_rate=0.15,
                 batch_size=16,
                 learning_rate=0.002566076505216372,
                 max_epochs=50):
        """Initialize the Temporal Fusion Transformer model for solar forecasting.
        
        This model provides probabilistic forecasts with 95% confidence intervals that
        are optimized to be as narrow as possible while maintaining the specified coverage.
        The Coverage Width-based Criterion (CWC) is calculated using Prediction Interval
        Normalized Average Width (PINAW) to balance coverage and interval width.
        
        Args:
            input_seq_length (int): Number of time steps in input sequence
            forecast_horizons (int): Number of time steps to forecast ahead
            hidden_size (int): Size of hidden layers in the model
            attention_heads (int): Number of attention heads in multi-head attention
            dropout_rate (float): Dropout rate for regularization
            batch_size (int): Mini-batch size for training
            learning_rate (float): Learning rate for optimizer
            max_epochs (int): Maximum number of training epochs
        """
        
        self.input_seq_length = input_seq_length
        self.forecast_horizons = forecast_horizons
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.history = None
        self.feature_importance = None
        self.feature_weights = None  # New attribute to store learned feature weights
    
    def _create_tft_model(self, input_shape, output_shape, feature_names=None):
        """Create TFT model architecture with learned feature weights"""
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Add feature weighting layer if feature names are provided
        if feature_names is not None and len(feature_names) == input_shape[1]:
            # Apply feature weighting using our custom layer
            weighted_inputs = FeatureWeightingLayer(name='feature_weights')(inputs)
        else:
            # If feature names not provided, don't use feature weighting
            weighted_inputs = inputs
        
        # LSTM encoders for sequential processing with bidirectional wrapper
        lstm1 = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(weighted_inputs)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # Self-attention mechanism
        query = Dense(self.hidden_size)(lstm1)
        key = Dense(self.hidden_size)(lstm1)
        value = Dense(self.hidden_size)(lstm1)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=self.hidden_size // self.attention_heads
        )(query, key, value)
        
        attention_output = Dropout(self.dropout_rate)(attention_output)
        lstm_attention_concat = Concatenate()([lstm1, attention_output])
        
        # Skip connections and layer normalization
        normalized = LayerNormalization()(lstm_attention_concat)
        
        # Final LSTM layer with increased capacity
        lstm_final = LSTM(self.hidden_size*2)(normalized)
        lstm_final = LayerNormalization()(lstm_final)
        lstm_final = Dropout(self.dropout_rate)(lstm_final)
        
        # Output layers - one for each forecast horizon
        # Each horizon produces 3 outputs: mean, lower bound (2.5%), upper bound (97.5%)
        outputs = []
        for h in range(self.forecast_horizons):
            # Shared layers for each horizon - deeper network
            horizon_specific = Dense(self.hidden_size, activation='elu')(lstm_final)
            horizon_specific = LayerNormalization()(horizon_specific)
            horizon_specific = Dropout(self.dropout_rate/2)(horizon_specific)
            horizon_specific = Dense(self.hidden_size//2, activation='elu')(horizon_specific)
            horizon_specific = Dropout(self.dropout_rate/2)(horizon_specific)
            
            # Mean prediction
            mean = Dense(1, name=f'horizon_{h+1}_mean')(horizon_specific)
            
            # Lower and upper bounds for 95% prediction interval (using 2.5% and 97.5% quantiles)
            lower_bound = Dense(1, name=f'horizon_{h+1}_lower')(horizon_specific)
            upper_bound = Dense(1, name=f'horizon_{h+1}_upper')(horizon_specific)
            
            outputs.extend([mean, lower_bound, upper_bound])
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Custom loss function - combination of MSE and quantile loss
        losses = {}
        loss_weights = {}
        
        # Give 95% weight to the 1-hour forecast as specified
        horizon_weights = [0.95, 0.03, 0.01, 0.01]  # 95% on first horizon, 5% on remaining
        
        for h in range(self.forecast_horizons):
            # Use Huber loss for mean prediction for better robustness to outliers
            losses[f'horizon_{h+1}_mean'] = tf.keras.losses.Huber(delta=1.0)
            loss_weights[f'horizon_{h+1}_mean'] = horizon_weights[h]
            
            # Quantile losses for lower (2.5%) and upper (97.5%) bounds for 95% prediction interval
            losses[f'horizon_{h+1}_lower'] = self._quantile_loss(0.025)  # Changed from 0.1
            losses[f'horizon_{h+1}_upper'] = self._quantile_loss(0.975)  # Changed from 0.9
            
            loss_weights[f'horizon_{h+1}_lower'] = horizon_weights[h] * 0.5
            loss_weights[f'horizon_{h+1}_upper'] = horizon_weights[h] * 0.5
        
        # Add gradient clipping to optimizer for stability
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0  # Add gradient clipping
            ),
            loss=losses,
            loss_weights=loss_weights
        )
        
        return model
    
    def _quantile_loss(self, q):
        """Custom quantile loss function for probabilistic forecasting"""
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
        return loss
    
    def _prepare_sequences(self, data, target_col):
        """Prepare input sequences and multi-horizon targets"""
        X, y_mean, y_lower, y_upper = [], [], [], []
        
        # IMPORTANT: Create a copy of data without the target column for input sequences
        # This ensures X will have exactly the number of features expected by the model
        features_only = data.drop(columns=[target_col])
        
        # Adjust for multi-horizon forecasting
        for i in range(len(data) - self.input_seq_length - self.forecast_horizons + 1):
            # Input sequence - ONLY use features (not target) for X
            X.append(features_only[i:(i + self.input_seq_length)].values)
            
            # Output targets for each horizon
            horizons_mean, horizons_lower, horizons_upper = [], [], []
            
            for h in range(self.forecast_horizons):
                target_idx = i + self.input_seq_length + h
                target_value = data.iloc[target_idx][target_col]
                
                horizons_mean.append(target_value)
                # For initial training, set lower/upper bounds around the mean
                # These will be refined by the model during training
                horizons_lower.append(target_value)
                horizons_upper.append(target_value)
            
            y_mean.append(horizons_mean)
            y_lower.append(horizons_lower)
            y_upper.append(horizons_upper)
        
        X = np.array(X)
        
        # Reshape targets for model output format
        targets = []
        for h in range(self.forecast_horizons):
            # Extract each horizon's targets
            h_mean = np.array([y[h] for y in y_mean]).reshape(-1, 1)
            h_lower = np.array([y[h] for y in y_lower]).reshape(-1, 1)
            h_upper = np.array([y[h] for y in y_upper]).reshape(-1, 1)
            
            targets.extend([h_mean, h_lower, h_upper])
        
        return X, targets
    
    def fit(self, df, target_col='GHI - W/m^2', validation_split=0.2, verbose=2, callbacks=None):
        """Train the TFT model on the provided dataset"""
        
        # Filter out non-numeric columns
        non_numeric_cols = ['Date', 'Start Period', 'End Period', 'Timestamp']
        numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
        
        # Keep only the target column and numeric columns
        df_numeric = df[numeric_cols]
        
        # DISABLED: Additional feature creation
        # The following code is commented out to prevent automatic feature creation
        """
        # Create additional lag features for the target
        if target_col in df_numeric.columns:
            # Create multiple lag features at various intervals
            for lag in [1, 2, 3, 6, 12, 24]:
                lag_name = f'{target_col}_lag_{lag}'
                if lag_name not in df_numeric.columns:  # Only create if it doesn't exist
                    print(f"Creating lag feature: {lag_name}")
                    df_numeric[lag_name] = df_numeric[target_col].shift(lag).fillna(0)
                
        # Add time features for better temporal information
        if isinstance(df.index, pd.DatetimeIndex):
            # Add day of week and hour sin/cos for cyclical representation
            if 'day_of_week' not in df_numeric.columns:
                print("Creating day_of_week feature")
                df_numeric['day_of_week'] = df.index.dayofweek
                
            if 'is_weekend' not in df_numeric.columns:
                print("Creating is_weekend feature")
                df_numeric['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
                
            # Hour as sine and cosine for cyclical representation
            if 'hour_sin' not in df_numeric.columns:
                print("Creating hour_sin and hour_cos features")
                hours = df.index.hour
                df_numeric['hour_sin'] = np.sin(2 * np.pi * hours / 24)
                df_numeric['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        """
        
        # Data splitting - training and validation
        train_size = int(len(df_numeric) * (1 - validation_split))
        train_data = df_numeric.iloc[:train_size]
        val_data = df_numeric.iloc[train_size:]
        
        # Separate features and target for proper scaling
        feature_cols = [col for col in df_numeric.columns if col != target_col]
        
        # Fit the scalers
        self.feature_scaler.fit(train_data[feature_cols])
        self.target_scaler.fit(train_data[[target_col]])
        
        # Scale features and target separately
        train_features_scaled = pd.DataFrame(
            self.feature_scaler.transform(train_data[feature_cols]),
            columns=feature_cols,
            index=train_data.index
        )
        
        val_features_scaled = pd.DataFrame(
            self.feature_scaler.transform(val_data[feature_cols]),
            columns=feature_cols,
            index=val_data.index
        )
        
        # Scale target
        train_target_scaled = pd.DataFrame(
            self.target_scaler.transform(train_data[[target_col]]),
            columns=[target_col],
            index=train_data.index
        )
        
        val_target_scaled = pd.DataFrame(
            self.target_scaler.transform(val_data[[target_col]]),
            columns=[target_col],
            index=val_data.index
        )
        
        # Combine scaled features and target
        train_data_scaled = pd.concat([train_features_scaled, train_target_scaled], axis=1)
        val_data_scaled = pd.concat([val_features_scaled, val_target_scaled], axis=1)
        
        # Prepare sequences
        X_train, y_train = self._prepare_sequences(train_data_scaled, target_col)
        X_val, y_val = self._prepare_sequences(val_data_scaled, target_col)
        
        # Create and compile the model - pass only feature names (excluding target)
        input_shape = (self.input_seq_length, len(feature_cols))
        output_shape = self.forecast_horizons
        
        # Pass feature names to the model creation function
        if self.model is None:
            self.model = self._create_tft_model(input_shape, output_shape, feature_cols)
        
        # Callbacks with improved early stopping and learning rate scheduling
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint('tft_model.keras', save_best_only=True),
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Extract learned feature weights
        self._extract_feature_weights(feature_cols)
        
        return self.history
    
    def _extract_feature_weights(self, feature_names):
        """Extract learned feature weights from the model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        try:
            # Get the feature weights layer
            feature_weights_layer = self.model.get_layer('feature_weights')
            
            # Extract the weights
            weights = feature_weights_layer.get_weights()[0]
            
            # Create a DataFrame with feature names and their weights
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.abs(weights)  # Take absolute value to ensure positive importance
            })
            
            # Store the raw weights for potential use during prediction
            self.feature_weights = weights
            
            # Sort by importance for visualization
            self.feature_importance = self.feature_importance.sort_values('Importance', ascending=False)
            
        except ValueError:
            print("Warning: Could not extract feature weights. Using placeholder values instead.")
            # Fallback to placeholder if the layer doesn't exist
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.random.rand(len(feature_names))
            })
            self.feature_importance = self.feature_importance.sort_values('Importance', ascending=False)
    
    def predict(self, df, return_intervals=True):
        """Make predictions using the trained model with optimized prediction intervals"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Filter out non-numeric columns
        non_numeric_cols = ['Date', 'Start Period', 'End Period', 'Timestamp']
        numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
        
        # Keep only numeric columns
        df_numeric = df[numeric_cols]
        
        # Check if GHI_lag (t-1) exists - if not and we have GHI, create it for backward compatibility
        if 'GHI_lag (t-1)' not in df_numeric.columns and 'GHI - W/m^2' in df_numeric.columns:
            print("Warning: 'GHI_lag (t-1)' column not found, creating it from 'GHI - W/m^2'")
            df_numeric['GHI_lag (t-1)'] = df_numeric['GHI - W/m^2'].shift(1).fillna(0)
        
        # Separate features from target
        target_col = 'GHI - W/m^2'
        feature_cols = [col for col in df_numeric.columns if col != target_col]
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.feature_scaler.transform(df_numeric[feature_cols]),
            columns=feature_cols,
            index=df_numeric.index
        )
        
        # If target column exists, scale it too
        if target_col in df_numeric.columns:
            target_scaled = pd.DataFrame(
                self.target_scaler.transform(df_numeric[[target_col]]),
                columns=[target_col],
                index=df_numeric.index
            )
            # Combine scaled features and target
            df_scaled = pd.concat([features_scaled, target_scaled], axis=1)
        else:
            # For prediction without target, just use features
            df_scaled = features_scaled
            # Add a dummy target column of zeros for compatibility with _prepare_sequences
            df_scaled[target_col] = 0
        
        # Prepare input sequences
        X, _ = self._prepare_sequences(df_scaled, target_col)
        
        # Make predictions (suppress progress bar)
        predictions = self.model.predict(X, verbose=0)
        
        # Process predictions for each horizon
        results = {}
        
        # Required PICP threshold - only calibrate if we're below this value
        picp_threshold = 0.80
        
        # Calculate approximate PICP if we have true values
        hour_picp_estimates = {}
        if target_col in df_numeric.columns:
            # We have true values, so we can calculate PICP by hour
            start_idx = self.input_seq_length
            hour_values = []
            
            # Try to get hour information if available
            if hasattr(df_numeric.index, 'hour'):
                hour_values = df_numeric.index[start_idx:].hour.values
            elif 'Hour of Day' in df_numeric.columns:
                hour_values = df_numeric['Hour of Day'].values[start_idx:]
                
            # Only do this calculation if we have hour values
            if len(hour_values) > 0:
                # Group true values by hour
                for hour in range(6, 17):  
                    hour_indices = [i for i, h in enumerate(hour_values) if h == hour]
                    if hour_indices:
                        hour_picp_estimates[hour] = 0  # Will be updated per horizon
        
        for h in range(self.forecast_horizons):
            # Get predictions for this horizon
            mean_idx = h * 3
            lower_idx = h * 3 + 1
            upper_idx = h * 3 + 2
            
            # Inverse transform to original scale using target scaler
            mean_pred = self.target_scaler.inverse_transform(predictions[mean_idx])
            lower_pred = self.target_scaler.inverse_transform(predictions[lower_idx])
            upper_pred = self.target_scaler.inverse_transform(predictions[upper_idx])
            
            # Apply constraints:
            # 1. Ensure GHI values are non-negative
            mean_pred = np.maximum(0, mean_pred)
            lower_pred = np.maximum(0, lower_pred)
            upper_pred = np.maximum(0, upper_pred)
            
            # 2. Set GHI to 0 during nighttime hours if Daytime feature exists
            if 'Daytime' in df_numeric.columns:
                start_idx = self.input_seq_length
                daytime_values = df_numeric['Daytime'].values
                for i in range(len(mean_pred)):
                    pred_idx = start_idx + i + h
                    if pred_idx < len(daytime_values) and daytime_values[pred_idx] == 0:
                        mean_pred[i] = 0
                        lower_pred[i] = 0
                        upper_pred[i] = 0
            
            # 3. Ensure lower <= mean <= upper
            for i in range(len(mean_pred)):
                if lower_pred[i] > mean_pred[i]:
                    lower_pred[i] = mean_pred[i]
                if upper_pred[i] < mean_pred[i]:
                    upper_pred[i] = mean_pred[i]
                if upper_pred[i] == lower_pred[i]:
                    uncertainty = 0.05 * mean_pred[i] if mean_pred[i] > 0 else 1.0
                    upper_pred[i] = mean_pred[i] + uncertainty
                    lower_pred[i] = max(0, mean_pred[i] - uncertainty)
            
            horizon_name = f'horizon_{h+1}'
            results[f'{horizon_name}_mean'] = mean_pred.flatten()
            
            if return_intervals:
                results[f'{horizon_name}_lower'] = lower_pred.flatten()
                results[f'{horizon_name}_upper'] = upper_pred.flatten()
        
        return pd.DataFrame(results)
    
    def _calculate_cwc(self, picp, piw, alpha=0.05, y_true=None):
        """Calculate Coverage Width-based Criterion (CWC) using PINAW"""
        # alpha is the significance level (now 0.05 for 95% prediction interval)
        # eta (η) is the trade-off parameter balancing coverage and width (now 10)
        eta = 10  # Trade-off parameter as specified
        
        # Calculate nominal coverage (1-alpha) = PINC (Prediction Interval Nominal Coverage)
        pinc = 1.0 - alpha
        
        # Calculate PINAW (Prediction Interval Normalized Average Width) if true values are provided
        if y_true is not None and len(y_true) > 0:
            # Calculate range of true values to normalize the width
            y_range = np.max(y_true) - np.min(y_true)
            if y_range > 0:
                pinaw = piw / y_range
            else:
                # Handle case where range is 0 (all values are the same)
                pinaw = piw  # Use unnormalized PIW in this case
        else:
            # Use unnormalized PIW if true values aren't available
            pinaw = piw
        
        # Calculate φ(PICP) - the indicator function
        # φ(PICP) = 1 if PICP < PINC, otherwise 0
        phi = 1.0 if picp < pinc else 0.0
        
        # Calculate penalty term: φ(PICP)·exp(-η(PICP - PINC))
        penalty_term = phi * np.exp(-eta * (picp - pinc))
        
        # CWC = PINAW[1 + φ(PICP)·exp(-η(PICP - PINC))]
        cwc = pinaw * (1.0 + penalty_term)
        
        return cwc, pinaw  # Added pinaw to return values
    
    def evaluate(self, df, target_col='GHI - W/m^2'):
        """Evaluate the model performance using various metrics for daytime hours (6:00-17:00)
        
        The model uses 95% confidence intervals and calculates CWC using PINAW (Prediction Interval
        Normalized Average Width) to balance coverage probability and interval width. Prediction 
        intervals are optimized to be as narrow as possible while maintaining the desired coverage.
        """
        
        # Get a copy of the dataframe with all columns
        df_full = df.copy()
        
        # Filter out non-numeric columns for model processing
        non_numeric_cols = ['Date', 'Start Period', 'End Period', 'Timestamp']
        numeric_cols = [col for col in df.columns if col not in non_numeric_cols]
        
        # Keep only the target column and numeric columns
        df_numeric = df[numeric_cols]
        
        # Make predictions
        predictions_df = self.predict(df_numeric)
        
        # Prepare actual values for each horizon
        actuals = {}
        for h in range(self.forecast_horizons):
            horizon_name = f'horizon_{h+1}'
            actuals[horizon_name] = df_numeric[target_col][self.input_seq_length+h:self.input_seq_length+h+len(predictions_df)].values
        
        # Extract time periods for filtering and display
        has_start_period = 'Start Period' in df_full.columns
        has_end_period = 'End Period' in df_full.columns
        
        # Prepare metrics storage
        metrics = {}
        hourly_metrics = {}
        hourly_metric_lookup = {}  # Renamed for clarity
        hourly_cwc_metrics = {}  # New dictionary for hourly CWC metrics
        hourly_pinaw_metrics = {}  # New dictionary for hourly PINAW metrics
        overall_mae_by_horizon = {}  # New dictionary for overall MAE by horizon
        
        for h in range(self.forecast_horizons):
            horizon_name = f'horizon_{h+1}'
            
            # Actual values
            y_true = actuals[horizon_name]
            
            # Predicted values
            y_pred_mean = predictions_df[f'{horizon_name}_mean']
            y_pred_lower = predictions_df[f'{horizon_name}_lower']
            y_pred_upper = predictions_df[f'{horizon_name}_upper']
            
            # Initialize hourly metrics for this horizon
            hourly_metrics[horizon_name] = {}
            hourly_metric_lookup[horizon_name] = {}  # Initialize metric lookup
            hourly_cwc_metrics[horizon_name] = {}  # Initialize CWC metrics
            hourly_pinaw_metrics[horizon_name] = {}  # Initialize PINAW metrics
            
            # Get time information if available
            if has_start_period:
                # Get corresponding times for this horizon's predictions
                start_idx = self.input_seq_length + h
                end_idx = start_idx + len(predictions_df)
                start_times = df_full['Start Period'].iloc[start_idx:end_idx]
                
                # Convert time strings to datetime.time objects if needed
                if isinstance(start_times.iloc[0], str):
                    try:
                        # Try to convert to datetime.time objects
                        start_times = pd.to_datetime(start_times, format='%H:%M:%S').dt.time
                    except:
                        # If conversion fails, keep as is
                        print("Warning: Could not convert Start Period to time objects")
            else:
                # Use index hour if timestamp index
                if hasattr(df_full.index, 'hour'):
                    start_times = df_full.index[self.input_seq_length+h:self.input_seq_length+h+len(predictions_df)].hour
                else:
                    # Create a dummy time range if no time information
                    start_times = pd.Series(['N/A'] * len(y_true))
            
            # Get end times if available
            if has_end_period:
                end_times = df_full['End Period'].iloc[start_idx:end_idx]
                if isinstance(end_times.iloc[0], str):
                    try:
                        end_times = pd.to_datetime(end_times, format='%H:%M:%S').dt.time
                    except:
                        print("Warning: Could not convert End Period to time objects")
            else:
                # Create end times as start + 1 hour if we have proper start times
                if has_start_period:
                    # Check if start_times contains datetime.time objects safely
                    if hasattr(start_times.iloc[0], 'hour'):
                        # This means it's likely a time object with hour attribute
                        end_times = [(datetime.datetime.combine(datetime.date.today(), t) + 
                                    datetime.timedelta(hours=1)).time() for t in start_times]
                        end_times = pd.Series(end_times)
                    else:
                        end_times = pd.Series(['N/A'] * len(y_true))
                else:
                    end_times = pd.Series(['N/A'] * len(y_true))
            
            # Filter for daytime hours (6:00-17:59)
            daytime_mask = np.ones(len(y_true), dtype=bool)  # Default all True
            
            if has_start_period:
                # Check if start_times contains time objects with hour attribute
                if hasattr(start_times.iloc[0], 'hour'):
                    # Create mask for hours between 6:00-17:59
                    daytime_mask = [(t.hour >= 6 and t.hour < 18) for t in start_times]
            
            # Apply daytime filter
            y_true_daytime = y_true[daytime_mask]
            y_pred_mean_daytime = y_pred_mean[daytime_mask]
            y_pred_lower_daytime = y_pred_lower[daytime_mask]
            y_pred_upper_daytime = y_pred_upper[daytime_mask]
            
            # Get filtered time periods
            filtered_start_times = start_times[daytime_mask]
            filtered_end_times = end_times[daytime_mask]
            
            # Skip if no daytime data points
            if len(y_true_daytime) == 0:
                metrics[horizon_name] = {
                    'MAE': float('nan'),
                    'PICP': float('nan'),
                    'PIW': float('nan'),
                    'PINAW': float('nan'),  # Added PINAW metric
                    'Winkler_Score': float('nan'),
                    'CWC': float('nan'),
                    'Samples': 0
                }
                overall_mae_by_horizon[f"{h+1} Hour Ahead"] = float('nan')
                continue
            
            # Calculate overall daytime metrics
            # MAE for deterministic forecast
            mae = mean_absolute_error(y_true_daytime, y_pred_mean_daytime)
            overall_mae_by_horizon[f"{h+1} Hour Ahead"] = mae
            
            # PICP - Prediction Interval Coverage Probability
            covered = np.sum((y_true_daytime >= y_pred_lower_daytime) & (y_true_daytime <= y_pred_upper_daytime))
            picp = covered / len(y_true_daytime)
            
            # PIW - Prediction Interval Width
            piw = np.mean(y_pred_upper_daytime - y_pred_lower_daytime)
            
            # Winkler Score - using alpha=0.05 for 95% prediction interval
            alpha = 0.05  # Changed from 0.2 for 95% confidence interval
            winkler_score = self._calculate_winkler_score(
                y_true_daytime, y_pred_lower_daytime, y_pred_upper_daytime, alpha
            )
            
            # Calculate CWC - Coverage Width-based Criterion and PINAW
            cwc, pinaw = self._calculate_cwc(picp, piw, alpha, y_true=y_true_daytime)
            
            # Store overall metrics
            metrics[horizon_name] = {
                'MAE': mae,
                'PICP': picp,
                'PIW': piw,
                'PINAW': pinaw,  # Added PINAW metric
                'Winkler_Score': winkler_score,
                'CWC': cwc,
                'Samples': len(y_true_daytime)
            }
            
            # Calculate hourly metrics if we have time information
            # Check if filtered_start_times has time objects with hour attribute 
            has_time_objects = (len(filtered_start_times) > 0 and hasattr(filtered_start_times.iloc[0], 'hour'))
            
            if has_start_period and has_time_objects:
                # Group by hour
                hour_groups = {}
                
                # Create hour bins (6-18)
                for hour in range(6, 18):
                    hour_str = f'{hour:02d}:00:00'
                    next_hour = f'{(hour+1):02d}:00:00'
                    forecast_interval = f"{hour_str} – {next_hour}"
                    
                    # Filter data for this hour
                    hour_mask = [(t.hour == hour) for t in filtered_start_times]
                    
                    if sum(hour_mask) > 0:  # Only process if we have data for this hour
                        hour_true = y_true_daytime[hour_mask]
                        hour_pred_mean = y_pred_mean_daytime[hour_mask]
                        hour_pred_lower = y_pred_lower_daytime[hour_mask]
                        hour_pred_upper = y_pred_upper_daytime[hour_mask]
                        
                        # Calculate metrics for this hour
                        hour_mae = mean_absolute_error(hour_true, hour_pred_mean)
                        
                        hour_covered = np.sum((hour_true >= hour_pred_lower) & (hour_true <= hour_pred_upper))
                        hour_picp = hour_covered / len(hour_true) if len(hour_true) > 0 else float('nan')
                        
                        hour_piw = np.mean(hour_pred_upper - hour_pred_lower)
                        
                        hour_winkler = self._calculate_winkler_score(
                            hour_true, hour_pred_lower, hour_pred_upper, alpha
                        )
                        
                        # Calculate hourly CWC and PINAW
                        hour_cwc, hour_pinaw = self._calculate_cwc(hour_picp, hour_piw, alpha, y_true=hour_true)
                        
                        # Store hourly metrics
                        hourly_metrics[horizon_name][hour_str] = {
                            'end_period': next_hour,
                            'forecast_interval': forecast_interval,
                            'PICP': hour_picp,
                            'PIW': hour_piw,
                            'PINAW': hour_pinaw,  # Added PINAW
                            'Winkler': hour_winkler,
                            'CWC': hour_cwc,
                            'MAE': hour_mae,
                            'Samples': len(hour_true)
                        }
                        
                        # Store both MAE and CWC for lookup
                        hourly_metric_lookup[horizon_name][forecast_interval] = hour_mae
                        hourly_cwc_metrics[horizon_name][forecast_interval] = hour_cwc
                        hourly_pinaw_metrics[horizon_name][forecast_interval] = hour_pinaw  # Store PINAW
            
            # Print the formatted table for this horizon
            print(f"\n{'='*80}")
            print(f"Forecast Horizon {h+1} Hour Metrics:")
            print(f"{'='*80}")
            
            # Print table header
            header = f"| {'Start Period':<12} | {'End Period':<12} | {'PICP':^8} | {'PIW':^15} | {'PINAW':^8} | {'CWC':^15} | {'MAE':^15} |"
            divider = f"|{'-'*14}|{'-'*14}|{'-'*10}|{'-'*17}|{'-'*10}|{'-'*17}|{'-'*17}|"
            print(divider)
            print(header)
            print(divider)
            
            # Print hourly metrics
            overall_picp = 0
            overall_piw = 0
            overall_pinaw = 0  # Added overall PINAW
            overall_mae = 0
            overall_cwc = 0
            total_samples = 0
            
            if hourly_metrics[horizon_name]:
                for hour_str, hour_data in sorted(hourly_metrics[horizon_name].items()):
                    # Format metrics with proper units
                    picp_str = f"{hour_data['PICP']:.3f}"
                    piw_str = f"{hour_data['PIW']:.2f} W/m²"
                    pinaw_str = f"{hour_data['PINAW']:.3f}"  # Added PINAW formatting
                    mae_str = f"{hour_data['MAE']:.2f} W/m²"
                    cwc_str = f"{hour_data['CWC']:.2f}"
                    
                    # Print row
                    print(f"| {hour_str:<12} | {hour_data['end_period']:<12} | {picp_str:^8} | {piw_str:^15} | {pinaw_str:^8} | {cwc_str:^15} | {mae_str:^15} |")
                    
                    # Accumulate weighted metrics for overall calculation
                    weight = hour_data['Samples']
                    overall_picp += hour_data['PICP'] * weight
                    overall_piw += hour_data['PIW'] * weight
                    overall_pinaw += hour_data['PINAW'] * weight  # Added PINAW accumulation
                    overall_mae += hour_data['MAE'] * weight
                    overall_cwc += hour_data['CWC'] * weight
                    total_samples += weight
            else:
                # If no hourly breakdown, use overall metrics
                overall_picp = metrics[horizon_name]['PICP'] * metrics[horizon_name]['Samples']
                overall_piw = metrics[horizon_name]['PIW'] * metrics[horizon_name]['Samples']
                overall_pinaw = metrics[horizon_name]['PINAW'] * metrics[horizon_name]['Samples']  # Added PINAW
                overall_mae = metrics[horizon_name]['MAE'] * metrics[horizon_name]['Samples']
                overall_cwc = metrics[horizon_name]['CWC'] * metrics[horizon_name]['Samples']
                total_samples = metrics[horizon_name]['Samples']
                
                # Print "No hourly breakdown available"
                print(f"| {'No hourly breakdown available':<70} |")
            
            # Calculate and print overall metrics
            print(divider)
            if total_samples > 0:
                avg_picp = overall_picp / total_samples
                avg_piw = overall_piw / total_samples
                avg_pinaw = overall_pinaw / total_samples  # Calculate average PINAW
                avg_mae = overall_mae / total_samples
                avg_cwc = overall_cwc / total_samples
                
                picp_str = f"{avg_picp:.3f}"
                piw_str = f"{avg_piw:.2f} W/m²"
                pinaw_str = f"{avg_pinaw:.3f}"  # Format PINAW
                mae_str = f"{avg_mae:.2f} W/m²"
                cwc_str = f"{avg_cwc:.2f}"
                
                print(f"| {'Overall':<12} | {'':<12} | {picp_str:^8} | {piw_str:^15} | {pinaw_str:^8} | {cwc_str:^15} | {mae_str:^15} |")
            else:
                print(f"| {'Overall':<12} | {'':<12} | {'N/A':^8} | {'N/A':^15} | {'N/A':^8} | {'N/A':^15} | {'N/A':^15} |")
            
            print(divider)
            print(f"Total samples: {total_samples}\n")
        
        # Print overall MAE by forecast horizon
        print(f"\n{'='*70}")
        print(f"Overall MAE by Forecast Horizon:")
        print(f"{'='*70}")
        for horizon, mae in overall_mae_by_horizon.items():
            print(f"{horizon}: {mae:.2f} W/m²")
        
        # Return metrics and hourly breakdown for exporting
        return metrics, hourly_metrics, hourly_metric_lookup, hourly_cwc_metrics, hourly_pinaw_metrics, overall_mae_by_horizon
    
    def _calculate_winkler_score(self, y_true, lower, upper, alpha):
        """Calculate Winkler score for prediction intervals"""
        # Convert to numpy arrays if they are pandas Series
        y_true_arr = np.array(y_true)
        lower_arr = np.array(lower)
        upper_arr = np.array(upper)
        
        n = len(y_true_arr)
        width = upper_arr - lower_arr
        penalty = 0
        
        # Vectorized calculation for better performance and to avoid indexing issues
        underprediction = np.maximum(0, lower_arr - y_true_arr)
        overprediction = np.maximum(0, y_true_arr - upper_arr)
        
        penalty = (2/alpha) * (np.sum(underprediction) + np.sum(overprediction))
        
        return np.mean(width) + penalty / n
    
    def plot_forecasts(self, df, target_col='GHI - W/m^2', start_idx=None, end_idx=None, title='TFT Forecasts'):
        """This method is disabled to remove unnecessary visualizations"""
        print("Forecast visualization is disabled. Only Feature Importance charts will be generated.")
        return None
    
    def plot_feature_importance(self):
        """Plot feature importance from the model"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train the model first.")
        
        plt.figure(figsize=(12, 8))
        importance_df = self.feature_importance.sort_values('Importance')
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title('TFT Feature Importance')
        plt.tight_layout()
        plt.savefig('tft_feature_importance.png', dpi=300)
        plt.show()

    def export_metrics_to_csv(self, metrics, hourly_metrics, hourly_metric_lookup, hourly_cwc_metrics, hourly_pinaw_metrics, overall_mae_by_horizon, model_name, output_dir):
        """Export metrics to CSV files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed metrics by horizon to CSV
        detailed_csv_path = os.path.join(output_dir, f'Detailed_Metrics_{model_name}_{timestamp}.csv')
        
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Metric Type', 'Forecast Horizon', 'Metric', 'Value', 'Units'])
            
            # Write overall MAE by forecast horizon
            for horizon, mae in overall_mae_by_horizon.items():
                writer.writerow(['Overall', horizon, 'MAE', f"{mae:.4f}", 'W/m²'])
            
            # Write all metrics by horizon
            for h in range(self.forecast_horizons):
                horizon_name = f'horizon_{h+1}'
                horizon_display = f'{h+1} Hour Ahead'
                
                if horizon_name in metrics:
                    for metric_name, value in metrics[horizon_name].items():
                        if metric_name != 'Samples':
                            # Add proper units
                            units = 'W/m²' if metric_name in ['MAE', 'PIW', 'Winkler_Score'] else ''
                            writer.writerow(['Overall', horizon_display, metric_name, f"{value:.4f}", units])
                    writer.writerow(['Overall', horizon_display, 'Samples', metrics[horizon_name]['Samples'], ''])
                
                # Write hourly metrics for each horizon
                for hour_str, hour_data in sorted(hourly_metrics[horizon_name].items()):
                    forecast_interval = hour_data['forecast_interval']
                    
                    for metric_name, value in hour_data.items():
                        # Skip non-metric fields
                        if metric_name not in ['end_period', 'forecast_interval']:
                            # Add proper units
                            units = 'W/m²' if metric_name in ['MAE', 'PIW', 'Winkler'] else ''
                            
                            if metric_name != 'Samples':
                                writer.writerow(['Hourly', horizon_display, f"{metric_name} ({forecast_interval})", f"{value:.4f}", units])
                            else:
                                writer.writerow(['Hourly', horizon_display, f"Samples ({forecast_interval})", value, ''])
        
        print(f"Detailed metrics saved to: {detailed_csv_path}")
        
        # Create forecast table in the format shown in the image but as CSV
        forecast_csv_path = os.path.join(output_dir, f'Forecast_Table_{model_name}_{timestamp}.csv')
        
        with open(forecast_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Model Name', model_name])
            writer.writerow(['Generated', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])  # Empty row for spacing
            
            # Create headers for the forecast table
            header_row = ['Forecasted Hour', 'Forecast Horizon', 'PICP', 'PIW (W/m²)', 'PINAW', 'CWC', 'MAE (W/m²)']
            writer.writerow(header_row)
            
            # Create the standard time periods for the table (6 AM to 6 PM)
            time_periods = []
            
            for hour in range(6, 18):
                if hour < 12:
                    start = f'{hour}:00 AM'
                    end = f'{hour+1}:00 AM' if hour < 11 else '12:00 PM'
                else:
                    start_h = hour if hour == 12 else hour - 12
                    end_h = hour + 1 if hour != 12 else 1
                    start = f'{start_h}:00 PM'
                    end = f'{end_h}:00 PM' if hour < 23 else '12:00 AM'
                    
                display_str = f'{start} – {end}'
                time_periods.append((hour, display_str))
            
            # Write data for each time period
            for hour, period in time_periods:
                hour_key = f'{hour:02d}:00:00'
                
                # For each forecast horizon
                for h in range(self.forecast_horizons):
                    horizon_name = f'horizon_{h+1}'
                    horizon_display = f'{h+1} Hour Ahead'
                    
                    # Check if metrics exist for this hour/horizon
                    if horizon_name in hourly_metrics and hour_key in hourly_metrics[horizon_name]:
                        hour_data = hourly_metrics[horizon_name][hour_key]
                        
                        row = [
                            period,
                            horizon_display,
                            f"{hour_data['PICP']:.3f}",
                            f"{hour_data['PIW']:.2f}",
                            f"{hour_data['PINAW']:.3f}",
                            f"{hour_data['CWC']:.2f}",
                            f"{hour_data['MAE']:.2f}"
                        ]
                        writer.writerow(row)
                    else:
                        # No data
                        row = [period, horizon_display, "-", "-", "-", "-", "-"]
                        writer.writerow(row)
                
                # Add empty row for spacing between hours
                writer.writerow([])
            
            # Add overall section
            writer.writerow(['Overall', 'Forecast Horizon', 'PICP', 'PIW (W/m²)', 'PINAW', 'CWC', 'MAE (W/m²)'])
            
            # Write overall metrics for each horizon
            for h in range(self.forecast_horizons):
                horizon_name = f'horizon_{h+1}'
                horizon_display = f'{h+1} Hour Ahead'
                
                if horizon_name in metrics:
                    row = [
                        'All hours',
                        horizon_display,
                        f"{metrics[horizon_name]['PICP']:.3f}",
                        f"{metrics[horizon_name]['PIW']:.2f}",
                        f"{metrics[horizon_name]['PINAW']:.3f}",
                        f"{metrics[horizon_name]['CWC']:.2f}",
                        f"{metrics[horizon_name]['MAE']:.2f}"
                    ]
                    writer.writerow(row)
                else:
                    row = ['All hours', horizon_display, "-", "-", "-", "-", "-"]
                    writer.writerow(row)
        
        print(f"Forecast metrics table saved to: {forecast_csv_path}")
        
        # Save a copy in the current directory as requested
        current_dir_path = os.path.join(os.getcwd(), f'Metrics_{model_name}_{timestamp}.csv')
        
        # Create a combined metrics file with the most important information
        with open(current_dir_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with model info
            writer.writerow(['DavCast LSTM Forecast Metrics'])
            writer.writerow(['Model Name', model_name])
            writer.writerow(['Generated', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])
            
            # Overall performance by horizon
            writer.writerow(['Overall Performance by Forecast Horizon'])
            writer.writerow(['Forecast Horizon', 'MAE (W/m²)', 'PICP', 'PIW (W/m²)', 'PINAW', 'CWC'])
            
            for h in range(self.forecast_horizons):
                horizon_name = f'horizon_{h+1}'
                horizon_display = f'{h+1} Hour Ahead'
                
                if horizon_name in metrics:
                    row = [
                        horizon_display,
                        f"{metrics[horizon_name]['MAE']:.2f}",
                        f"{metrics[horizon_name]['PICP']:.3f}",
                        f"{metrics[horizon_name]['PIW']:.2f}",
                        f"{metrics[horizon_name]['PINAW']:.3f}",
                        f"{metrics[horizon_name]['CWC']:.2f}"
                    ]
                    writer.writerow(row)
            
            writer.writerow([])
            
            # Hourly breakdown for 1-hour ahead forecast (the most important)
            writer.writerow(['Hourly Breakdown for 1-Hour Ahead Forecast'])
            writer.writerow(['Hour', 'MAE (W/m²)', 'PICP', 'PIW (W/m²)', 'PINAW', 'CWC', 'Samples'])
            
            horizon_name = 'horizon_1'
            if horizon_name in hourly_metrics:
                for hour_str, hour_data in sorted(hourly_metrics[horizon_name].items()):
                    writer.writerow([
                        hour_data['forecast_interval'],
                        f"{hour_data['MAE']:.2f}",
                        f"{hour_data['PICP']:.3f}",
                        f"{hour_data['PIW']:.2f}",
                        f"{hour_data['PINAW']:.3f}",
                        f"{hour_data['CWC']:.2f}",
                        hour_data['Samples']
                    ])
        
        print(f"Combined metrics saved to: {current_dir_path}")
        
        return detailed_csv_path, forecast_csv_path, current_dir_path


# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("DavCast TFT Solar Forecasting")
    print("="*50)
    
    # Function to load data from file dialog
    def load_data():
        """Load data using file dialog"""
        root = Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        
        print("\nSelect input data file...")
        file_path = filedialog.askopenfilename(
            title="Select Input Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:  # User cancelled
            print("No input file selected. Operation cancelled.")
            return None
            
        print(f"\nInput file: {file_path}")
        
        # List of encodings to try
        encodings = ['utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                data = pd.read_csv(file_path, encoding=encoding)
                
                if data.empty:
                    raise ValueError("Loaded data is empty. Please check the input file.")

                # If we get here, the file was successfully read
                print(f"Successfully loaded data using {encoding} encoding")
                
                # Convert date columns to timestamp if they exist
                print("Processing date columns...")
                try:
                    # Convert Date and Start Period to datetime
                    if 'Date' in data.columns and 'Start Period' in data.columns:
                        data['Timestamp'] = pd.to_datetime(data['Date'] + ' ' + data['Start Period'])
                        
                        # Sort by timestamp to ensure data is in chronological order
                        data = data.sort_values('Timestamp')
                        
                        # Check for missing timestamps
                        time_diff = data['Timestamp'].diff()
                        if time_diff.dt.total_seconds().max() > 3600:  # More than 1 hour gap
                            print("Warning: Data contains gaps larger than 1 hour")
                            
                        # Check for duplicate timestamps
                        if data['Timestamp'].duplicated().any():
                            print("Warning: Data contains duplicate timestamps")
                    else:
                        print("Warning: Expected 'Date' and 'Start Period' columns not found")
                        
                except Exception as e:
                    print(f"Error processing date columns: {e}")
                    return None
                
                # Create timestamp index if not present
                if 'Timestamp' in data.columns:
                    # Keep Timestamp as a column but also set as index for easier modeling
                    data.set_index('Timestamp', inplace=True, drop=False)
                
                return data
                
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding, trying next encoding...")
                continue
            except Exception as e:
                print(f"Error loading data: {e}")
                return None
        
        print("Failed to read file with any of the attempted encodings")
        return None
    
    # Get training hyperparameters
    def get_hyperparameters():
        """Get model hyperparameters from user input or use defaults or Optuna optimization"""
        print("\nHyperparameter Selection:")
        print("------------------------")
        
        options = {
            '1': 'Use default hyperparameters',
            '2': 'Manual configuration',
            '3': 'Optuna optimization (20 trials)'
        }
        
        for key, value in options.items():
            print(f"{key}. {value}")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print("Using default hyperparameters...")
            return {
                'input_seq_length': 16,
                'forecast_horizons': 4,
                'hidden_size': 96,
                'attention_heads': 3,
                'dropout_rate': 0.15, 
                'batch_size': 16,
                'learning_rate': 0.002566076505216372,
                'max_epochs': 50,
                'validation_split': 0.2
            }
        elif choice == '2':
            print("\nManual hyperparameter configuration:")
            params = {}
            
            try:
                params['input_seq_length'] = int(input("Input sequence length (default=16): ") or 16)
                params['forecast_horizons'] = int(input("Forecast horizons (default=4): ") or 4)
                params['hidden_size'] = int(input("Hidden size (default=96): ") or 96)
                params['attention_heads'] = int(input("Attention heads (default=3): ") or 3)
                params['dropout_rate'] = float(input("Dropout rate (default=0.15): ") or 0.15)
                params['batch_size'] = int(input("Batch size (default=16): ") or 16)
                params['learning_rate'] = float(input("Learning rate (default=0.0025661): ") or 0.002566076505216372)
                params['max_epochs'] = int(input("Maximum epochs (default=50): ") or 50)
                params['validation_split'] = float(input("Validation split (default=0.2): ") or 0.2)
            except ValueError as e:
                print(f"Error in parameter input: {e}. Using defaults.")
                return get_hyperparameters()
                
            return params
        elif choice == '3':
            print("\nStep 2: Preparing for Optuna optimization...")
            # Run Optuna with default validation split
            validation_split = 0.2
            return run_optuna_optimization(data, target_col, validation_split)
        else:
            print("Invalid option. Using default hyperparameters.")
            return get_hyperparameters()
            
    # Define Optuna optimization
    def run_optuna_optimization(data, target_col, validation_split=0.2):
        """Run Optuna hyperparameter optimization"""
        
        # Filter out non-numeric columns based on the dataset structure
        non_numeric_cols = ['Date', 'Start Period', 'End Period', 'Timestamp']
        
        # First, select only the precise features we want to use
        exact_features = [f for f in required_features if f in data.columns]
        
        # Create a subset of data with exactly these columns
        data_subset = data[[target_col] + exact_features].copy()
        
        print(f"\nUsing exactly these {len(exact_features)} features for all trials:")
        for f in exact_features:
            print(f"  - {f}")
        
        # Verify target column exists
        if target_col not in data_subset.columns:
            print(f"Error: Target column '{target_col}' not found in dataset.")
            return None
        
        # Split data for training and validation
        train_size = int(len(data_subset) * (1 - validation_split))
        train_data = data_subset.iloc[:train_size]
        val_data = data_subset.iloc[train_size:]
        
        print(f"Training data: {len(train_data)} samples | Validation data: {len(val_data)} samples")
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to optimize - including input_seq_length
            params = {
                'input_seq_length': trial.suggest_int('input_seq_length', 4, 24, step=4),  # Dynamic sequence length
                'forecast_horizons': 4,  # Fixed for this implementation
                'hidden_size': trial.suggest_int('hidden_size', 32, 192, step=16),  # Wider range
                'attention_heads': trial.suggest_int('attention_heads', 2, 8, step=1),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3, step=0.05),
                'batch_size': trial.suggest_int('batch_size', 16, 64, step=8),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'max_epochs': 30,  # Using 30 epochs for optimization to save time
                'validation_split': validation_split
            }
            
            print(f"\nTrial {trial.number} parameters:")
            for param, value in params.items():
                print(f"  {param}: {value}")
            
            print(f"Model input shape will be: (batch_size, {params['input_seq_length']}, {len(exact_features)})")
            
            # Initialize model with current hyperparameters
            tft_trial = TemporalFusionTransformer(
                input_seq_length=params['input_seq_length'],
                forecast_horizons=params['forecast_horizons'],
                hidden_size=params['hidden_size'],
                attention_heads=params['attention_heads'],
                dropout_rate=params['dropout_rate'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                max_epochs=params['max_epochs']
            )
            
            # Train model with early stopping
            try:
                # Train using exactly the same subset of data for all trials
                # Set verbose=0 to suppress output during optimization
                history = tft_trial.fit(data_subset, target_col=target_col, validation_split=params['validation_split'], verbose=0)
                
                # Use the best validation loss as the optimization target
                best_val_loss = min(history.history['val_loss'])
                print(f"Validation loss: {best_val_loss:.4f}")
                
                return best_val_loss
                
            except Exception as e:
                print(f"Trial failed: {e}")
                import traceback
                traceback.print_exc()
                # Return a high loss value to discourage this parameter combination
                return float('inf')
        
        # Create a study and optimize
        print("\nStarting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        
        # Add fixed parameters
        best_params['forecast_horizons'] = 4
        best_params['max_epochs'] = 50  # Restore full epochs (limited to 50) for final training
        best_params['validation_split'] = validation_split
        
        print("\nBest trial:")
        print(f"Value: {study.best_trial.value:.4f}")
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    data = load_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        exit()
    
    print(f"Loaded data with shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Step 2: Verify and select target column
    print("\nStep 2: Verifying data...")
    
    # Look for GHI column with different possible names
    ghi_column_names = ['GHI - W/m^2', 'GHI - W/m²', 'GHI']
    found_ghi = None
    
    for col in ghi_column_names:
        if col in data.columns:
            found_ghi = col
            break
    
    if found_ghi is None:
        print("Warning: No standard GHI column found.")
        print("Available columns:")
        for i, col in enumerate(data.columns):
            print(f"{i}: {col}")
        
        col_idx = int(input("\nSelect GHI column by index: "))
        found_ghi = data.columns[col_idx]
    
    target_col = found_ghi
    print(f"Using {target_col} as target column")
    
    # Step 3: Check for required features
    required_features = [
        'Barometer - hPa',
        'Temp - °C',
        'Hum - %',
        'Dew Point - °C',
        'Wet Bulb - °C',
        'Avg Wind Speed - km/h',
        'Rain - mm',
        'High Rain Rate - mm/h',
        'UV Index',
        'Wind Run - km',
        'Month of Year',
        'Hour of Day',
        'Solar Zenith Angle',
        'GHI_lag (t-1)',
        'Daytime'
    ]
    
    # Check which features are already in the dataset
    existing_features = [f for f in required_features if f in data.columns]
    missing_features = [f for f in required_features if f not in data.columns]
    
    # Report on existing features
    print("\nRequired features status:")
    for feature in existing_features:
        print(f"  ✓ {feature} (found in dataset)")
    
    # Only create missing features
    if missing_features:
        print("\nMissing features to be created:")
        for feature in missing_features:
            if feature == 'Hour of Day':
                print(f"  + Creating {feature}...")
                if 'Timestamp' in data.columns:
                    data[feature] = data['Timestamp'].dt.hour
                else:
                    data[feature] = data.index.hour
                    
            elif feature == 'Daytime':
                print(f"  + Creating {feature}...")
                if 'Solar Zenith Angle' in data.columns:
                    data[feature] = (data['Solar Zenith Angle'] < 90).astype(int)
                else:
                    # Use hours between 6-18 as proxy for daytime
                    hour_col = 'Hour of Day' if 'Hour of Day' in data.columns else data.index.hour
                    data[feature] = ((hour_col >= 6) & (hour_col <= 18)).astype(int)
                
            elif feature == 'Month of Year':
                print(f"  + Creating {feature}...")
                if 'Timestamp' in data.columns:
                    data[feature] = data['Timestamp'].dt.month
                else:
                    data[feature] = data.index.month
                
            elif feature == 'GHI_lag (t-1)':
                print(f"  + Creating {feature}...")
                if 'GHI - W/m^2' in data.columns:
                    data[feature] = data['GHI - W/m^2'].shift(1).fillna(0)
                else:
                    print(f"  ! Cannot create {feature} as 'GHI - W/m^2' is not present")
                
            else:
                print(f"  ! Warning: Required feature '{feature}' is missing and cannot be created automatically")
    else:
        print("\nAll required features are present in the dataset")
    
    # Only use columns that exist in the data
    available_features = [f for f in required_features if f in data.columns]
    
    # Step 4: Get hyperparameters
    params = get_hyperparameters()
    
    # Step 5: Initialize and train the model
    print("\nStep 5: Initializing model...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "tft_models", f"TFT_Model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model outputs will be saved to: {output_dir}")
    
    tft = TemporalFusionTransformer(
        input_seq_length=params['input_seq_length'],
        forecast_horizons=params['forecast_horizons'],
        hidden_size=params['hidden_size'],
        attention_heads=params['attention_heads'],
        dropout_rate=params['dropout_rate'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        max_epochs=params['max_epochs']
    )
    
    # Step 6: Train the model
    print("\nStep 6: Training the model...")
    try:
        # Use exactly the same feature subset approach as in Optuna
        target_col = found_ghi
        
        # Select exact features from the required list
        exact_features = [f for f in required_features if f in data.columns]
        
        # Create data subset with exactly these columns
        model_data = data[[target_col] + exact_features].copy()
        
        print(f"\nUsing exactly these {len(exact_features)} features for training:")
        for f in exact_features:
            print(f"  - {f}")
        
        print(f"Model input shape: (batch_size, {params['input_seq_length']}, {len(exact_features)})")
        
        # Define model checkpoint path
        checkpoint_path = os.path.join(output_dir, 'tft_model.keras')

        # Updated callbacks with correct path
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train the model with the selected features
        history = tft.fit(model_data, target_col=target_col, validation_split=params['validation_split'], callbacks=callbacks)
        
        # Save training history plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('TFT Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(history_plot_path, dpi=300)
        
        # Step 7: Evaluate the model
        print("\nStep 7: Evaluating model...")
        metrics, hourly_metrics, hourly_metric_lookup, hourly_cwc_metrics, hourly_pinaw_metrics, overall_mae_by_horizon = tft.evaluate(data, target_col=target_col)
        
        # Write metrics to file
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("TFT Model Evaluation Metrics (Daytime Hours 6:00-17:00 Only):\n")
            f.write("=" * 50 + "\n\n")
            
            for horizon, horizon_metrics in metrics.items():
                f.write(f"\n{horizon}:\n")
                f.write(f"  Samples: {horizon_metrics.get('Samples', 0)} daytime hours\n")
                for metric_name, value in horizon_metrics.items():
                    if metric_name != 'Samples':  # Skip samples in this loop
                        f.write(f"  {metric_name}: {value:.4f}\n")
        
        print("Model Evaluation Metrics (Daytime Hours 6:00-17:00 Only):")
        print("=" * 50)
        for horizon, horizon_metrics in metrics.items():
            print(f"\n{horizon}:")
            print(f"  Samples: {horizon_metrics.get('Samples', 0)} daytime hours")
            for metric_name, value in horizon_metrics.items():
                if metric_name != 'Samples':  # Skip samples in this loop
                    print(f"  {metric_name}: {value:.4f}")
                
        # Step 8: Exporting validation set predictions to CSV...
        print("\nStep 8: Exporting validation set predictions to CSV...")
        # Get validation data indices
        val_size = int(len(data) * params['validation_split'])
        validation_data = data.iloc[-val_size:]
        
        # Generate predictions
        predictions_df = tft.predict(validation_data)
        
        # Create a DataFrame for the validation predictions
        validation_export = pd.DataFrame()
        
        # Calculate offset for proper alignment
        offset = tft.input_seq_length
        
        # Create date and time columns
        timestamp_data = validation_data.iloc[offset:offset+len(predictions_df)]
        
        # Extract date components
        if 'Date' in timestamp_data.columns:
            validation_export['Date'] = timestamp_data['Date'].values
        else:
            validation_export['Date'] = timestamp_data.index.strftime('%d/%m/%Y')
            
        # Extract time components
        if 'Start Period' in timestamp_data.columns:
            validation_export['Start Period'] = timestamp_data['Start Period'].values
            # Calculate End Period (1 hour after Start Period)
            validation_export['End Period'] = pd.to_datetime(timestamp_data['Start Period']).apply(
                lambda x: (x + datetime.timedelta(hours=1)).strftime('%H:%M:%S')
            ).values
        else:
            validation_export['Start Period'] = timestamp_data.index.strftime('%H:%M:%S')
            validation_export['End Period'] = (timestamp_data.index + datetime.timedelta(hours=1)).strftime('%H:%M:%S')
        
        # Get actual values and add prediction columns for each horizon
        for h in range(tft.forecast_horizons):
            horizon_name = f'horizon_{h+1}'
            horizon_display = f't+{h+1}'
            
            # Calculate actuals with proper alignment
            actuals = validation_data[target_col][offset+h:offset+h+len(predictions_df)].values
            
            # Add columns for actual, predicted, lower bound, upper bound, and error
            validation_export[f'Actual_{horizon_display} (W/m²)'] = actuals
            validation_export[f'Predicted_{horizon_display} (W/m²)'] = predictions_df[f'{horizon_name}_mean'].values
            validation_export[f'Lower_Bound_{horizon_display} (W/m²)'] = predictions_df[f'{horizon_name}_lower'].values
            validation_export[f'Upper_Bound_{horizon_display} (W/m²)'] = predictions_df[f'{horizon_name}_upper'].values
            validation_export[f'Error_{horizon_display} (W/m²)'] = validation_export[f'Predicted_{horizon_display} (W/m²)'] - validation_export[f'Actual_{horizon_display} (W/m²)']
        
        # Rename columns to match the format in the image
        validation_export.rename(columns={
            'Actual_t+1 (W/m²)': 'Actual_t+1 (W/m²)',
            'Predicted_t+1 (W/m²)': 'Predicted_t+1 (W/m²)',
            'Lower_Bound_t+1 (W/m²)': 'Lower_Bound_t+1 (W/m²)',
            'Upper_Bound_t+1 (W/m²)': 'Upper_Bound_t+1 (W/m²)',
            'Error_t+1 (W/m²)': 'Error_t+1 (W/m²)',
            'Actual_t+2 (W/m²)': 'Actual_t+2 (W/m²)',
            'Predicted_t+2 (W/m²)': 'Predicted_t+2 (W/m²)',
            'Lower_Bound_t+2 (W/m²)': 'Lower_Bound_t+2 (W/m²)',
            'Upper_Bound_t+2 (W/m²)': 'Upper_Bound_t+2 (W/m²)',
            'Error_t+2 (W/m²)': 'Error_t+2 (W/m²)',
            'Actual_t+3 (W/m²)': 'Actual_t+3 (W/m²)',
            'Predicted_t+3 (W/m²)': 'Predicted_t+3 (W/m²)', 
            'Lower_Bound_t+3 (W/m²)': 'Lower_Bound_t+3 (W/m²)',
            'Upper_Bound_t+3 (W/m²)': 'Upper_Bound_t+3 (W/m²)',
            'Error_t+3 (W/m²)': 'Error_t+3 (W/m²)',
            'Actual_t+4 (W/m²)': 'Actual_t+4 (W/m²)',
            'Predicted_t+4 (W/m²)': 'Predicted_t+4 (W/m²)',
            'Lower_Bound_t+4 (W/m²)': 'Lower_Bound_t+4 (W/m²)',
            'Upper_Bound_t+4 (W/m²)': 'Upper_Bound_t+4 (W/m²)',
            'Error_t+4 (W/m²)': 'Error_t+4 (W/m²)'
        }, inplace=True)
        
        # Save to CSV
        validation_export_path = os.path.join(output_dir, f'Validation_Set_Predictions_{timestamp}.csv')
        validation_export.to_csv(validation_export_path, index=False)
        print(f"Validation predictions saved to: {validation_export_path}")
        
        # Also save a copy to the current working directory
        current_dir_path = os.path.join(os.getcwd(), f'Validation_Set_Predictions_{timestamp}.csv')
        validation_export.to_csv(current_dir_path, index=False)
        print(f"Validation predictions also saved to: {current_dir_path}")
        
        # Save model and scalers
        model_path = os.path.join(output_dir, 'tft_model.keras')
        tft.model.save(model_path)
        joblib.dump(tft.feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
        joblib.dump(tft.target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        
        # Plot feature importance
        tft.plot_feature_importance()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
        
        # Step 10: Create configuration file
        print("\nStep 9: Creating configuration files...")
        config_path = os.path.join(output_dir, 'model_config.txt')
        with open(config_path, 'w') as f:
            f.write("TFT Model Configuration\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Training timestamp: {timestamp}\n")
            f.write(f"Data shape: {data.shape}\n")
            f.write(f"Date range: {data.index.min()} to {data.index.max()}\n\n")
            f.write("Hyperparameters:\n")
            
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
                
            f.write("\nFeatures used:\n")
            for feature in available_features:
                f.write(f"  - {feature}\n")
            
            f.write(f"\nTarget column: {target_col}\n")
            
        # Create CSV with hyperparameters, metrics, and feature weights
        csv_config_path = os.path.join(output_dir, 'model_performance.csv')
        
        # Extract feature weights from the model if available
        feature_weights = {}
        if tft.feature_weights is not None:
            feature_weights = {feature: weight for feature, weight in zip(available_features, tft.feature_weights)}
                
        # Write to CSV
        with open(csv_config_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Category', 'Name', 'Value'])
            
            # Write hyperparameters
            for param, value in params.items():
                writer.writerow(['Hyperparameter', param, value])
                
            # Write metrics for each horizon
            for horizon, horizon_metrics in metrics.items():
                for metric_name, value in horizon_metrics.items():
                    writer.writerow(['Metric', f'{horizon}_{metric_name}', f'{value:.4f}'])
                    
            # Write feature weights
            for feature, weight in feature_weights.items():
                writer.writerow(['Feature Weight', feature, f'{weight:.4f}'])
        
        print(f"\nModel training complete. All outputs saved to: {output_dir}")
        print(f"Hyperparameters, metrics, and feature weights saved to: {csv_config_path}")
        print("\nTFT Model training and validation completed successfully!")
        
        # Export metrics to CSV files
        print("\nStep 10: Exporting metrics to CSV files...")
        model_display_name = "DavCast LSTM"  # Use a more readable name for the CSV files
        detailed_csv_path, forecast_csv_path, current_dir_path = tft.export_metrics_to_csv(metrics, hourly_metrics, hourly_metric_lookup, hourly_cwc_metrics, hourly_pinaw_metrics, overall_mae_by_horizon, model_display_name, output_dir)
        print(f"Detailed metrics saved to: {detailed_csv_path}")
        print(f"Forecast table saved to: {forecast_csv_path}")
        print(f"Combined metrics saved to: {current_dir_path}")
        print("Note: CSV files contain all the metrics including PINAW")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()