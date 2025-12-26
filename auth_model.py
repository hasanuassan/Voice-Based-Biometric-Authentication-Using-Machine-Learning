"""
Voice Authentication ML Models
Implements SVM, CNN, and LSTM models for voice authentication
"""

import numpy as np
import json
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import *

class VoiceAuthenticationModel:
    """Main authentication model class supporting multiple ML algorithms"""
    
    def __init__(self, model_type='svm'):
        """
        Initialize authentication model
        model_type: 'svm', 'cnn', or 'lstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = os.path.join(MODELS_DIR, f'auth_model_{model_type}.pkl')
        self.scaler_path = os.path.join(MODELS_DIR, f'scaler_{model_type}.pkl')
    
    def create_svm_model(self):
        """Create Support Vector Machine model"""
        self.model = SVC(
            kernel='rbf',
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True  # Enable probability estimates for confidence scores
        )
    
    def create_cnn_model(self, input_shape):
        """Create Convolutional Neural Network model for voice authentication"""
        model = keras.Sequential([
            # Reshape input for CNN (treat features as 1D signal)
            layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            
            # First convolutional block
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Second convolutional block
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer (binary classification: match or no match)
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model for sequential voice feature analysis"""
        model = keras.Sequential([
            # Reshape for LSTM (sequence, features)
            layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            
            # First LSTM layer
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_svm(self, X_train, y_train):
        """Train SVM model"""
        if self.model is None:
            self.create_svm_model()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Save model
        self.save_model()
    
    def train_cnn(self, X_train, y_train, X_val=None, y_val=None):
        """Train CNN model"""
        if self.model is None:
            self.create_cnn_model((X_train.shape[1],))
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Reshape for CNN
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        if validation_data:
            X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
            validation_data = (X_val_reshaped, y_val)
        
        # Train model
        history = self.model.fit(
            X_train_reshaped, y_train,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            validation_data=validation_data,
            verbose=1
        )
        
        self.is_trained = True
        self.save_model()
        return history
    
    def train_lstm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LSTM model"""
        if self.model is None:
            self.create_lstm_model((X_train.shape[1],))
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Reshape for LSTM
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        if validation_data:
            X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
            validation_data = (X_val_reshaped, y_val)
        
        # Train model
        history = self.model.fit(
            X_train_reshaped, y_train,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_data=validation_data,
            verbose=1
        )
        
        self.is_trained = True
        self.save_model()
        return history
    
    def predict(self, feature_vector):
        """
        Predict authentication result
        Returns: (is_verified: bool, confidence_score: float)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to numpy array if needed
        if isinstance(feature_vector, list):
            feature_vector = np.array(feature_vector)
        
        # Ensure correct shape
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)
        
        if self.model_type == 'svm':
            # SVM prediction
            confidence = self.model.predict_proba(feature_scaled)[0][1]  # Probability of match
            is_verified = confidence >= AUTHENTICATION_THRESHOLD
        elif self.model_type in ['cnn', 'lstm']:
            # Deep learning prediction
            feature_reshaped = feature_scaled.reshape(feature_scaled.shape[0], feature_scaled.shape[1], 1)
            confidence = float(self.model.predict(feature_reshaped, verbose=0)[0][0])
            is_verified = confidence >= AUTHENTICATION_THRESHOLD
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return is_verified, float(confidence)
    
    def calculate_similarity(self, feature_vector1, feature_vector2):
        """
        Calculate cosine similarity between two feature vectors
        Used for voice aging adaptation
        """
        vec1 = np.array(feature_vector1)
        vec2 = np.array(feature_vector2)
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def save_model(self):
        """Save trained model to disk"""
        if self.model_type == 'svm':
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            # Save TensorFlow models
            self.model.save(self.model_path.replace('.pkl', '.h5'))
            joblib.dump(self.scaler, self.scaler_path)
    
    def load_model(self):
        """Load trained model from disk"""
        if not os.path.exists(self.model_path):
            return False
        
        if self.model_type == 'svm':
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            # Load TensorFlow models
            model_path_h5 = self.model_path.replace('.pkl', '.h5')
            if os.path.exists(model_path_h5):
                self.model = keras.models.load_model(model_path_h5)
                self.scaler = joblib.load(self.scaler_path)
            else:
                return False
        
        self.is_trained = True
        return True

