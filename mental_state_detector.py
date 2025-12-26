"""
Mental State Detection Module
Analyzes voice features to classify mental state: Calm, Stressed, Anxious, Fatigued
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import *

class MentalStateDetector:
    """Detect mental state from voice features"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = os.path.join(MODELS_DIR, 'mental_state_model.pkl')
        self.scaler_path = os.path.join(MODELS_DIR, 'mental_state_scaler.pkl')
    
    def extract_mental_state_features(self, voice_features):
        """
        Extract features specifically relevant for mental state detection
        Based on: pitch variation, energy patterns, pause frequency, speaking rate
        """
        mfcc = voice_features['mfcc']
        pitch = voice_features['pitch']
        energy = voice_features['energy']
        speaking_rate = voice_features['speaking_rate']
        
        # Feature engineering for mental state
        features = []
        
        # 1. Pitch variation (stress/anxiety increases pitch variability)
        features.append(pitch['std'] / (pitch['mean'] + 1e-8))  # Coefficient of variation
        
        # 2. Energy variation (stress affects energy consistency)
        features.append(energy['std'] / (energy['mean'] + 1e-8))
        
        # 3. Speaking rate (anxiety/stress increases rate, fatigue decreases)
        features.append(speaking_rate['rate'])
        
        # 4. Pause ratio (anxiety reduces pauses, fatigue increases)
        features.append(speaking_rate['pause_ratio'])
        
        # 5. MFCC variation (captures overall voice quality changes)
        mfcc_mean = np.array(mfcc['mean'])
        mfcc_std = np.array(mfcc['std'])
        features.append(np.mean(mfcc_std) / (np.mean(np.abs(mfcc_mean)) + 1e-8))
        
        # 6. Pitch mean (stress/anxiety raise pitch)
        features.append(pitch['mean'])
        
        # 7. Energy mean (fatigue reduces energy)
        features.append(energy['mean'])
        
        # 8. High frequency energy (captured in higher MFCC coefficients)
        if len(mfcc['mean']) >= 13:
            high_freq_energy = np.mean(mfcc['mean'][-5:])  # Last 5 MFCC coefficients
            features.append(high_freq_energy)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def train(self, X_train, y_train):
        """
        Train mental state classifier
        X_train: list of voice features dictionaries
        y_train: list of mental state labels ('Calm', 'Stressed', 'Anxious', 'Fatigued')
        """
        # Extract mental state features
        mental_features = []
        for features in X_train:
            mental_feat = self.extract_mental_state_features(features)
            mental_features.append(mental_feat)
        
        X_mental = np.array(mental_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_mental)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        
        # Save model
        self.save_model()
    
    def predict(self, voice_features):
        """
        Predict mental state from voice features
        Returns: (state: str, confidence: float, explanation: str)
        """
        if not self.is_trained:
            # Return default prediction if model not trained
            return 'Calm', 0.5, "Model not trained. Using default prediction."
        
        # Extract mental state features
        mental_feat = self.extract_mental_state_features(voice_features)
        mental_feat = mental_feat.reshape(1, -1)
        
        # Scale features
        mental_feat_scaled = self.scaler.transform(mental_feat)
        
        # Predict
        prediction = self.model.predict(mental_feat_scaled)[0]
        probabilities = self.model.predict_proba(mental_feat_scaled)[0]
        confidence = float(np.max(probabilities))
        
        # Get explanation
        explanation = self._get_explanation(prediction, voice_features)
        
        return prediction, confidence, explanation
    
    def _get_explanation(self, state, voice_features):
        """Generate human-readable explanation for mental state"""
        pitch = voice_features['pitch']
        energy = voice_features['energy']
        speaking_rate = voice_features['speaking_rate']
        
        explanations = {
            'Calm': f"Voice shows normal pitch variation ({pitch['std']:.1f} Hz), "
                   f"steady energy levels, and balanced speaking rate. "
                   f"Indicates relaxed mental state.",
            
            'Stressed': f"Elevated pitch variability ({pitch['std']:.1f} Hz) and "
                       f"increased speaking rate ({speaking_rate['rate']:.2f} events/sec) "
                       f"suggest heightened stress levels.",
            
            'Anxious': f"High pitch variation ({pitch['std']:.1f} Hz), "
                      f"reduced pause ratio ({speaking_rate['pause_ratio']:.2%}), "
                      f"and rapid speech patterns indicate anxiety.",
            
            'Fatigued': f"Lower energy levels ({energy['mean']:.4f}), "
                       f"increased pause ratio ({speaking_rate['pause_ratio']:.2%}), "
                       f"and slower speech patterns suggest fatigue."
        }
        
        return explanations.get(state, "Unable to determine mental state.")
    
    def save_model(self):
        """Save trained model"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
            return True
        return False

