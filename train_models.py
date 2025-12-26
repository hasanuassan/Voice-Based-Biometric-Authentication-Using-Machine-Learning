"""
Training Script for Voice Authentication Models
This script helps train the ML models with sample data
Note: In production, models are trained on-the-fly during registration
"""

import numpy as np
from database import init_db, SessionLocal, Student, VoicePrint
from auth_model import VoiceAuthenticationModel
from mental_state_detector import MentalStateDetector
from feature_extractor import VoiceFeatureExtractor
import config

def create_sample_data():
    """
    Create sample training data for demonstration
    In real usage, data comes from actual registrations
    """
    print("Creating sample training data...")
    
    # This is a placeholder - in real usage, you would:
    # 1. Register multiple students
    # 2. Collect their voiceprints
    # 3. Use those for training
    
    # For demonstration, we create synthetic feature vectors
    n_samples = 20
    n_features = 32  # Based on feature extraction
    
    # Generate synthetic feature vectors
    X_train = np.random.randn(n_samples, n_features)
    
    # Normalize to realistic ranges
    # MFCC features (first 26: 13 mean + 13 std)
    X_train[:, :26] = X_train[:, :26] * 5 + 0  # MFCC typically -20 to 20
    
    # Pitch (features 26-27)
    X_train[:, 26:28] = np.abs(X_train[:, 26:28]) * 50 + 100  # Pitch 100-200 Hz
    
    # Energy (features 28-29)
    X_train[:, 28:30] = np.abs(X_train[:, 28:30]) * 0.1 + 0.05  # Energy 0.05-0.15
    
    # Speaking rate (features 30-31)
    X_train[:, 30:32] = np.abs(X_train[:, 30:32]) * 2 + 1  # Speaking rate 1-3
    
    # Binary labels (1 = match, 0 = no match)
    # For demonstration: first 10 are matches, rest are non-matches
    y_train = np.array([1] * 10 + [0] * 10)
    
    return X_train, y_train

def train_authentication_model():
    """Train the authentication model"""
    print("=" * 50)
    print("Training Authentication Model (SVM)")
    print("=" * 50)
    
    # Create sample data
    X_train, y_train = create_sample_data()
    
    # Initialize and train model
    model = VoiceAuthenticationModel(model_type='svm')
    model.train_svm(X_train, y_train)
    
    print("✅ Authentication model trained and saved!")
    print(f"Model saved to: {model.model_path}")
    return model

def train_mental_state_model():
    """Train the mental state detection model"""
    print("=" * 50)
    print("Training Mental State Detection Model")
    print("=" * 50)
    
    # Create sample voice features
    detector = MentalStateDetector()
    
    # Generate sample features
    sample_features = []
    labels = []
    
    for i in range(20):
        # Create synthetic voice features
        features = {
            'mfcc': {
                'mean': np.random.randn(13).tolist(),
                'std': np.abs(np.random.randn(13)).tolist()
            },
            'pitch': {
                'mean': 150 + np.random.randn() * 20,
                'std': 10 + np.abs(np.random.randn()) * 5
            },
            'energy': {
                'mean': 0.1 + np.random.randn() * 0.02,
                'std': 0.01 + np.abs(np.random.randn()) * 0.01
            },
            'speaking_rate': {
                'rate': 2.0 + np.random.randn() * 0.5,
                'pause_ratio': 0.2 + np.random.randn() * 0.1
            }
        }
        
        sample_features.append(features)
        
        # Assign labels (balanced distribution)
        labels.append(config.MENTAL_STATE_CLASSES[i % 4])
    
    # Train model
    detector.train(sample_features, labels)
    
    print("✅ Mental state detection model trained and saved!")
    print(f"Model saved to: {detector.model_path}")
    return detector

def main():
    """Main training function"""
    print("🎤 Voice Authentication Model Training")
    print("=" * 50)
    print()
    
    # Initialize database
    print("Initializing database...")
    init_db()
    print("✅ Database initialized")
    print()
    
    # Train authentication model
    try:
        train_authentication_model()
        print()
    except Exception as e:
        print(f"⚠️  Warning: Could not train authentication model: {e}")
        print("   Models will be trained on-the-fly during registration")
        print()
    
    # Train mental state model
    try:
        train_mental_state_model()
        print()
    except Exception as e:
        print(f"⚠️  Warning: Could not train mental state model: {e}")
        print("   Model will use default predictions")
        print()
    
    print("=" * 50)
    print("✅ Training complete!")
    print()
    print("Note: In production, models learn from actual user registrations.")
    print("This script is for demonstration purposes only.")

if __name__ == "__main__":
    main()

