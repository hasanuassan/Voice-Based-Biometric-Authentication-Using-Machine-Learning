"""
System Test Script
Tests all components of the Voice Authentication System
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import fastapi
        import streamlit
        import librosa
        import sklearn
        import tensorflow
        import numpy
        import sqlalchemy
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database initialization"""
    print("\nTesting database...")
    try:
        from database import init_db, SessionLocal, Student, VoicePrint
        init_db()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_feature_extractor():
    """Test feature extraction"""
    print("\nTesting feature extractor...")
    try:
        from feature_extractor import VoiceFeatureExtractor
        import numpy as np
        
        extractor = VoiceFeatureExtractor()
        
        # Create dummy audio data
        dummy_audio = np.random.randn(16000 * 5)  # 5 seconds at 16kHz
        
        # Test feature extraction
        features = extractor.extract_all_features(audio_array=dummy_audio, sr=16000)
        
        assert 'mfcc' in features
        assert 'pitch' in features
        assert 'energy' in features
        assert 'speaking_rate' in features
        assert 'feature_vector' in features
        
        print(f"✅ Feature extraction successful")
        print(f"   Feature vector shape: {len(features['feature_vector'])}")
        return True
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return False

def test_auth_model():
    """Test authentication model"""
    print("\nTesting authentication model...")
    try:
        from auth_model import VoiceAuthenticationModel
        import numpy as np
        
        model = VoiceAuthenticationModel(model_type='svm')
        
        # Test similarity calculation (doesn't require training)
        vec1 = np.random.randn(32)
        vec2 = np.random.randn(32)
        
        similarity = model.calculate_similarity(vec1, vec2)
        assert 0 <= similarity <= 1
        
        print(f"✅ Authentication model initialized")
        print(f"   Similarity calculation works: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"❌ Authentication model error: {e}")
        return False

def test_mental_state_detector():
    """Test mental state detector"""
    print("\nTesting mental state detector...")
    try:
        from mental_state_detector import MentalStateDetector
        
        detector = MentalStateDetector()
        
        # Test with dummy features
        dummy_features = {
            'mfcc': {
                'mean': [0] * 13,
                'std': [1] * 13
            },
            'pitch': {
                'mean': 150.0,
                'std': 10.0
            },
            'energy': {
                'mean': 0.1,
                'std': 0.01
            },
            'speaking_rate': {
                'rate': 2.0,
                'pause_ratio': 0.2
            }
        }
        
        # Should work even without training (returns default)
        state, confidence, explanation = detector.predict(dummy_features)
        
        print(f"✅ Mental state detector initialized")
        print(f"   Default prediction: {state} ({confidence:.2f})")
        return True
    except Exception as e:
        print(f"❌ Mental state detector error: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        import config
        
        assert hasattr(config, 'SAMPLE_RATE')
        assert hasattr(config, 'RECORDING_DURATION')
        assert hasattr(config, 'AUTHENTICATION_THRESHOLD')
        
        print("✅ Configuration loaded successfully")
        print(f"   Sample rate: {config.SAMPLE_RATE} Hz")
        print(f"   Recording duration: {config.RECORDING_DURATION} seconds")
        print(f"   Auth threshold: {config.AUTHENTICATION_THRESHOLD}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Voice Authentication System Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_imports,
        test_config,
        test_database,
        test_feature_extractor,
        test_auth_model,
        test_mental_state_detector
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

