"""
Configuration file for Voice-Based Biometric Authentication System
Contains all system parameters and settings
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")
DATABASE_PATH = os.path.join(BASE_DIR, "voice_auth.db")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Audio processing parameters
SAMPLE_RATE = 16000  # Hz
RECORDING_DURATION = 5  # seconds
N_MFCC = 13  # Number of MFCC coefficients
N_MELS = 128  # Number of mel filter banks
HOP_LENGTH = 512
N_FFT = 2048

# Feature extraction parameters
MFCC_FEATURES = 13
PITCH_FEATURES = 1
ENERGY_FEATURES = 1
SPEAKING_RATE_FEATURES = 1
TOTAL_FEATURES = MFCC_FEATURES + PITCH_FEATURES + ENERGY_FEATURES + SPEAKING_RATE_FEATURES

# Authentication parameters
AUTHENTICATION_THRESHOLD = 0.75  # Confidence threshold for verification
VOICE_AGING_ADAPTATION_RATE = 0.1  # Learning rate for voice aging adaptation (10% new, 90% old)

# Model parameters
SVM_C = 1.0
SVM_GAMMA = 'scale'
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Mental state detection parameters
MENTAL_STATE_CLASSES = ['Calm', 'Stressed', 'Anxious', 'Fatigued']
MENTAL_STATE_COLORS = {
    'Calm': '#28a745',      # Green
    'Stressed': '#ffc107',  # Yellow
    'Anxious': '#fd7e14',   # Orange
    'Fatigued': '#dc3545'   # Red
}

# UI Configuration
UI_TITLE = "Voice-Based Biometric Authentication System"
UI_SUBTITLE = "Voice Aging Adaptation & Cognitive State Analysis"

