# Voice-Based Biometric Authentication System
## Voice Aging Adaptation & Cognitive State Analysis

A comprehensive final-year project implementing voice-based biometric authentication with adaptive learning and mental state detection capabilities.

---

## 🎯 Project Overview

This system provides secure voice-based authentication that:
- **Registers** user voices using advanced feature extraction
- **Verifies** identity through machine learning models
- **Adapts** to voice changes over time (voice aging)
- **Analyzes** mental state from voice patterns (Calm, Stressed, Anxious, Fatigued)

---

## 🏗️ System Architecture

### Components

1. **Feature Extraction Module** (`feature_extractor.py`)
   - MFCC (Mel-Frequency Cepstral Coefficients) extraction
   - Pitch detection using autocorrelation
   - Energy/RMS analysis
   - Speaking rate calculation

2. **Authentication Models** (`auth_model.py`)
   - Support Vector Machine (SVM) - Primary model
   - Convolutional Neural Network (CNN) - Alternative deep learning model
   - Long Short-Term Memory (LSTM) - Sequential pattern recognition

3. **Voice Aging Adaptation** (`voice_aging.py`)
   - Exponential moving average for gradual adaptation
   - Prevents accuracy degradation over time
   - Adaptive threshold management

4. **Mental State Detection** (`mental_state_detector.py`)
   - Random Forest classifier
   - Multi-class classification (4 states)
   - Feature engineering for cognitive analysis

5. **Backend API** (`api.py`)
   - FastAPI RESTful API
   - Endpoints for registration, verification, and logging
   - Database integration

6. **Frontend UI** (`app.py`)
   - Streamlit web application
   - Professional dashboard interface
   - Real-time visualization

---

## 📋 Features

### Voice Registration
- 5-second voice recording
- Automatic feature extraction
- Secure voiceprint storage
- Student ID/Name association

### Voice Verification
- Real-time authentication
- Confidence scoring
- Automatic voice aging adaptation
- Authentication logging

### Mental State Analysis
- **Calm**: Normal voice patterns, balanced speaking rate
- **Stressed**: Elevated pitch variation, increased speaking rate
- **Anxious**: High pitch variability, reduced pauses
- **Fatigued**: Lower energy, increased pause ratio

### Technical Features
- Noise normalization
- Dynamic threshold adjustment
- Accent and speaking-speed tolerance
- Robust feature extraction

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Microphone access (for recording)

### Step 1: Clone/Download Project

```bash
cd D:\voicebased
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** On some systems, you may need to install `pyaudio` separately:
- **Windows**: `pip install pipwin && pipwin install pyaudio`
- **Linux**: `sudo apt-get install portaudio19-dev && pip install pyaudio`
- **Mac**: `brew install portaudio && pip install pyaudio`

### Step 3: Initialize Database

The database will be automatically created on first run. To manually initialize:

```python
from database import init_db
init_db()
```

### Step 4: Start the Backend API

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### Step 5: Start the Frontend

Open a new terminal and run:

```bash
streamlit run app.py
```

The UI will open automatically in your browser at: `http://localhost:8501`

---

## 📖 Usage Guide

### Voice Registration

1. Navigate to **"Voice Registration"** page
2. Click the microphone icon to record
3. Allow browser microphone access
4. Speak clearly for 5 seconds
5. Enter your **Student ID** and **Full Name**
6. Click **"Register Voice"**
7. View extracted features (MFCC, Pitch, Energy, Speaking Rate)

### Voice Verification & Analysis

1. Navigate to **"Voice Verification & Analysis"** page
2. Click the microphone icon to record
3. Speak for 5 seconds
4. Click **"Verify Voice & Analyze Mental State"**
5. View results:
   - Authentication status (Verified/Not Verified)
   - Confidence score (circular gauge)
   - Student information (if verified)
   - Mental state analysis with explanation

### View Students

- Navigate to **"View Students"** page
- See all registered students
- View registration dates

### View Logs

- Navigate to **"View Logs"** page
- See authentication history
- Filter by number of logs
- View confidence scores and mental states

---

## 🔧 Configuration

Edit `config.py` to customize:

- **Audio Parameters**: Sample rate, recording duration
- **Feature Extraction**: MFCC coefficients, mel filters
- **Authentication Threshold**: Confidence threshold (default: 0.75)
- **Voice Aging Rate**: Adaptation rate (default: 0.1 = 10%)
- **Model Parameters**: SVM, CNN, LSTM settings

---

## 🧪 API Endpoints

### POST `/register`
Register a new voice
- **Parameters**: `student_id`, `name`, `audio_file`
- **Response**: Registration status and extracted features

### POST `/verify`
Verify voice and analyze mental state
- **Parameters**: `audio_file`
- **Response**: Authentication result, confidence, mental state

### GET `/students`
Get list of all registered students
- **Response**: List of students with metadata

### GET `/logs`
Get authentication logs
- **Parameters**: `limit` (optional, default: 50)
- **Response**: List of authentication attempts

### GET `/student/{student_id}`
Get specific student information
- **Response**: Student details and voiceprint status

---

## 🧠 Machine Learning Models

### Authentication Models

1. **SVM (Support Vector Machine)**
   - Kernel: RBF
   - Probability estimates enabled
   - Fast training and inference

2. **CNN (Convolutional Neural Network)**
   - 3 convolutional blocks
   - Batch normalization
   - Dropout regularization
   - Global average pooling

3. **LSTM (Long Short-Term Memory)**
   - Sequential feature analysis
   - 2 LSTM layers
   - Temporal pattern recognition

### Mental State Detection

- **Random Forest Classifier**
  - 100 estimators
  - Balanced class weights
  - Feature engineering for cognitive analysis

---

## 📊 Feature Extraction Details

### MFCC (13 coefficients)
- Captures spectral characteristics
- Mean and standard deviation across time frames
- Robust to noise

### Pitch
- Fundamental frequency detection
- Mean and standard deviation
- Unique to each speaker

### Energy
- RMS energy calculation
- Mean, std, max, min values
- Amplitude patterns

### Speaking Rate
- Onset detection
- Events per second
- Pause ratio calculation

**Total Feature Vector**: 32 features (13 MFCC mean + 13 MFCC std + 2 pitch + 2 energy + 2 speaking rate)

---

## 🔄 Voice Aging Adaptation

The system uses **exponential moving average** to adapt voiceprints:

```
new_voiceprint = (1 - α) × old_voiceprint + α × new_features
```

Where `α = 0.1` (10% adaptation rate)

**Benefits:**
- Prevents accuracy degradation over time
- Adapts to natural voice changes
- Maintains security (only adapts on verified authentications)

---

## 🧪 Testing

### Manual Testing

1. Register multiple students
2. Verify each student's voice
3. Test with different mental states
4. Verify voice aging adaptation over time

### API Testing

Use tools like Postman or curl:

```bash
# Register
curl -X POST "http://localhost:8000/register" \
  -F "student_id=STU001" \
  -F "name=John Doe" \
  -F "audio_file=@voice.wav"

# Verify
curl -X POST "http://localhost:8000/verify" \
  -F "audio_file=@voice.wav"
```

---

## 📁 Project Structure

```
voicebased/
├── api.py                 # FastAPI backend
├── app.py                 # Streamlit frontend
├── config.py              # Configuration
├── database.py            # Database models
├── feature_extractor.py   # Voice feature extraction
├── auth_model.py          # ML authentication models
├── mental_state_detector.py  # Mental state classification
├── voice_aging.py         # Voice aging adaptation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Saved ML models (auto-created)
├── uploads/              # Temporary audio files (auto-created)
└── voice_auth.db         # SQLite database (auto-created)
```

---

## 🎓 Viva/Examination Points

### Key Technical Points

1. **Feature Extraction**
   - Why MFCC? Captures spectral envelope, robust to noise
   - Pitch detection: Fundamental frequency unique to speaker
   - Energy patterns: Amplitude characteristics

2. **Machine Learning**
   - SVM: Fast, effective for high-dimensional features
   - CNN: Learns hierarchical patterns
   - LSTM: Captures temporal dependencies

3. **Voice Aging Adaptation**
   - Exponential moving average prevents degradation
   - Only adapts on verified authentications
   - Maintains security while improving accuracy

4. **Mental State Detection**
   - Feature engineering: Pitch variation, energy patterns
   - Multi-class classification
   - Real-world applications: Healthcare, security

### System Advantages

- **Robustness**: Noise normalization, dynamic thresholds
- **Adaptability**: Voice aging adaptation
- **Security**: Confidence scoring, authentication logging
- **Usability**: Professional UI, real-time feedback

### Limitations & Future Work

- **Noise**: Works best in quiet environments
- **Accent**: May need retraining for different accents
- **Emotion**: Mental state detection is indicative, not diagnostic
- **Future**: Real-time streaming, multi-language support, cloud deployment

---

## 🐛 Troubleshooting

### Audio Recording Issues

- **No microphone access**: Check browser permissions
- **Audio format error**: Ensure WAV format, 16kHz sample rate
- **Recording too short**: Must be at least 5 seconds

### API Connection Errors

- **Connection refused**: Ensure backend is running on port 8000
- **Timeout**: Increase timeout in requests
- **CORS errors**: Check CORS settings in `api.py`

### Model Loading Errors

- **Model not found**: Models are created on first training
- **Version mismatch**: Reinstall TensorFlow if needed

---

## 📝 License

This project is created for educational purposes as a final-year project.

---

## 👨‍💻 Author

Final Year Project - Voice-Based Biometric Authentication System

---

## 🙏 Acknowledgments

- Librosa for audio processing
- Scikit-learn for ML algorithms
- TensorFlow for deep learning
- FastAPI for backend framework
- Streamlit for frontend framework

---

## 📧 Support

For questions or issues, refer to the code comments and documentation.

---

**Good luck with your final year project! 🎓**

