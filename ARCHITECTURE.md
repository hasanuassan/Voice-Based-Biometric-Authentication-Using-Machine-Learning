# System Architecture Documentation

## Overview

The Voice-Based Biometric Authentication System is built using a modular architecture with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer (Streamlit)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Registration │  │ Verification │  │   Analytics   │     │
│  │    Page      │  │     Page     │  │     Page     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼────────────────────────────────┐
│                  API Layer (FastAPI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ /register│  │ /verify  │  │ /students│  │  /logs   │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│              Business Logic Layer                            │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ Feature Extractor │  │  Auth Model      │               │
│  │  - MFCC          │  │  - SVM           │               │
│  │  - Pitch         │  │  - CNN           │               │
│  │  - Energy        │  │  - LSTM          │               │
│  │  - Speaking Rate │  └──────────────────┘               │
│  └──────────────────┘  ┌──────────────────┐               │
│                        │ Mental State      │               │
│                        │ Detector          │               │
│                        └──────────────────┘               │
│  ┌──────────────────┐                                     │
│  │ Voice Aging      │                                     │
│  │ Adapter          │                                     │
│  └──────────────────┘                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  Data Layer                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Students   │  │  Voiceprints │  │     Logs     │    │
│  │   Table      │  │    Table     │  │    Table     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│                    SQLite Database                          │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend Layer (Streamlit)

**File**: `app.py`

**Responsibilities**:
- User interface rendering
- Audio recording interface
- Visualization (waveforms, confidence meters)
- User interaction handling

**Key Features**:
- Multi-page navigation
- Real-time audio visualization
- Interactive charts (Plotly)
- Professional styling

### 2. API Layer (FastAPI)

**File**: `api.py`

**Responsibilities**:
- RESTful API endpoints
- Request/response handling
- Error management
- CORS configuration

**Endpoints**:
- `POST /register` - Voice registration
- `POST /verify` - Voice verification
- `GET /students` - List students
- `GET /logs` - Authentication logs
- `GET /student/{id}` - Student details

### 3. Feature Extraction Module

**File**: `feature_extractor.py`

**Class**: `VoiceFeatureExtractor`

**Methods**:
- `extract_mfcc()` - Mel-Frequency Cepstral Coefficients
- `extract_pitch()` - Fundamental frequency
- `extract_energy()` - RMS energy
- `extract_speaking_rate()` - Temporal features
- `extract_all_features()` - Complete feature vector

**Output**: 32-dimensional feature vector

### 4. Authentication Models

**File**: `auth_model.py`

**Class**: `VoiceAuthenticationModel`

**Supported Models**:
1. **SVM** (Support Vector Machine)
   - RBF kernel
   - Probability estimates
   - Fast inference

2. **CNN** (Convolutional Neural Network)
   - 3 convolutional blocks
   - Batch normalization
   - Global average pooling

3. **LSTM** (Long Short-Term Memory)
   - Sequential processing
   - Temporal dependencies
   - Recurrent layers

**Methods**:
- `train_svm()` / `train_cnn()` / `train_lstm()`
- `predict()` - Authentication prediction
- `calculate_similarity()` - Feature comparison

### 5. Mental State Detection

**File**: `mental_state_detector.py`

**Class**: `MentalStateDetector`

**Model**: Random Forest Classifier

**States**:
- Calm
- Stressed
- Anxious
- Fatigued

**Features**:
- Pitch variation coefficient
- Energy variation
- Speaking rate
- Pause ratio
- MFCC characteristics

### 6. Voice Aging Adaptation

**File**: `voice_aging.py`

**Class**: `VoiceAgingAdapter`

**Algorithm**: Exponential Moving Average

```
new_voiceprint = (1 - α) × old_voiceprint + α × new_features
```

**Adaptation Rate**: 10% (configurable)

**Conditions**:
- Only adapts on verified authentications
- Similarity threshold: 0.85
- Prevents unauthorized adaptation

### 7. Data Layer

**File**: `database.py`

**Database**: SQLite

**Tables**:

1. **Students**
   - id (Primary Key)
   - student_id (Unique)
   - name
   - created_at
   - updated_at

2. **Voiceprints**
   - id (Primary Key)
   - student_id (Foreign Key)
   - mfcc_features (JSON)
   - pitch_mean
   - energy_mean
   - speaking_rate
   - feature_vector (JSON)
   - created_at
   - updated_at

3. **AuthenticationLogs**
   - id (Primary Key)
   - student_id
   - result
   - confidence_score
   - mental_state
   - timestamp

## Data Flow

### Registration Flow

```
User → Frontend → API → Feature Extractor → Database
                      ↓
                   Voiceprint Storage
```

1. User records voice in frontend
2. Audio sent to `/register` endpoint
3. Feature extraction (MFCC, Pitch, Energy, Speaking Rate)
4. Store voiceprint in database
5. Return success confirmation

### Verification Flow

```
User → Frontend → API → Feature Extractor → Auth Model → Database
                                              ↓
                                        Mental State Detector
                                              ↓
                                        Voice Aging Adapter
```

1. User records voice for verification
2. Extract features from audio
3. Compare with stored voiceprints
4. Calculate similarity scores
5. Determine authentication result
6. Analyze mental state
7. Adapt voiceprint if verified
8. Log authentication attempt
9. Return results to frontend

## Security Considerations

1. **Authentication Threshold**: 0.75 confidence required
2. **Voice Aging**: Only adapts on verified authentications
3. **Database**: SQLite with proper indexing
4. **Input Validation**: Audio format and duration checks
5. **Error Handling**: Graceful failure without exposing internals

## Performance Optimizations

1. **Feature Caching**: Extracted features stored in database
2. **Model Persistence**: Trained models saved to disk
3. **Batch Processing**: Multiple feature extractions
4. **Database Indexing**: Fast lookups on student_id
5. **Audio Normalization**: Preprocessing for consistency

## Scalability Considerations

**Current Limitations**:
- Single-threaded SQLite database
- In-memory model loading
- Local file storage

**Future Improvements**:
- PostgreSQL for multi-user support
- Model serving API (TensorFlow Serving)
- Cloud storage (S3, Azure Blob)
- Redis caching
- Load balancing

## Testing Strategy

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: API endpoint testing
3. **End-to-End Tests**: Full workflow testing
4. **Performance Tests**: Load and stress testing

## Deployment

**Development**:
- Local SQLite database
- Single server deployment
- Direct model loading

**Production** (Recommended):
- PostgreSQL database
- Containerized deployment (Docker)
- Model serving service
- CDN for static assets
- Monitoring and logging

## Technology Stack

- **Backend**: FastAPI, Python 3.8+
- **Frontend**: Streamlit
- **ML**: Scikit-learn, TensorFlow/Keras
- **Audio**: Librosa, SoundFile
- **Database**: SQLite (SQLAlchemy ORM)
- **Visualization**: Plotly, Matplotlib
- **API**: RESTful architecture

## Configuration Management

**File**: `config.py`

Centralized configuration for:
- Audio parameters
- Feature extraction settings
- Model hyperparameters
- Authentication thresholds
- UI settings

## Error Handling

- **API Errors**: HTTP status codes with descriptive messages
- **Model Errors**: Graceful fallbacks
- **Database Errors**: Transaction rollback
- **Audio Errors**: Format validation and normalization

## Logging

- Authentication attempts logged
- Error logging for debugging
- Performance metrics tracking
- User activity monitoring

---

This architecture provides a solid foundation for a production-ready voice authentication system while maintaining code clarity and modularity for academic presentation.

