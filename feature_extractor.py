"""
Voice Feature Extraction Module
Extracts MFCC, Pitch, Energy, and Speaking Rate features from audio
"""

import librosa
import numpy as np
import json
from scipy import signal
from config import *





class VoiceFeatureExtractor:
    """Extract comprehensive voice features for biometric authentication"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.n_mfcc = N_MFCC
        self.n_mels = N_MELS
        self.hop_length = HOP_LENGTH
        self.n_fft = N_FFT
    



    def extract_mfcc(self, audio, sr=None):
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCC)
        MFCC captures the spectral characteristics of voice
        """
        if sr is None:
            sr = self.sample_rate
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Return mean and std across time frames
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        return {
            'mean': mfcc_mean.tolist(),
            'std': mfcc_std.tolist(),
            'full': mfccs.tolist()
        }
    
    def extract_pitch(self, audio, sr=None):
        """
        Extract fundamental frequency (pitch) using autocorrelation
        Pitch is a key biometric feature unique to each individual
        """
        if sr is None:
            sr = self.sample_rate
        
        # Use librosa's pitch detection
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=self.hop_length)
        
        # Extract pitch values (non-zero)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return 0.0
        
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        
        return {
            'mean': float(pitch_mean),
            'std': float(pitch_std),
            'values': [float(p) for p in pitch_values]
        }
    
    def extract_energy(self, audio, sr=None):
        """
        Extract energy/amplitude features
        Energy patterns are unique to each speaker
        """
        if sr is None:
            sr = self.sample_rate
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_max = np.max(rms)
        energy_min = np.min(rms)
        
        return {
            'mean': float(energy_mean),
            'std': float(energy_std),
            'max': float(energy_max),
            'min': float(energy_min)
        }
    
    def extract_speaking_rate(self, audio, sr=None):
        """
        Extract speaking rate (words per second equivalent)
        Calculated using onset detection and temporal features
        """
        if sr is None:
            sr = self.sample_rate
        
        # Detect onsets (speech events)
        onsets = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            units='time'
        )
        
        # Calculate speaking rate (onsets per second)
        duration = len(audio) / sr
        if duration > 0:
            speaking_rate = len(onsets) / duration
        else:
            speaking_rate = 0.0
        
        # Calculate pause ratio (silence detection)
        # Use energy-based voice activity detection
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        energy_threshold = np.percentile(rms, 20)  # Bottom 20% considered silence
        silence_frames = np.sum(rms < energy_threshold)
        total_frames = len(rms)
        pause_ratio = silence_frames / total_frames if total_frames > 0 else 0.0
        
        return {
            'rate': float(speaking_rate),
            'pause_ratio': float(pause_ratio),
            'onsets': len(onsets)
        }
    
    def normalize_audio(self, audio):
        """
        Normalize audio to handle different recording volumes
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize amplitude
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio = audio / max_amplitude
        
        return audio
    
    def extract_all_features(self, audio_path=None, audio_array=None, sr=None):
        """
        Extract all voice features from audio file or array
        Returns comprehensive feature vector for authentication
        """
        # Load audio if path is provided
        if audio_path:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=RECORDING_DURATION)
        elif audio_array is not None:
            if sr is None:
                sr = self.sample_rate
            audio = audio_array
        else:
            raise ValueError("Either audio_path or audio_array must be provided")
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Extract all features
        mfcc_features = self.extract_mfcc(audio, sr)
        pitch_features = self.extract_pitch(audio, sr)
        energy_features = self.extract_energy(audio, sr)
        speaking_rate_features = self.extract_speaking_rate(audio, sr)
        
        # Create feature vector for ML model
        # Combine: MFCC mean (13) + MFCC std (13) + pitch mean (1) + pitch std (1) + 
        #          energy mean (1) + energy std (1) + speaking rate (1) + pause ratio (1)
        feature_vector = np.concatenate([
            mfcc_features['mean'],
            mfcc_features['std'],
            [pitch_features['mean']],
            [pitch_features['std']],
            [energy_features['mean']],
            [energy_features['std']],
            [speaking_rate_features['rate']],
            [speaking_rate_features['pause_ratio']]
        ])
        
        return {
            'mfcc': mfcc_features,
            'pitch': pitch_features,
            'energy': energy_features,
            'speaking_rate': speaking_rate_features,
            'feature_vector': feature_vector.tolist(),
            'feature_vector_shape': feature_vector.shape
        }
    
    def features_to_json(self, features):
        """Convert features to JSON-serializable format"""
        return json.dumps(features)
    
    def json_to_features(self, json_str):
        """Parse JSON string back to features"""
        return json.loads(json_str)

