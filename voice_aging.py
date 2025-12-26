"""
Voice Aging Adaptation Module
Implements gradual learning to adapt to voice changes over time
"""

import numpy as np
import json
from database import VoicePrint, SessionLocal
from config import VOICE_AGING_ADAPTATION_RATE

class VoiceAgingAdapter:
    """Adapt stored voiceprints to account for natural voice changes"""
    
    def __init__(self):
        self.adaptation_rate = VOICE_AGING_ADAPTATION_RATE
    
    def adapt_voiceprint(self, student_id, new_feature_vector, db_session):
        """
        Adapt existing voiceprint with new features
        Uses exponential moving average: new_voiceprint = (1-α) * old + α * new
        where α is the adaptation rate
        """
        # Get existing voiceprint
        voiceprint = db_session.query(VoicePrint).filter(
            VoicePrint.student_id == student_id
        ).first()
        
        if not voiceprint:
            return False
        
        # Parse existing features
        old_feature_vector = json.loads(voiceprint.feature_vector)
        old_feature_vector = np.array(old_feature_vector)
        new_feature_vector = np.array(new_feature_vector)
        
        # Ensure same dimensions
        if old_feature_vector.shape != new_feature_vector.shape:
            return False
        
        # Exponential moving average adaptation
        # Higher adaptation_rate means more weight to new features
        adapted_vector = (1 - self.adaptation_rate) * old_feature_vector + \
                        self.adaptation_rate * new_feature_vector
        
        # Update voiceprint features
        # Update MFCC (mean and std separately)
        old_mfcc = json.loads(voiceprint.mfcc_features)
        new_mfcc_mean = np.array(new_feature_vector[:13])  # First 13 are MFCC mean
        new_mfcc_std = np.array(new_feature_vector[13:26])  # Next 13 are MFCC std
        
        adapted_mfcc_mean = (1 - self.adaptation_rate) * np.array(old_mfcc['mean']) + \
                           self.adaptation_rate * new_mfcc_mean
        adapted_mfcc_std = (1 - self.adaptation_rate) * np.array(old_mfcc['std']) + \
                          self.adaptation_rate * new_mfcc_std
        
        # Update pitch, energy, speaking rate
        feature_idx = 26  # After MFCC mean (13) + MFCC std (13)
        adapted_pitch_mean = (1 - self.adaptation_rate) * voiceprint.pitch_mean + \
                            self.adaptation_rate * new_feature_vector[feature_idx]
        feature_idx += 1
        adapted_energy_mean = (1 - self.adaptation_rate) * voiceprint.energy_mean + \
                             self.adaptation_rate * new_feature_vector[feature_idx]
        feature_idx += 1
        adapted_speaking_rate = (1 - self.adaptation_rate) * voiceprint.speaking_rate + \
                               self.adaptation_rate * new_feature_vector[feature_idx]
        
        # Save adapted features
        voiceprint.mfcc_features = json.dumps({
            'mean': adapted_mfcc_mean.tolist(),
            'std': adapted_mfcc_std.tolist()
        })
        voiceprint.pitch_mean = float(adapted_pitch_mean)
        voiceprint.energy_mean = float(adapted_energy_mean)
        voiceprint.speaking_rate = float(adapted_speaking_rate)
        voiceprint.feature_vector = json.dumps(adapted_vector.tolist())
        
        db_session.commit()
        return True
    
    def should_adapt(self, old_features, new_features, similarity_threshold=0.85):
        """
        Determine if voiceprint should be adapted
        Only adapt if similarity is high enough (verified user)
        """
        old_vec = np.array(old_features)
        new_vec = np.array(new_features)
        
        # Normalize vectors
        old_norm = old_vec / (np.linalg.norm(old_vec) + 1e-8)
        new_norm = new_vec / (np.linalg.norm(new_vec) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(old_norm, new_norm)
        
        # Adapt if similarity is above threshold (user is verified)
        return similarity >= similarity_threshold

