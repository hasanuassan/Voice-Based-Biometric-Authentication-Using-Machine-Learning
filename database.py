"""
Database models and operations for Voice Authentication System
Uses SQLAlchemy for database management
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import config

Base = declarative_base()

class Student(Base):
    """Student model for storing student information and voiceprints"""
    __tablename__ = 'students'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Student(student_id='{self.student_id}', name='{self.name}')>"

class VoicePrint(Base):
    """Voiceprint model for storing extracted voice features"""
    __tablename__ = 'voiceprints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), nullable=False, index=True)
    mfcc_features = Column(Text, nullable=False)  # JSON string of MFCC features
    pitch_mean = Column(Float, nullable=False)
    energy_mean = Column(Float, nullable=False)
    speaking_rate = Column(Float, nullable=False)
    feature_vector = Column(Text, nullable=False)  # Complete feature vector as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<VoicePrint(student_id='{self.student_id}')>"

class AuthenticationLog(Base):
    """Log model for tracking authentication attempts"""
    __tablename__ = 'authentication_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), nullable=True, index=True)
    result = Column(String(20), nullable=False)  # 'Verified' or 'Not Verified'
    confidence_score = Column(Float, nullable=False)
    mental_state = Column(String(20), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AuthenticationLog(student_id='{self.student_id}', result='{self.result}')>"

# Database engine and session
engine = create_engine(f'sqlite:///{config.DATABASE_PATH}', echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

