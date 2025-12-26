"""
FastAPI Backend for Voice-Based Biometric Authentication System
Provides REST API endpoints for voice registration, verification, and analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import json
import os
import tempfile
from typing import Optional


from database import init_db, get_db, Student, VoicePrint, AuthenticationLog
from feature_extractor import VoiceFeatureExtractor
from auth_model import VoiceAuthenticationModel
from mental_state_detector import MentalStateDetector
from voice_aging import VoiceAgingAdapter
from attendance_manager import AttendanceManager
import config

# Initialize FastAPI app
app = FastAPI(
    title="Voice-Based Biometric Authentication API",
    description="API for voice registration, verification, and mental state analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
feature_extractor = VoiceFeatureExtractor()
auth_model = VoiceAuthenticationModel(model_type='svm')
mental_detector = MentalStateDetector()
aging_adapter = VoiceAgingAdapter()
attendance_manager = AttendanceManager()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    # Try to load existing models
    auth_model.load_model()
    mental_detector.load_model()

# Pydantic models for request/response
class RegistrationRequest(BaseModel):
    student_id: str
    name: str

class VerificationResponse(BaseModel):
    verified: bool
    confidence: float
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    mental_state: Optional[str] = None
    mental_confidence: Optional[float] = None
    explanation: Optional[str] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice-Based Biometric Authentication API",
        "version": "1.0.0",
        "endpoints": {
            "register": "/register",
            "verify": "/verify",
            "students": "/students",
            "logs": "/logs"
        }
    }

@app.post("/register")
async def register_voice(
    student_id: str = Form(...),
    name: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Register a new student's voice
    Records voiceprint and stores it in database
    """
    try:
        # Check if student already exists
        existing_student = db.query(Student).filter(
            Student.student_id == student_id
        ).first()
        
        if existing_student:
            # Update existing student
            existing_student.name = name
            student = existing_student
        else:
            # Create new student
            student = Student(student_id=student_id, name=name)
            db.add(student)
            db.commit()
            db.refresh(student)
        
        # Read audio content
        content = await audio_file.read()
        
        # Save uploaded audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Save registration recording in daily folder
            recording_path = attendance_manager.save_recording(
                content, student_id, recording_type="registration"
            )
            
            # Extract voice features
            features = feature_extractor.extract_all_features(audio_path=tmp_path)
            
            # Store voiceprint
            existing_voiceprint = db.query(VoicePrint).filter(
                VoicePrint.student_id == student_id
            ).first()
            
            if existing_voiceprint:
                # Update existing voiceprint
                existing_voiceprint.mfcc_features = json.dumps(features['mfcc'])
                existing_voiceprint.pitch_mean = features['pitch']['mean']
                existing_voiceprint.energy_mean = features['energy']['mean']
                existing_voiceprint.speaking_rate = features['speaking_rate']['rate']
                existing_voiceprint.feature_vector = json.dumps(features['feature_vector'])
                voiceprint = existing_voiceprint
            else:
                # Create new voiceprint
                voiceprint = VoicePrint(
                    student_id=student_id,
                    mfcc_features=json.dumps(features['mfcc']),
                    pitch_mean=features['pitch']['mean'],
                    energy_mean=features['energy']['mean'],
                    speaking_rate=features['speaking_rate']['rate'],
                    feature_vector=json.dumps(features['feature_vector'])
                )
                db.add(voiceprint)
            
            db.commit()
            
            return {
                "success": True,
                "message": f"Voice registered successfully for {name}",
                "student_id": student_id,
                "recording_saved": recording_path,
                "features_extracted": {
                    "mfcc_coefficients": len(features['mfcc']['mean']),
                    "pitch_mean": features['pitch']['mean'],
                    "energy_mean": features['energy']['mean'],
                    "speaking_rate": features['speaking_rate']['rate']
                }
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/verify")
async def verify_voice(
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify a voice against registered voiceprints
    Returns authentication result and mental state analysis
    """
    try:
        # Read audio content
        content = await audio_file.read()
        
        # Save uploaded audio file temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Initialize variables for attendance
        recording_path = None
        student_name_for_attendance = None
        
        try:
            # Extract voice features
            features = feature_extractor.extract_all_features(audio_path=tmp_path)
            feature_vector = np.array(features['feature_vector'])
            
            # Get all registered voiceprints
            voiceprints = db.query(VoicePrint).all()
            
            if not voiceprints:
                return {
                    "verified": False,
                    "confidence": 0.0,
                    "message": "No registered voices found"
                }
            
            # Find best match
            best_match = None
            best_confidence = 0.0
            best_student_id = None
            
            for vp in voiceprints:
                stored_vector = np.array(json.loads(vp.feature_vector))
                
                # Calculate similarity using cosine similarity
                similarity = auth_model.calculate_similarity(feature_vector, stored_vector)
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = vp
                    best_student_id = vp.student_id
            
            # Check if confidence meets threshold
            is_verified = best_confidence >= config.AUTHENTICATION_THRESHOLD
            
            # Get student information if verified
            student_name = None
            if is_verified and best_student_id:
                student = db.query(Student).filter(
                    Student.student_id == best_student_id
                ).first()
                if student:
                    student_name = student.name
                
                # Apply voice aging adaptation if verified
                if aging_adapter.should_adapt(
                    json.loads(best_match.feature_vector),
                    features['feature_vector']
                ):
                    aging_adapter.adapt_voiceprint(
                        best_student_id,
                        features['feature_vector'],
                        db
                    )
            
            # Detect mental state
            mental_state, mental_confidence, explanation = mental_detector.predict(features)
            
            # Save recording in daily folder and update attendance
            if is_verified and best_student_id:
                # Save verification recording
                recording_path = attendance_manager.save_recording(
                    content, best_student_id, recording_type="verification"
                )
                student_name_for_attendance = student_name
            else:
                # Save as unknown verification attempt
                recording_path = attendance_manager.save_recording(
                    content, "UNKNOWN", recording_type="verification"
                )
            
            # Update attendance Excel sheet
            try:
                attendance_manager.add_attendance_entry(
                    student_id=best_student_id if is_verified else "UNKNOWN",
                    student_name=student_name_for_attendance or "Unknown",
                    verified=is_verified,
                    confidence=best_confidence,
                    mental_state=mental_state,
                    recording_path=recording_path
                )
            except Exception as e:
                # Log error but don't fail the request
                print(f"Error updating attendance: {e}")
            
            # Log authentication attempt
            log_entry = AuthenticationLog(
                student_id=best_student_id if is_verified else None,
                result="Verified" if is_verified else "Not Verified",
                confidence_score=best_confidence,
                mental_state=mental_state
            )
            db.add(log_entry)
            db.commit()
            
            return {
                "verified": is_verified,
                "confidence": float(best_confidence),
                "student_id": best_student_id if is_verified else None,
                "student_name": student_name,
                "mental_state": mental_state,
                "mental_confidence": float(mental_confidence),
                "explanation": explanation,
                "recording_saved": recording_path is not None
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/students")
async def get_students(db: Session = Depends(get_db)):
    """Get list of all registered students"""
    students = db.query(Student).all()
    return {
        "students": [
            {
                "student_id": s.student_id,
                "name": s.name,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None
            }
            for s in students
        ]
    }

@app.get("/logs")
async def get_logs(
    limit: int = 50, 
    date: str = None,
    start_time: str = None,
    end_time: str = None,
    search_name: str = None,
    db: Session = Depends(get_db)
):
    """Get authentication logs with filtering options"""
    query = db.query(AuthenticationLog)
    
    # Filter by date
    if date:
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            query = query.filter(
                db.func.date(AuthenticationLog.timestamp) == date_obj
            )
        except:
            pass
    
    # Filter by time range
    if start_time:
        try:
            start_datetime = datetime.strptime(f"{date or datetime.now().date()} {start_time}", "%Y-%m-%d %H:%M")
            query = query.filter(AuthenticationLog.timestamp >= start_datetime)
        except:
            pass
    
    if end_time:
        try:
            end_datetime = datetime.strptime(f"{date or datetime.now().date()} {end_time}", "%Y-%m-%d %H:%M")
            query = query.filter(AuthenticationLog.timestamp <= end_datetime)
        except:
            pass
    
    # Filter by student name (search in student_id)
    if search_name:
        # Get student IDs that match the search
        students = db.query(Student).filter(
            (Student.name.ilike(f"%{search_name}%")) | 
            (Student.student_id.ilike(f"%{search_name}%"))
        ).all()
        student_ids = [s.student_id for s in students]
        if student_ids:
            query = query.filter(AuthenticationLog.student_id.in_(student_ids))
        else:
            # No matches, return empty
            query = query.filter(False)
    
    logs = query.order_by(AuthenticationLog.timestamp.desc()).limit(limit).all()
    
    # Get student names for each log
    result_logs = []
    for log in logs:
        student_name = None
        if log.student_id:
            student = db.query(Student).filter(Student.student_id == log.student_id).first()
            if student:
                student_name = student.name
        
        result_logs.append({
            "id": log.id,
            "student_id": log.student_id,
            "student_name": student_name,
            "result": log.result,
            "confidence": log.confidence_score,
            "mental_state": log.mental_state,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None
        })
    
    return {
        "logs": result_logs,
        "total": len(result_logs)
    }

@app.get("/student/{student_id}")
async def get_student(student_id: str, db: Session = Depends(get_db)):
    """Get specific student information"""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    voiceprint = db.query(VoicePrint).filter(
        VoicePrint.student_id == student_id
    ).first()
    
    return {
        "student_id": student.student_id,
        "name": student.name,
        "has_voiceprint": voiceprint is not None,
        "created_at": student.created_at.isoformat() if student.created_at else None
    }

@app.get("/attendance")
async def get_attendance(date: str = None):
    """Get attendance records for a specific date"""
    try:
        summary = attendance_manager.get_attendance_summary(date)
        return {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching attendance: {str(e)}")

@app.get("/attendance/download")
async def download_attendance(month: str = None):
    """Download attendance Excel file for a month"""
    try:
        if month is None:
            month = datetime.now()
        
        attendance_file = attendance_manager.get_attendance_file_path(month)
        
        if not os.path.exists(attendance_file):
            raise HTTPException(status_code=404, detail="Attendance file not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            attendance_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=os.path.basename(attendance_file)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading attendance: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

