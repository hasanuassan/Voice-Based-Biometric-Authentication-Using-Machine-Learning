"""
Attendance Management Module
Handles saving recordings and maintaining Excel attendance sheets
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
import config

class AttendanceManager:
    """Manage attendance records and daily recording folders"""
    
    def __init__(self):
        self.recordings_dir = config.RECORDINGS_DIR
        self.attendance_dir = config.ATTENDANCE_DIR
    
    def get_daily_folder(self, date=None):
        """
        Get or create daily folder for recordings
        Format: recordings/YYYY-MM-DD/
        """
        if date is None:
            date = datetime.now()
        
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")
        
        daily_folder = os.path.join(self.recordings_dir, date_str)
        os.makedirs(daily_folder, exist_ok=True)
        return daily_folder
    
    def save_recording(self, audio_bytes, student_id, recording_type="verification"):
        """
        Save recording file in daily folder
        recording_type: 'registration' or 'verification'
        Returns: path to saved file
        """
        # Get today's folder
        daily_folder = self.get_daily_folder()
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_id}_{recording_type}_{timestamp}.wav"
        filepath = os.path.join(daily_folder, filename)
        
        # Save file
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        
        return filepath
    
    def get_attendance_file_path(self, date=None):
        """
        Get attendance Excel file path for a specific date
        Format: attendance/Attendance_YYYY-MM.xlsx (monthly file)
        """
        if date is None:
            date = datetime.now()
        
        if isinstance(date, str):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        else:
            date_obj = date
        
        # Monthly attendance file
        month_str = date_obj.strftime("%Y-%m")
        filename = f"Attendance_{month_str}.xlsx"
        filepath = os.path.join(self.attendance_dir, filename)
        return filepath
    
    def add_attendance_entry(self, student_id, student_name, verified, confidence, 
                           mental_state=None, recording_path=None):
        """
        Add attendance entry to Excel sheet
        """
        today = datetime.now()
        date_str = today.strftime("%Y-%m-%d")
        time_str = today.strftime("%H:%M:%S")
        
        # Get attendance file path
        attendance_file = self.get_attendance_file_path(today)
        
        # Create new entry
        new_entry = {
            'Date': date_str,
            'Time': time_str,
            'Student ID': student_id,
            'Student Name': student_name,
            'Status': 'Present' if verified else 'Absent',
            'Verified': 'Yes' if verified else 'No',
            'Confidence Score': f"{confidence:.2%}" if confidence else "N/A",
            'Mental State': mental_state or "N/A",
            'Recording Path': recording_path or "N/A"
        }
        
        # Load existing data or create new DataFrame
        if os.path.exists(attendance_file):
            try:
                df = pd.read_excel(attendance_file)
            except:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        # Append new entry
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Sort by Date and Time
        df = df.sort_values(['Date', 'Time'], ascending=[False, False])
        
        # Save to Excel
        df.to_excel(attendance_file, index=False, engine='openpyxl')
        
        return attendance_file
    
    def get_attendance_summary(self, date=None):
        """
        Get attendance summary for a specific date
        """
        if date is None:
            date = datetime.now()
        
        attendance_file = self.get_attendance_file_path(date)
        
        if not os.path.exists(attendance_file):
            return None
        
        df = pd.read_excel(attendance_file)
        
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime("%Y-%m-%d")
        
        # Filter by date
        daily_attendance = df[df['Date'] == date_str]
        
        return {
            'total_entries': len(daily_attendance),
            'present': len(daily_attendance[daily_attendance['Status'] == 'Present']),
            'absent': len(daily_attendance[daily_attendance['Status'] == 'Absent']),
            'data': daily_attendance.to_dict('records')
        }
    
    def get_all_attendance(self, month=None):
        """
        Get all attendance records for a month
        """
        if month is None:
            month = datetime.now()
        
        attendance_file = self.get_attendance_file_path(month)
        
        if not os.path.exists(attendance_file):
            return pd.DataFrame()
        
        return pd.read_excel(attendance_file)

