"""
Streamlit Frontend Application
Professional UI for Voice-Based Biometric Authentication System
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import tempfile
import os
import pandas as pd
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Voice-Based Biometric Authentication",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .error-card {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None

def record_audio(duration=5):
    """Record audio using browser's microphone"""
    # This will be handled by the frontend JavaScript
    pass

def plot_waveform(audio_data, sr=16000):
    """Plot audio waveform"""
    if audio_data is None or len(audio_data) == 0:
        return None
    
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white"
    )
    return fig

def plot_confidence_meter(confidence):
    """Plot confidence score as circular gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score (%)"},
        delta={'reference': 75},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🎤 Voice-Based Biometric Authentication</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Voice Aging Adaptation & Cognitive State Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Voice Registration", "Voice Verification & Analysis", "View Students", "View Logs"]
    )
    
    # Voice Registration Page
    if page == "Voice Registration":
        st.header("📝 Voice Registration")
        st.markdown("Register your voice for biometric authentication. Record 5 seconds of clear speech.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Record Your Voice")
            
            # Audio recorder
            audio_file = st.audio_input("Record audio (5 seconds)")
            
            if audio_file is not None:
                # Display waveform
                audio_bytes = audio_file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                
                st.plotly_chart(plot_waveform(audio_data, sr), use_container_width=True)
                
                # Student information form
                with st.form("registration_form"):
                    student_id = st.text_input("Student ID / Roll Number *", placeholder="e.g., STU001")
                    student_name = st.text_input("Full Name *", placeholder="e.g., John Doe")
                    submit_button = st.form_submit_button("Register Voice", type="primary")
                    
                    if submit_button:
                        if not student_id or not student_name:
                            st.error("Please fill in all required fields")
                        else:
                            # Upload to API
                            with st.spinner("Processing voice registration..."):
                                try:
                                    files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
                                    data = {
                                        "student_id": student_id,
                                        "name": student_name
                                    }
                                    
                                    response = requests.post(
                                        f"{API_BASE_URL}/register",
                                        files=files,
                                        data=data,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success(f"✅ Voice registered successfully for {student_name}!")
                                        
                                        # Display extracted features
                                        st.subheader("Extracted Voice Features")
                                        features = result.get("features_extracted", {})
                                        
                                        col_a, col_b, col_c, col_d = st.columns(4)
                                        with col_a:
                                            st.metric("MFCC Coefficients", features.get("mfcc_coefficients", 13))
                                        with col_b:
                                            st.metric("Pitch (Hz)", f"{features.get('pitch_mean', 0):.2f}")
                                        with col_c:
                                            st.metric("Energy", f"{features.get('energy_mean', 0):.4f}")
                                        with col_d:
                                            st.metric("Speaking Rate", f"{features.get('speaking_rate', 0):.2f}")
                                    else:
                                        st.error(f"Registration failed: {response.text}")
                                
                                except requests.exceptions.RequestException as e:
                                    st.error(f"Connection error: {str(e)}")
                                    st.info("Make sure the API server is running on http://localhost:8000")
        
        with col2:
            st.subheader("Instructions")
            st.info("""
            **How to Register:**
            1. Click the microphone icon
            2. Allow browser microphone access
            3. Speak clearly for 5 seconds
            4. Enter your Student ID and Name
            5. Click "Register Voice"
            
            **Tips:**
            - Speak in a quiet environment
            - Use a consistent speaking style
            - Avoid background noise
            """)
    
    # Voice Verification Page
    elif page == "Voice Verification & Analysis":
        st.header("🔐 Voice Verification & Mental State Analysis")
        st.markdown("Verify your identity and analyze your mental state from voice patterns.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Record Verification Voice")
            st.info("⚠️ **Important:** Please record exactly 5 seconds of audio for best results")
            
            # Recording timer indicator
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center;">
                <p style="margin: 0; font-size: 1.2rem; color: #1f77b4;">
                    <strong>⏱️ Recording Timer: Aim for 5 seconds</strong>
                </p>
                <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">
                    Click the microphone icon below and speak for 5 seconds
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio recorder
            audio_file = st.audio_input("Record audio for verification (5 seconds)")
            
            if audio_file is not None:
                # Display waveform
                audio_bytes = audio_file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                
                # Calculate duration
                duration = len(audio_data) / sr
                
                # Display duration and warning if not 5 seconds
                duration_col1, duration_col2 = st.columns([1, 2])
                with duration_col1:
                    st.metric("Recording Duration", f"{duration:.2f} seconds")
                with duration_col2:
                    if abs(duration - 5.0) > 0.5:  # Allow 0.5 second tolerance
                        st.warning(f"⚠️ Recording is {duration:.2f} seconds. Recommended: 5 seconds for best accuracy.")
                    else:
                        st.success("✅ Recording duration is optimal (5 seconds)")
                
                st.plotly_chart(plot_waveform(audio_data, sr), use_container_width=True)
                
                # Verify button
                if st.button("🔍 Verify Voice & Analyze Mental State", type="primary", use_container_width=True):
                    with st.spinner("Processing verification and mental state analysis..."):
                        try:
                            files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
                            
                            response = requests.post(
                                f"{API_BASE_URL}/verify",
                                files=files,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Display results
                                st.markdown("---")
                                st.subheader("🔍 Authentication Results")
                                
                                # Authentication result
                                if result.get("verified", False):
                                    student_name = result.get('student_name', 'User')
                                    
                                    st.markdown('<div class="result-card success-card">', unsafe_allow_html=True)
                                    st.success(f"✅ **VERIFIED** - Welcome, {student_name}!")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Voice welcome message
                                    welcome_message = f"Welcome to {student_name}"
                                    
                                    # Display welcome message
                                    st.markdown(f"""
                                    <div style="background-color: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0;">
                                        <h4 style="margin: 0; color: #155724;">🔊 Voice Welcome Message</h4>
                                        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #155724;">
                                            <strong>{welcome_message}</strong>
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Use Streamlit components to execute JavaScript for voice welcome
                                    import streamlit.components.v1 as components
                                    
                                    # JavaScript code to play welcome message
                                    welcome_js = f"""
                                    <script>
                                    (function() {{
                                        function playWelcome() {{
                                            if ('speechSynthesis' in window) {{
                                                // Cancel any ongoing speech
                                                window.speechSynthesis.cancel();
                                                
                                                // Small delay to ensure page is ready
                                                setTimeout(function() {{
                                                    const utterance = new SpeechSynthesisUtterance('{welcome_message}');
                                                    utterance.lang = 'en-US';
                                                    utterance.rate = 0.9;
                                                    utterance.pitch = 1.0;
                                                    utterance.volume = 1.0;
                                                    
                                                    // Event handlers
                                                    utterance.onstart = function() {{
                                                        console.log('Welcome message started');
                                                    }};
                                                    
                                                    utterance.onerror = function(event) {{
                                                        console.log('Speech error:', event);
                                                    }};
                                                    
                                                    // Play the message
                                                    window.speechSynthesis.speak(utterance);
                                                }}, 300);
                                            }} else {{
                                                console.log('Speech synthesis not supported');
                                            }}
                                        }}
                                        
                                        // Execute when script loads
                                        if (document.readyState === 'loading') {{
                                            document.addEventListener('DOMContentLoaded', playWelcome);
                                        }} else {{
                                            playWelcome();
                                        }}
                                    }})();
                                    </script>
                                    """
                                    
                                    # Render the JavaScript using components
                                    components.html(welcome_js, height=0)
                                    
                                    # Student info
                                    col_info1, col_info2 = st.columns(2)
                                    with col_info1:
                                        st.metric("Student ID", result.get("student_id", "N/A"))
                                    with col_info2:
                                        st.metric("Student Name", student_name)
                                else:
                                    st.markdown('<div class="result-card error-card">', unsafe_allow_html=True)
                                    st.error("❌ **NOT VERIFIED** - Voice does not match any registered user")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Confidence score
                                confidence = result.get("confidence", 0.0)
                                st.subheader("📊 Confidence Score")
                                st.plotly_chart(plot_confidence_meter(confidence), use_container_width=True)
                                
                                # Mental state analysis
                                st.markdown("---")
                                st.subheader("🧠 Mental State Analysis")
                                
                                mental_state = result.get("mental_state", "Unknown")
                                mental_confidence = result.get("mental_confidence", 0.0)
                                explanation = result.get("explanation", "")
                                
                                # Color coding for mental state
                                state_colors = {
                                    "Calm": "#28a745",
                                    "Stressed": "#ffc107",
                                    "Anxious": "#fd7e14",
                                    "Fatigued": "#dc3545"
                                }
                                
                                color = state_colors.get(mental_state, "#666")
                                
                                # Mental state card
                                st.markdown(f"""
                                <div style="background-color: {color}20; padding: 1.5rem; border-radius: 10px; 
                                            border-left: 4px solid {color}; margin: 1rem 0;">
                                    <h3 style="color: {color}; margin-top: 0;">{mental_state}</h3>
                                    <p><strong>Confidence:</strong> {mental_confidence*100:.1f}%</p>
                                    <p>{explanation}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Mental state visualization
                                states = ["Calm", "Stressed", "Anxious", "Fatigued"]
                                values = [0.0] * 4
                                if mental_state in states:
                                    idx = states.index(mental_state)
                                    values[idx] = mental_confidence
                                
                                fig = px.bar(
                                    x=states,
                                    y=values,
                                    labels={"x": "Mental State", "y": "Confidence"},
                                    color=values,
                                    color_continuous_scale="RdYlGn",
                                    title="Mental State Distribution"
                                )
                                fig.update_layout(height=300, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:
                                st.error(f"Verification failed: {response.text}")
                        
                        except requests.exceptions.RequestException as e:
                            st.error(f"Connection error: {str(e)}")
                            st.info("Make sure the API server is running on http://localhost:8000")
        
        with col2:
            st.subheader("Analysis Features")
            st.info("""
            **What We Analyze:**
            
            **Authentication:**
            - Voice biometric matching
            - Confidence scoring
            - Voice aging adaptation
            
            **Mental State:**
            - Calm detection
            - Stress indicators
            - Anxiety patterns
            - Fatigue analysis
            
            **Features Used:**
            - Pitch variation
            - Energy patterns
            - Speaking rate
            - Pause frequency
            """)
    
    # View Students Page
    elif page == "View Students":
        st.header("👥 Registered Students")
        
        if st.button("🔄 Refresh List"):
            st.rerun()
        
        try:
            response = requests.get(f"{API_BASE_URL}/students", timeout=10)
            if response.status_code == 200:
                data = response.json()
                students = data.get("students", [])
                
                if students:
                    st.success(f"Found {len(students)} registered student(s)")
                    
                    for student in students:
                        with st.expander(f"📋 {student['name']} - {student['student_id']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Student ID:** {student['student_id']}")
                                st.write(f"**Name:** {student['name']}")
                            with col2:
                                if student.get('created_at'):
                                    st.write(f"**Registered:** {student['created_at'][:10]}")
                else:
                    st.info("No students registered yet.")
            else:
                st.error("Failed to fetch students")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
    
    # View Logs Page
    elif page == "View Logs":
        st.header("📊 Authentication Logs")
        
        # Create two columns for filters and chatbot
        col_filter, col_chat = st.columns([2, 1])
        
        with col_filter:
            # Filters section
            st.subheader("🔍 Filters")
            
            # Date filter
            filter_date = st.date_input(
                "Filter by Date",
                value=None,
                help="Select a date to filter logs, or leave empty for all dates"
            )
            
            # Time range filters
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                start_time = st.time_input("Start Time", value=None)
            with col_time2:
                end_time = st.time_input("End Time", value=None)
            
            # Search bar for name
            search_name = st.text_input(
                "🔎 Search by Name or Student ID",
                placeholder="Enter student name or ID...",
                help="Search logs by student name or student ID"
            )
            
            # Limit slider
            limit = st.slider("Number of logs to display", 10, 200, 50)
            
            # Apply filters button
            if st.button("🔍 Apply Filters", type="primary"):
                st.rerun()
        
        with col_chat:
            # Chatbot section
            st.subheader("💬 Enquiry Chatbot")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, (role, message) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Bot:** {message}")
                    st.markdown("---")
            
            # Chat input
            user_query = st.text_input(
                "Ask a question about the logs:",
                placeholder="e.g., How many students verified today?",
                key="chat_input"
            )
            
            if st.button("Send", key="send_chat"):
                if user_query:
                    # Add user message to history
                    st.session_state.chat_history.append(("user", user_query))
                    
                    # Generate bot response
                    bot_response = generate_chatbot_response(user_query, API_BASE_URL)
                    st.session_state.chat_history.append(("bot", bot_response))
                    st.rerun()
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Fetch and display logs
        try:
            # Build query parameters
            params = {"limit": limit}
            if filter_date:
                params["date"] = filter_date.strftime("%Y-%m-%d")
            if start_time:
                params["start_time"] = start_time.strftime("%H:%M")
            if end_time:
                params["end_time"] = end_time.strftime("%H:%M")
            if search_name:
                params["search_name"] = search_name
            
            response = requests.get(f"{API_BASE_URL}/logs", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logs = data.get("logs", [])
                total = data.get("total", len(logs))
                
                if logs:
                    st.success(f"Displaying {len(logs)} authentication attempt(s)")
                    
                    # Create a DataFrame for better display
                    log_data = []
                    for log in logs:
                        timestamp = log.get('timestamp', '')
                        if timestamp:
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                date_str = dt.strftime("%Y-%m-%d")
                                time_str = dt.strftime("%H:%M:%S")
                            except:
                                date_str = timestamp[:10] if len(timestamp) >= 10 else "N/A"
                                time_str = timestamp[11:19] if len(timestamp) >= 19 else "N/A"
                        else:
                            date_str = "N/A"
                            time_str = "N/A"
                        
                        log_data.append({
                            "Status": "✅ Verified" if log['result'] == "Verified" else "❌ Not Verified",
                            "Date": date_str,
                            "Time": time_str,
                            "Student ID": log.get('student_id', 'N/A'),
                            "Student Name": log.get('student_name', 'N/A'),
                            "Confidence": f"{log['confidence']:.2%}",
                            "Mental State": log.get('mental_state', 'N/A')
                        })
                    
                    df = pd.DataFrame(log_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No authentication logs found matching your filters.")
            else:
                st.error("Failed to fetch logs")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            st.info("Make sure the API server is running on http://localhost:8000")

def generate_chatbot_response(query, api_base_url):
    """Generate chatbot response based on user query"""
    query_lower = query.lower()
    
    # Try to fetch logs for analysis
    try:
        response = requests.get(f"{api_base_url}/logs?limit=100", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logs = data.get("logs", [])
            
            # Count verified vs not verified
            verified_count = sum(1 for log in logs if log.get('result') == 'Verified')
            not_verified_count = len(logs) - verified_count
            
            # Analyze mental states
            mental_states = {}
            for log in logs:
                state = log.get('mental_state')
                if state:
                    mental_states[state] = mental_states.get(state, 0) + 1
            
            # Answer questions
            if "how many" in query_lower and "verified" in query_lower:
                return f"Based on recent logs, {verified_count} student(s) were verified successfully."
            
            elif "how many" in query_lower and ("not verified" in query_lower or "failed" in query_lower):
                return f"Based on recent logs, {not_verified_count} verification attempt(s) failed."
            
            elif "mental state" in query_lower or "state" in query_lower:
                if mental_states:
                    states_str = ", ".join([f"{k}: {v}" for k, v in mental_states.items()])
                    return f"Mental state distribution: {states_str}"
                else:
                    return "No mental state data available in recent logs."
            
            elif "today" in query_lower:
                today = datetime.now().strftime("%Y-%m-%d")
                today_logs = [log for log in logs if log.get('timestamp', '').startswith(today)]
                return f"Today ({today}), there were {len(today_logs)} authentication attempt(s)."
            
            elif "help" in query_lower or "what" in query_lower:
                return """I can help you with:
- Number of verified/not verified students
- Mental state statistics
- Today's authentication attempts
- General information about the system

Try asking: How many students verified today? or What are the mental states?"""
            
            else:
                return f"I found {len(logs)} recent log entries. You can filter by date, time, or search by name to get more specific information."
        
    except:
        pass
    
    # Default responses
    if "hello" in query_lower or "hi" in query_lower:
        return "Hello! I'm here to help you with authentication logs. Ask me about verification statistics, mental states, or today's activity."
    
    elif "help" in query_lower:
        return """I can help you with:
- Number of verified/not verified students
- Mental state statistics  
- Today's authentication attempts
- General information about the system"""
    
    else:
        return "I'm here to help! Try asking about verification statistics, mental states, or use the filters above to search specific logs."

if __name__ == "__main__":
    main()

