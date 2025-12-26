# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Troubleshooting:**
- If `pyaudio` fails, try: `pip install pipwin && pipwin install pyaudio`
- On Linux: `sudo apt-get install portaudio19-dev`
- On Mac: `brew install portaudio`

### Step 2: Start Backend Server

Open Terminal 1:

```bash
python api.py
```

Or:

```bash
uvicorn api:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Start Frontend

Open Terminal 2:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

### Step 4: Register Your Voice

1. Go to **"Voice Registration"** page
2. Click microphone icon
3. Allow browser microphone access
4. Speak for 5 seconds
5. Enter Student ID and Name
6. Click **"Register Voice"**

### Step 5: Verify Your Voice

1. Go to **"Voice Verification & Analysis"** page
2. Click microphone icon
3. Speak for 5 seconds
4. Click **"Verify Voice & Analyze Mental State"**
5. View results!

---

## 📋 Testing Checklist

- [ ] Backend API running on port 8000
- [ ] Frontend UI accessible on port 8501
- [ ] Microphone access granted
- [ ] Voice registration successful
- [ ] Voice verification working
- [ ] Mental state analysis displayed
- [ ] Students list shows registered users
- [ ] Logs page shows authentication history

---

## 🐛 Common Issues

### Issue: "Connection refused" error

**Solution:** Make sure backend is running:
```bash
python api.py
```

### Issue: Microphone not working

**Solution:** 
- Check browser permissions
- Use Chrome or Firefox (best compatibility)
- Try refreshing the page

### Issue: Audio format error

**Solution:**
- Ensure recording is at least 5 seconds
- Use WAV format (automatic in browser)
- Check audio file is not corrupted

### Issue: Model not found

**Solution:**
- Models are created automatically on first use
- If needed, run: `python train_models.py`

---

## 📊 Expected Results

### Registration
- ✅ Success message
- ✅ Extracted features displayed
- ✅ Student added to database

### Verification
- ✅ Authentication status (Verified/Not Verified)
- ✅ Confidence score (0-100%)
- ✅ Mental state (Calm/Stressed/Anxious/Fatigued)
- ✅ Explanation text

---

## 🎓 For Viva/Examination

### Key Points to Demonstrate

1. **Registration Flow**
   - Show feature extraction
   - Explain MFCC, Pitch, Energy, Speaking Rate

2. **Verification Flow**
   - Show confidence scoring
   - Demonstrate voice aging adaptation
   - Explain similarity calculation

3. **Mental State Analysis**
   - Show different states
   - Explain feature engineering
   - Discuss real-world applications

4. **System Architecture**
   - Frontend (Streamlit)
   - Backend (FastAPI)
   - Database (SQLite)
   - ML Models (SVM, Random Forest)

### Demo Script

1. Register 2-3 different students
2. Verify each student's voice
3. Show mental state variations
4. View authentication logs
5. Explain voice aging adaptation

---

## 📚 Next Steps

1. Read `README.md` for detailed documentation
2. Check `ARCHITECTURE.md` for system design
3. Explore `config.py` for customization
4. Review code comments for implementation details

---

**Happy Coding! 🎤**

