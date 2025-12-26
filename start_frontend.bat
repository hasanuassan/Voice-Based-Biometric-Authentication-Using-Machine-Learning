@echo off
echo Starting Voice Authentication Frontend...
echo.
cd /d %~dp0
streamlit run app.py
pause

