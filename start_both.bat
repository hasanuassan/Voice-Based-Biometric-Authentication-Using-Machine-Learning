@echo off
echo ========================================
echo Voice Authentication System
echo ========================================
echo.
echo Starting Backend API Server...
start "Backend API" cmd /k "cd /d %~dp0 && python api.py"
timeout /t 3 /nobreak >nul
echo.
echo Starting Frontend UI...
start "Frontend UI" cmd /k "cd /d %~dp0 && streamlit run app.py"
echo.
echo ========================================
echo Both servers are starting!
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:8501
echo.
echo Press any key to exit this window...
echo (Servers will continue running in separate windows)
pause >nul

