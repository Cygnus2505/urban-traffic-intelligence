@echo off
echo ==========================================
echo   Urban Traffic Intelligence Platform
echo ==========================================
echo Starting Backend API on port 8001...
start /B .\venv\Scripts\python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
timeout /t 5
echo Starting Dashboard on port 8502...
start /B .\venv\Scripts\python -m streamlit run dashboard/app.py --server.port 8502 --server.headless true
echo ==========================================
echo   Services are running!
echo   API: http://localhost:8001
echo   Dashboard: http://localhost:8502
echo ==========================================
pause
