@echo off
title Antigravity RAG & STT Launcher

echo ====================================================================
echo             Starting Antigravity RAG Agent & Local STT Server
echo ====================================================================
echo.

:: Verify virtual environment exists
if not exist "..\venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment python interpreter not found at:
    echo         ..\venv\Scripts\python.exe
    echo         Please verify that the virtual environment is installed in the root folder.
    echo.
    pause
    exit /b 1
)

echo [1/2] Launching Local Offline STT Server (Piper) on port 8001...
:: Starts the STT server in a separate command window so you can see its logs
start "Offline STT Server (Piper)" cmd /c "title Offline STT Server && ..\venv\Scripts\python.exe stt_server.py"

:: Small delay to let the STT server start loading the model
timeout /t 2 /nobreak > nul

echo [2/2] Launching Main RAG Agent Application on port 8000...
echo.
echo Open http://127.0.0.1:8000 in your browser to interact with the RAG agent.
echo Press Ctrl+C in this window to stop the main RAG server.
echo.

..\venv\Scripts\python.exe app.py

echo.
echo Main application stopped. You can close the STT server window manually.
pause
