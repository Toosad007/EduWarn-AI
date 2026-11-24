@echo off
cd /d "%~dp0"
echo ========================================================
echo       DANG KHOI CHAY UNG DUNG EDUWARN AI (FLASK)
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [CHU Y] Chua tim thay moi truong ao .venv!
    echo Dang tu dong chay cai dat...
    call cai_dat.bat
    if errorlevel 1 (
        echo [LOI] Cai dat khong thanh cong. Khong the tiep tuc.
        pause
        exit /b 1
    )
)

where ollama >nul 2>&1
if %errorlevel%==0 (
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo Dang khoi dong Local AI ^(Ollama^)...
        start "" /min ollama serve
        timeout /t 5 /nobreak >nul
    ) else (
        echo Local AI ^(Ollama^) dang san sang.
    )
) else (
    echo [CHU Y] Khong tim thay Ollama - Chatbot se chay che do tra cuu du phong.
)

echo Dang mo trinh duyet tai: http://localhost:5000 ...
start http://localhost:5000

echo.
echo May chu Flask dang hoat dong... (Nhan Ctrl+C de dung)
echo.
".venv\Scripts\python.exe" flask_app.py
pause
