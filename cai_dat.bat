@echo off
cd /d "%~dp0"
echo ========================================================
echo        CAI DAT DU AN EDUWARN AI (STUDENT DROPOUT)
echo ========================================================
echo.

echo [1/2] Dang kiem tra va khoi tao moi truong ao (.venv)...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo [LOI] Khong the tao moi truong ao.
        echo Vui long kiem tra Python da duoc cai dat va them vao PATH chua.
        pause
        exit /b 1
    )
    echo -> Da tao moi truong ao .venv thanh cong!
) else (
    echo -> Moi truong ao .venv da ton tai.
)

echo.
echo [2/2] Dang cai dat cac thu vien can thiet tu requirements.txt...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [CANH BAO] Co the mot so goi gap loi mang hoac chua hoan tat.
    echo Ban hay kiem tra lai ket noi Internet.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo [THANH CONG] Cai dat hoan tat!
echo Ban co the mo ung dung bang file: chay_du_an.bat
echo ========================================================
pause
