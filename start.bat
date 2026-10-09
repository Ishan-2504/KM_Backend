@echo off
setlocal enabledelayedexpansion
title KM Search Launcher

:: ── Colors / header ───────────────────────────────────────────
echo ============================================================
echo   KM Search - One-Click Setup and Launch
echo ============================================================
echo.

:: ── Resolve absolute project paths (handles spaces in path) ──
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BACKEND=%ROOT%\backend"
set "FRONTEND=%ROOT%\frontend"
set "DATA=%ROOT%\data"
set "VENV=%BACKEND%\venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "VENV_ACT=%VENV%\Scripts\activate.bat"

echo Project root: %ROOT%
echo.

:: ── Sanity check on project layout ───────────────────────────
if not exist "%BACKEND%\requirements.txt" (
    echo ERROR: backend\requirements.txt not found.
    echo        Make sure you are running start.bat from the project root.
    goto fail
)
if not exist "%FRONTEND%\package.json" (
    echo ERROR: frontend\package.json not found.
    echo        Make sure you are running start.bat from the project root.
    goto fail
)

:: ── [1/7] Detect Python 3.10+ ────────────────────────────────
echo [1/7] Detecting Python 3.10+...
set "PY_CMD="

where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
        if !errorlevel! equ 0 set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo ERROR: Python 3.10 or newer was not found on PATH.
    echo        Install from https://www.python.org/downloads/
    echo        During install, tick "Add python.exe to PATH".
    goto fail
)
for /f "tokens=*" %%v in ('%PY_CMD% --version 2^>^&1') do echo        %%v detected (%PY_CMD%)
echo.

:: ── [2/7] Detect Node.js 18+ and npm ─────────────────────────
echo [2/7] Detecting Node.js 18+...
where node >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Node.js is not installed or not on PATH.
    echo        Install the LTS release (18 or newer) from https://nodejs.org/
    goto fail
)

set "NODE_VER="
for /f "tokens=*" %%v in ('node --version') do set "NODE_VER=%%v"
set "NODE_VER_STRIPPED=!NODE_VER:v=!"
for /f "tokens=1 delims=." %%a in ("!NODE_VER_STRIPPED!") do set "NODE_MAJOR=%%a"
if "!NODE_MAJOR!"=="" set "NODE_MAJOR=0"
if !NODE_MAJOR! lss 18 (
    echo ERROR: Node.js !NODE_VER! detected. Version 18 or newer is required.
    echo        Upgrade from https://nodejs.org/
    goto fail
)
echo        Node !NODE_VER! detected

where npm >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: npm was not found on PATH. Reinstall Node.js from https://nodejs.org/
    goto fail
)
echo.

:: ── [3/7] Backend virtual environment ────────────────────────
echo [3/7] Preparing backend virtual environment...
if not exist "%VENV_PY%" (
    echo        Creating virtual environment at backend\venv ...
    %PY_CMD% -m venv "%VENV%"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo        Try deleting backend\venv and rerun start.bat.
        goto fail
    )
)

if not exist "%VENV_PY%" (
    echo ERROR: backend\venv is present but invalid.
    echo        Delete backend\venv and rerun start.bat.
    goto fail
)

echo        Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if !errorlevel! neq 0 (
    echo        WARNING: pip upgrade failed, continuing with current version.
)

echo        Installing backend dependencies (first run may take a minute) ...
"%VENV_PY%" -m pip install -r "%BACKEND%\requirements.txt" --quiet --disable-pip-version-check
if !errorlevel! neq 0 (
    echo        Retrying backend dependency install with verbose output ...
    "%VENV_PY%" -m pip install -r "%BACKEND%\requirements.txt" --disable-pip-version-check
    if !errorlevel! neq 0 (
        echo ERROR: Failed to install backend dependencies.
        echo        Try deleting backend\venv and rerun start.bat.
        goto fail
    )
)
echo        Backend dependencies ready.
echo.

:: ── [4/7] Backend .env ───────────────────────────────────────
echo [4/7] Checking backend configuration...
if not exist "%BACKEND%\.env" (
    echo        backend\.env not found - creating a template.
    > "%BACKEND%\.env" (
        echo AZURE_OPENAI_API_KEY=
        echo AZURE_OPENAI_ENDPOINT=
        echo AZURE_OPENAI_API_VERSION=2024-02-15-preview
        echo AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=
        echo AZURE_OPENAI_EMBEDDINGS_API_VERSION=2024-02-15-preview
        echo AZURE_OPENAI_CHAT_DEPLOYMENT=
        echo CHROMA_PERSIST_DIR=./app/db/chroma_data
        echo DATA_FILE_PATH=../data/km_final_output.xlsx
    )
    echo.
    echo ============================================================
    echo   ACTION REQUIRED
    echo   A template backend\.env was created.
    echo   Fill in your Azure OpenAI credentials, then rerun start.bat.
    echo   File: %BACKEND%\.env
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo        backend\.env found.
echo.

:: ── [5/7] Frontend dependencies and .env ─────────────────────
echo [5/7] Preparing frontend dependencies...

set "NEED_NPM_INSTALL=0"
if not exist "%FRONTEND%\node_modules" set "NEED_NPM_INSTALL=1"
if not exist "%FRONTEND%\node_modules\.package-lock.json" set "NEED_NPM_INSTALL=1"

if "!NEED_NPM_INSTALL!"=="1" (
    echo        Running npm install (first run may take a few minutes) ...
    pushd "%FRONTEND%" >nul
    call npm install --no-fund --no-audit --loglevel=error
    set "NPM_RC=!errorlevel!"
    popd >nul
    if !NPM_RC! neq 0 (
        echo ERROR: npm install failed.
        echo        Try deleting frontend\node_modules and frontend\package-lock.json,
        echo        then rerun start.bat.
        goto fail
    )
) else (
    echo        Frontend dependencies already installed.
)

if not exist "%FRONTEND%\.env" (
    echo        Creating frontend\.env with default API URL ...
    > "%FRONTEND%\.env" echo REACT_APP_API_URL=http://localhost:8000
)
echo        Frontend ready.
echo.

:: ── [6/7] Data file check (non-fatal) ────────────────────────
echo [6/7] Checking data file...
if not exist "%DATA%\km_final_output.xlsx" (
    echo        WARNING: data\km_final_output.xlsx is missing.
    echo                 The backend will start, but ingestion will fail
    echo                 until the file is placed at:
    echo                 %DATA%\km_final_output.xlsx
) else (
    echo        Data file found.
)
echo.

:: ── [7/7] Launch backend and frontend ────────────────────────
echo [7/7] Launching servers...

pushd "%BACKEND%" >nul
start "KM-Backend" cmd /k "call "%VENV_ACT%" && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
popd >nul

echo        Waiting for backend to become ready on http://localhost:8000 ...
set /a WAIT_MAX=45
set /a WAIT_I=0
set "HAVE_CURL=0"
where curl.exe >nul 2>&1
if !errorlevel! equ 0 set "HAVE_CURL=1"

:wait_backend
if "!HAVE_CURL!"=="1" (
    curl.exe -fsS -o NUL --max-time 1 http://localhost:8000/api/health >nul 2>&1
) else (
    powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8000/api/health' -TimeoutSec 1; if ($r.StatusCode -ne 200) { exit 1 } } catch { exit 1 }" >nul 2>&1
)
if !errorlevel! equ 0 goto backend_up
set /a WAIT_I+=1
if !WAIT_I! geq !WAIT_MAX! goto backend_timeout
>nul timeout /t 1 /nobreak
goto wait_backend

:backend_timeout
echo        Backend did not report ready after !WAIT_MAX! seconds.
echo        Check the KM-Backend window for errors. Continuing.
goto start_frontend

:backend_up
echo        Backend is up.

:start_frontend
pushd "%FRONTEND%" >nul
start "KM-Frontend" cmd /k "npm start"
popd >nul

echo        Opening browser at http://localhost:3000 ...
>nul timeout /t 3 /nobreak
start "" "http://localhost:3000"

echo.
echo ============================================================
echo   KM Search is running.
echo   Frontend : http://localhost:3000
echo   Backend  : http://localhost:8000
echo   API docs : http://localhost:8000/docs
echo.
echo   Close the KM-Backend and KM-Frontend windows to stop.
echo ============================================================
echo.
echo Press any key to close this launcher window (servers keep running) ...
pause >nul
endlocal
exit /b 0

:fail
echo.
echo ============================================================
echo   Setup did not complete. Please fix the error above and
echo   run start.bat again.
echo ============================================================
echo.
pause
endlocal
exit /b 1
