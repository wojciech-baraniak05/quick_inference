@echo off
rem Sets up the .venv virtual environment, installs requirements-dev.txt
rem (runtime packages plus the notebook tooling),
rem generates the synthetic datasets and downloads the real ones.
rem Usage: setup_windows.bat [--force-download]
setlocal

cd /d "%~dp0"
set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"

rem --- Find a Python interpreter (py launcher first, then python on PATH)
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.12+ from https://www.python.org/downloads/
    exit /b 1
)
%PY% -c "import sys; sys.exit(sys.version_info < (3, 12))"
if errorlevel 1 (
    echo [ERROR] Python 3.12+ is required ^(the pinned numpy and scipy need it^).
    exit /b 1
)

rem --- Create the virtual environment
if exist "%VPY%" (
    echo [1/4] Virtual environment %VENV% already exists.
) else (
    echo [1/4] Creating virtual environment in %VENV% ...
    %PY% -m venv "%VENV%" || goto :fail
)

rem --- Install requirements
echo [2/4] Installing requirements ...
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r requirements-dev.txt || goto :fail

if not exist "config.toml" (
    echo [ERROR] config.toml not found in %CD%.
    exit /b 1
)

rem --- Generate synthetic datasets
echo [3/4] Generating synthetic datasets ...
"%VPY%" src\data_generate\generate_datasets.py || goto :fail

rem --- Download real datasets (skipped when already present)
set "REAL=data\real_data"
if /i not "%~1"=="--force-download" (
    if exist "%REAL%\binary\susy.parquet" if exist "%REAL%\multiclass\shuttle.parquet" if exist "%REAL%\regression\gas_turbine.parquet" (
        echo [4/4] Real datasets already downloaded - skipping. Use --force-download to fetch them again.
        goto :done
    )
)
echo [4/4] Downloading real datasets ^(SUSY is large, this can take a while^) ...
"%VPY%" src\data_generate\data_download_real.py || goto :fail

:done
echo.
echo Setup complete. Activate the environment with:  %VENV%\Scripts\activate
exit /b 0

:fail
echo.
echo [ERROR] Setup failed ^(exit code %errorlevel%^).
exit /b 1
