@echo off
REM ============================================================
REM  Weekly model retraining for TennisBet
REM  Schedule via Windows Task Scheduler (e.g. every Sunday 03:00)
REM ============================================================

cd /d "%~dp0"

REM --- Activate venv ---
call "%~dp0.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate .venv
    pause
    exit /b 1
)
echo Using Python: & python --version
set PYTHONWARNINGS=ignore::UserWarning:fuzzywuzzy

REM --- Ensure dependencies are installed (check lightgbm as newest) ---
python -c "import lightgbm" 2>nul || (
    echo Installing dependencies...
    pip install -r requirements.txt || goto :error
)

REM --- Pull latest data & code ---
git pull --ff-only 2>nul
git lfs pull 2>nul

REM --- Step 1: Fetch fresh data from sources ---
echo [%date% %time%] Updating data sources...
python -m src.data_updater || echo WARNING: data_updater failed, continuing with existing data...
python -m src.tml_ingest || echo WARNING: tml_ingest failed, continuing...

REM --- Step 2: Build pipeline (features + elo) ---
echo [%date% %time%] Running data pipeline...
python -m src.data_pipeline --incremental || goto :error
python -m src.elo_engine --incremental || goto :error
python -m src.feature_engineering || goto :error

REM --- Step 3: Train all models with Optuna tuning ---
echo [%date% %time%] Training models (CatBoost + XGBoost + LightGBM + ensemble)...
python -m src.model_training --tours atp wta --tune || goto :error

REM --- Commit updated models and data ---
echo [%date% %time%] Committing results...
git add .gitattributes models/ data/meta/training_state.json data/meta/last_update.json data/meta/model_metrics.csv data/processed/ 2>nul
git diff --cached --quiet || git commit -m "retrain: weekly model update %date%"
git push

echo [%date% %time%] Done.
pause
exit /b 0

:error
echo [%date% %time%] ERROR: A step failed, aborting.
pause
exit /b 1
