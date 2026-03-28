@echo off
REM ============================================================
REM  Weekly local model retraining for TennisBet
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

REM --- Step 1: Fetch fresh data from all sources ---
echo [%date% %time%] Updating Sackmann repos...
python -m src.data_updater || echo WARNING: data_updater failed, continuing with existing data...
for /f %%i in ('powershell -NoProfile -Command "(Get-Date).Year"') do set CURRENT_YEAR=%%i
echo [%date% %time%] Ingesting ATP from TML-Database...
python -m src.tml_ingest --years %CURRENT_YEAR% || echo WARNING: tml_ingest failed, continuing...
echo [%date% %time%] Scraping Flashscore results...
python -c "from src.data_updater import scrape_flashscore_results; import json; print(json.dumps(scrape_flashscore_results(), indent=2, default=str))" || echo WARNING: Flashscore scrape failed, continuing...
echo [%date% %time%] Backfilling WTA from Tennis Explorer...
python -m src.wta_backfill --years %CURRENT_YEAR% || echo WARNING: wta_backfill failed, continuing...
echo [%date% %time%] Scraping today's odds...
python -m src.odds_scraper || echo WARNING: odds_scraper failed, continuing...

REM --- Step 2: Build pipeline (features + elo) ---
echo [%date% %time%] Running data pipeline...
python -m src.data_pipeline --incremental || goto :error
python -m src.elo_engine --incremental || goto :error
python -m src.feature_engineering || goto :error

REM --- Step 3: Train all models with Optuna tuning ---
echo [%date% %time%] Training models (CatBoost + XGBoost + LightGBM + ensemble)...
python -m src.model_training --tours atp wta --tune || goto :error

echo [%date% %time%] Local retrain completed.
echo Models and processed data were refreshed in-place under models/ and data/processed/.
pause
exit /b 0

:error
echo [%date% %time%] ERROR: A step failed, aborting.
pause
exit /b 1
