@echo off
REM ============================================================
REM  Refresh deploy runtime outputs and push them to GitHub
REM ============================================================

cd /d "%~dp0"

call "%~dp0.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate .venv
    exit /b 1
)

echo Using Python:
python --version
set PYTHONWARNINGS=ignore::UserWarning:fuzzywuzzy

python -c "import lightgbm" 2>nul || (
    echo Installing dependencies...
    pip install -r requirements.txt || goto :error
)

git pull --ff-only 2>nul

for /f %%i in ('powershell -NoProfile -Command "(Get-Date).Year"') do set CURRENT_YEAR=%%i

echo [%date% %time%] Updating raw sources...
python -m src.data_updater || echo WARNING: data_updater failed, continuing...
python -m src.tml_ingest --years %CURRENT_YEAR% || echo WARNING: tml_ingest failed, continuing...
python -c "from src.data_updater import scrape_flashscore_results; import json; print(json.dumps(scrape_flashscore_results(), indent=2, default=str))" || echo WARNING: Flashscore results scrape failed, continuing...
python -m src.wta_backfill --years %CURRENT_YEAR% || echo WARNING: wta_backfill failed, continuing...

echo [%date% %time%] Rebuilding runtime data...
python -m src.data_pipeline || goto :error
python -m src.elo_engine || goto :error
python -m src.feature_engineering || goto :error

echo [%date% %time%] Refreshing current odds...
python -m src.odds_scraper || echo WARNING: odds_scraper failed, continuing with current odds snapshot...

echo [%date% %time%] Building predictions...
powershell -NoProfile -Command "@'
from src.predictor import predict_from_feature_file, predict_from_odds
from config import PROCESSED_DIR

for tour in ('atp', 'wta'):
    output_path = PROCESSED_DIR / f'{tour}_predictions.csv'
    pred = predict_from_odds(tour=tour, output_path=output_path)
    if pred.empty:
        input_path = PROCESSED_DIR / f'{tour}_player_features.csv'
        pred = predict_from_feature_file(tour=tour, input_path=input_path, output_path=output_path, limit=2000)
    print({'tour': tour, 'rows': int(len(pred)), 'output': str(output_path)})
'@ | python -" || goto :error

echo [%date% %time%] Removing deploy-excluded noise from staging...
git add data/odds/upcoming_odds.csv data/processed/atp_predictions.csv data/processed/wta_predictions.csv data/meta/last_update.json 2>nul
git diff --cached --quiet && (
    echo No runtime output changes to commit.
    exit /b 0
)

git commit -m "data: refresh published predictions and odds"
git push origin main || goto :error

echo [%date% %time%] Done.
exit /b 0

:error
echo [%date% %time%] ERROR: Runtime publish failed.
exit /b 1
