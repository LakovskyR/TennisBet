import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path if running as script
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import run_pipeline
from src.data_updater import update_data_sources
from src.elo_engine import run_elo
from src.model_training import train_models

log = logging.getLogger("retrain_cli")

def run_retrain_flow(tours: tuple[str, ...], force: bool = True) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    log.info("Starting retrain flow for tours: %s", tours)

    log.info("Step 1/4: Updating data sources...")
    try:
        update_data_sources()
    except Exception as exc:
        log.warning("Data update failed (continuing with existing data): %s", exc)

    log.info("Step 2/4: Running pipeline (incremental)...")
    try:
        run_pipeline(incremental=True)
    except Exception as exc:
        log.warning("Pipeline failed (continuing with existing processed data): %s", exc)

    log.info("Step 3/4: Updating ELO ratings (incremental)...")
    try:
        run_elo(incremental=True)
    except Exception as exc:
        log.warning("ELO update failed (continuing with existing ratings): %s", exc)

    log.info("Step 4/4: Training models (force=%s)...", force)
    report = train_models(tours=tours)

    log.info("Retrain flow completed.")
    for tour, rep in report.items():
        metrics = rep.get("metrics", {}).get("ensemble", {})
        if metrics:
            log.info("%s: Accuracy=%.2f%%, LogLoss=%.4f",
                     tour.upper(),
                     metrics.get("accuracy", 0) * 100,
                     metrics.get("log_loss", 0))

def main() -> None:
    parser = argparse.ArgumentParser(description="Force a full data update and model retrain")
    parser.add_argument("--tours", default="atp,wta", help="Comma-separated tours, default atp,wta")
    args = parser.parse_args()
    
    tours = tuple(t.strip() for t in args.tours.split(",") if t.strip())
    run_retrain_flow(tours=tours)

if __name__ == "__main__":
    main()
