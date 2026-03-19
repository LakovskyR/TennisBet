# CODEX TASK: Retrain Models with Updated WTA Data

## CONTEXT

WTA data has been updated (tournament slugs fixed, more tournaments scraped).
Models need retraining on the new clean data.

### Current Baseline (clean deduplicated data)
- **ATP**: accuracy 64.59%, log-loss 0.620, ECE 0.023, 2,982 test rows
- **WTA**: accuracy 68.60%, log-loss 0.591, ECE 0.045, 1,796 test rows

### Key Finding: WTA Ensemble Weights Are Suboptimal
Weight sweep on clean data shows XGBoost alone (69.38%) beats the 0.6/0.4 CatBoost/XGBoost
ensemble (68.60%). Full sweep results:
```
CB=0.0 XGB=1.0: acc=0.6938 ll=0.5857  <-- best
CB=0.1 XGB=0.9: acc=0.6915 ll=0.5863
CB=0.2 XGB=0.8: acc=0.6893 ll=0.5869
CB=0.3 XGB=0.7: acc=0.6849 ll=0.5877
CB=0.4 XGB=0.6: acc=0.6865 ll=0.5885
CB=0.5 XGB=0.5: acc=0.6865 ll=0.5894
CB=0.6 XGB=0.4: acc=0.6860 ll=0.5905  <-- current
```

### DEPENDENCY
**Run this AFTER the WTA backfill task completes** (see `deepseek_wta_backfill_prompt.md`).
If WTA backfill hasn't added new data yet, skip to Step 3 (weight optimization).

## ACTUAL API SIGNATURES (verified)

```python
from src.data_pipeline import run_pipeline
result = run_pipeline(incremental=False)

from src.elo_engine import compute_elo_for_tour
result = compute_elo_for_tour("atp", incremental=False)
result = compute_elo_for_tour("wta", incremental=False)

from src.feature_engineering import build_features
result = build_features(("atp", "wta"))

from src.model_training import train_models
result = train_models(tours=("atp",), use_optuna=False)
result = train_models(tours=("wta",), use_optuna=False)
result = train_models(tours=("atp", "wta"), use_optuna=True, optuna_trials=50)

from src.backtest import run_backtest
result = run_backtest(tours=("atp", "wta"))
```

## TASK

### Step 1: Rebuild Full Pipeline
```bash
python -c "
import json
from src.data_pipeline import run_pipeline
result = run_pipeline(incremental=False)
for tour, info in result.items():
    print(f'{tour.upper()}: {info[\"rows\"]} rows, latest={info[\"latest_match_date\"]}')
"
```

### Step 2: Rebuild ELO + Features
```bash
python -c "
from src.elo_engine import compute_elo_for_tour
from src.feature_engineering import build_features
for tour in ['atp', 'wta']:
    compute_elo_for_tour(tour, incremental=False)
    print(f'{tour} ELO done')
build_features(('atp', 'wta'))
print('Features done')
"
```

### Step 3: Retrain Both Tours
```bash
python -c "
from src.model_training import train_models
import json
for tour in ['atp', 'wta']:
    result = train_models(tours=(tour,), use_optuna=False)
    m = result[tour]['metrics']['ensemble']
    print(f'{tour.upper()}: accuracy={m[\"accuracy\"]:.4f} logloss={m[\"log_loss\"]:.4f} ece={m[\"ece\"]:.4f} test_rows={result[tour][\"rows_test\"]}')
"
```

### Step 4: WTA Weight Optimization
Run blend sweep on new WTA test predictions:
```bash
python -c "
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, log_loss

preds = pd.read_csv('models/test_predictions_wta.csv')
y_true = preds['p1_wins']

best_acc, best_w = 0, 0.6
for cat_w in [round(x * 0.05, 2) for x in range(0, 21)]:
    xgb_w = 1 - cat_w
    blend = cat_w * preds['catboost_prob'] + xgb_w * preds['xgboost_prob']
    acc = accuracy_score(y_true, (blend >= 0.5).astype(int))
    ll = log_loss(y_true, np.clip(blend, 1e-7, 1-1e-7))
    marker = ' <-- current' if abs(cat_w - 0.6) < 0.01 else ''
    if acc > best_acc:
        best_acc = acc
        best_w = cat_w
    print(f'CB={cat_w:.2f} XGB={xgb_w:.2f}: acc={acc:.4f} ll={ll:.4f}{marker}')

print(f'\nBest WTA weight: CATBOOST_WEIGHT={best_w:.2f}, XGBOOST_WEIGHT={1-best_w:.2f} (acc={best_acc:.4f})')
print('NOTE: Report this finding but do NOT change config.py — we will review manually')
"
```

### Step 5: Run Backtest
```bash
python -c "
from src.backtest import run_backtest
import json
result = run_backtest(tours=('atp', 'wta'))
print(json.dumps(result, indent=2, default=str))
"
```

### Step 6: Optuna Tuning (if time permits)
```bash
python -c "
from src.model_training import train_models
import json
for tour in ['atp', 'wta']:
    result = train_models(tours=(tour,), use_optuna=True, optuna_trials=50)
    m = result[tour]['metrics']['ensemble']
    print(f'{tour.upper()} Optuna: accuracy={m[\"accuracy\"]:.4f} logloss={m[\"log_loss\"]:.4f}')
"
```

## DATA QUALITY WARNINGS

1. **ALWAYS check for duplicates after backfill** — the dedup key now uses `score` instead
   of `round`, so re-runs should be safe. But verify:
   ```python
   df['_dedup'] = df['tourney_id'].astype(str) + '|' + df['winner_id'].astype(str) + '|' + df['loser_id'].astype(str)
   dups = df[df['_dedup'].duplicated()]
   assert len(dups) == 0, f"Found {len(dups)} duplicate matches!"
   ```

2. **Verify R128 only in Grand Slams:**
   ```python
   r128_small = df[(df['draw_size'] <= 32) & (df['round'] == 'R128')]
   assert len(r128_small) == 0, f"Found {len(r128_small)} R128 matches in small-draw tournaments!"
   ```

## VALIDATION CHECKLIST

After all steps:
- [ ] ATP test accuracy 63-67%
- [ ] ATP log-loss < 0.65
- [ ] WTA test accuracy 66-72%
- [ ] WTA log-loss < 0.60
- [ ] Calibration CSVs have 10 bins: `models/calibration_atp.csv`, `models/calibration_wta.csv`
- [ ] Feature importance saved: `models/feature_importance_atp.csv`, `models/feature_importance_wta.csv`
- [ ] WTA weight sweep results reported (best weight for CatBoost/XGBoost blend)
- [ ] No duplicate matches in test set
- [ ] No R128 in small-draw tournaments
- [ ] Backtest run (even if no_odds_overlap)

## OUTPUT
```bash
python -c "
import json
for tour in ['atp', 'wta']:
    report = json.loads(open(f'models/model_report_{tour}.json').read())
    m = report['metrics']['ensemble']
    print(f'{tour.upper()}: accuracy={m[\"accuracy\"]:.4f} logloss={m[\"log_loss\"]:.4f} ece={m[\"ece\"]:.4f} test_rows={report[\"rows_test\"]}')
"
```
