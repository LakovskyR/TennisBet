# CODEX TASK: Batch A — Fix rtrvr.ai, Re-run WTA Backfill, Retrain, WTA Weights

## OVERVIEW

Three tasks in dependency order:
1. **12.2** Diagnose and fix rtrvr.ai 401 error (or confirm key is dead and ensure graceful skip)
2. **12.3** Re-run WTA backfill to capture 9 missing tournaments, then retrain
3. **11.4** Implement tour-specific ensemble weights (WTA uses different CB/XGB blend than ATP)

---

## TASK 1: Fix rtrvr.ai 401 (12.2)

### Problem
The rtrvr.ai fallback scraper in `src/wta_backfill.py` returns HTTP 401 Unauthorized.
API key is in `rtvt.ai.txt` (note the filename: `rtvt.ai.txt`, NOT `rtrvr.ai.txt`).

### Steps

1. Read the API key from `rtvt.ai.txt`
2. Test it directly:
```bash
set PYTHONIOENCODING=utf-8
python -c "
import requests, json
key = open('rtvt.ai.txt').read().strip()
print(f'Key: {key[:10]}...{key[-4:]}')
resp = requests.post(
    'https://api.rtrvr.ai/scrape',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json={'urls': ['https://www.tennisexplorer.com/madrid-wta/2025/wta-women/']},
    timeout=60
)
print(f'Status: {resp.status_code}')
print(f'Body: {resp.text[:500]}')
"
```

3. **If 401 persists**: The key is dead. Make `rtrvr_scrape()` in `wta_backfill.py`:
   - Log a clear WARNING: `log.warning("rtrvr.ai returned 401 — API key may be expired. Skipping rtrvr.ai fallback.")`
   - Return empty result gracefully (don't crash, don't retry endlessly)
   - The scraping waterfall will then be: Firecrawl → skip (rtrvr.ai disabled)

4. **If 200 OK**: Great — the key works. Check that `rtrvr_scrape()` is reading the response correctly:
   - rtrvr.ai returns data in `response["tabs"][0]["tree"]` field (NOT `content`)
   - The parser should use: `tab.get("content") or tab.get("tree")`
   - Verify this is implemented in `_parse_rtrvr_tree()`

### Files
- `rtvt.ai.txt` (API key file)
- `src/wta_backfill.py` — functions: `rtrvr_scrape()`, `_get_rtrvr_key()`, `_parse_rtrvr_tree()`

---

## TASK 2: Re-run WTA Backfill + Retrain (12.3)

### Prerequisites
- Task 1 done (rtrvr.ai either fixed or gracefully disabled)
- Firecrawl API key in `api.txt`: `fc-ea5148c21a594055b6017da6c99784c0`

### ⚠️ IMPORTANT: Firecrawl API is CONFIRMED WORKING (tested 2026-03-19)
The Firecrawl REST API works. If you get connection errors, **the issue is your network/sandbox**, not the API. The scraping code in `wta_backfill.py` calls Firecrawl via HTTP `requests.post("https://api.firecrawl.dev/v1/scrape", ...)` — it does NOT use MCP or npx. Verify:
```bash
set PYTHONIOENCODING=utf-8
python -c "
import requests
key = open('api.txt').read().strip()
resp = requests.post('https://api.firecrawl.dev/v1/scrape',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json={'url': 'https://www.tennisexplorer.com/madrid-wta/2025/wta-women/'},
    timeout=60)
print(f'Status: {resp.status_code}, Success: {resp.json().get(\"success\")}')
print(f'Content length: {len(resp.text)} chars')
"
```
If this test returns 200 + success=true, Firecrawl is fine and you can proceed with the backfill. If you get a connection error from your sandbox, report it and skip to retrain with existing data.

### Steps

1. **Delete old WTA CSVs and re-run backfill**:
```bash
set PYTHONIOENCODING=utf-8
del data\raw\tennis_wta\wta_matches_2025.csv
del data\raw\tennis_wta\wta_matches_2026.csv
python -m src.wta_backfill --years 2025 2026
```

2. **Verify results**:
```bash
set PYTHONIOENCODING=utf-8
python -c "
import pandas as pd
for year in [2025, 2026]:
    df = pd.read_csv(f'data/raw/tennis_wta/wta_matches_{year}.csv')
    print(f'{year}: {len(df)} rows, {df[\"tourney_name\"].nunique()} tournaments')
    print(f'  Tournaments: {sorted(df[\"tourney_name\"].unique())}')
    # Check for duplicates
    dedup = df['tourney_id'].astype(str) + '|' + df['winner_id'].astype(str) + '|' + df['loser_id'].astype(str) + '|' + df['score'].astype(str)
    dups = dedup[dedup.duplicated()]
    print(f'  Duplicates: {len(dups)}')
    # Check R128 in small draws
    if 'draw_size' in df.columns:
        r128_bad = df[(df['draw_size'] <= 32) & (df['round'] == 'R128')]
        print(f'  R128 in small draws: {len(r128_bad)}')
"
```

**Expected**: 2025 should have ~1,800-2,200 matches (up from 1,337) if the missing 9 tournaments are captured.

3. **If Firecrawl credits are 0 AND rtrvr.ai returns 401**: Skip to retrain with existing data. Report which tournaments are still missing.

4. **Retrain** — follow `codex_retrain_prompt.md` Steps 1-5:
```bash
set PYTHONIOENCODING=utf-8
python -c "
from src.data_pipeline import run_pipeline
result = run_pipeline(incremental=False)
for tour, info in result.items():
    print(f'{tour.upper()}: {info[\"rows\"]} rows, latest={info[\"latest_match_date\"]}')
"
```
```bash
set PYTHONIOENCODING=utf-8
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
```bash
set PYTHONIOENCODING=utf-8
python -c "
from src.model_training import train_models
for tour in ['atp', 'wta']:
    result = train_models(tours=(tour,), use_optuna=False)
    m = result[tour]['metrics']['ensemble']
    print(f'{tour.upper()}: accuracy={m[\"accuracy\"]:.4f} logloss={m[\"log_loss\"]:.4f} ece={m[\"ece\"]:.4f} test_rows={result[tour][\"rows_test\"]}')
"
```

5. **Run WTA weight sweep** (needed for Task 3):
```bash
set PYTHONIOENCODING=utf-8
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

print(f'\nBest WTA: CB={best_w:.2f} XGB={1-best_w:.2f} (acc={best_acc:.4f})')
"
```

### Validation Checklist (same as codex_retrain_prompt.md)
- [ ] ATP test accuracy 63-67%
- [ ] ATP log-loss < 0.65
- [ ] WTA test accuracy 66-72%
- [ ] WTA log-loss < 0.60
- [ ] No duplicate matches
- [ ] No R128 in small-draw tournaments
- [ ] WTA weight sweep results reported

---

## TASK 3: Tour-Specific Ensemble Weights (11.4)

### Problem
WTA weight sweep consistently shows XGBoost alone or heavy-XGBoost blend outperforms the global 0.6/0.4 CB/XGB used for both tours. ATP performs fine with 0.6/0.4.

### Steps

1. **Add tour-specific weight constants to `config.py`**:
```python
# After existing CATBOOST_WEIGHT / XGBOOST_WEIGHT lines:
CATBOOST_WEIGHT_ATP = 0.6
XGBOOST_WEIGHT_ATP = 0.4
# Use the best weight from the Task 2 sweep above:
CATBOOST_WEIGHT_WTA = <best_cat_w from sweep>
XGBOOST_WEIGHT_WTA = <1 - best_cat_w>
```

Keep the old `CATBOOST_WEIGHT = 0.6` / `XGBOOST_WEIGHT = 0.4` as fallback defaults.

2. **Update `src/predictor.py`** to use tour-specific weights:
   - Find where `CATBOOST_WEIGHT` / `XGBOOST_WEIGHT` are used for blending
   - Replace with tour-aware lookup:
```python
import config
cat_w = getattr(config, f'CATBOOST_WEIGHT_{tour.upper()}', config.CATBOOST_WEIGHT)
xgb_w = getattr(config, f'XGBOOST_WEIGHT_{tour.upper()}', config.XGBOOST_WEIGHT)
```

3. **Verify** — quick sanity check that predictions change for WTA but not ATP:
```bash
set PYTHONIOENCODING=utf-8
python -c "
import config
for tour in ['atp', 'wta']:
    cat_w = getattr(config, f'CATBOOST_WEIGHT_{tour.upper()}', config.CATBOOST_WEIGHT)
    xgb_w = getattr(config, f'XGBOOST_WEIGHT_{tour.upper()}', config.XGBOOST_WEIGHT)
    print(f'{tour.upper()}: CatBoost={cat_w}, XGBoost={xgb_w}')
"
```

### Files
- `config.py` — add `CATBOOST_WEIGHT_ATP`, `XGBOOST_WEIGHT_ATP`, `CATBOOST_WEIGHT_WTA`, `XGBOOST_WEIGHT_WTA`
- `src/predictor.py` — update blending to use tour-specific weights

---

## OUTPUT FORMAT

Report after all 3 tasks:

```
TASK 1 (rtrvr.ai):
  - Status: FIXED / KEY_DEAD / SKIPPED
  - Action taken: <what you did>

TASK 2 (WTA backfill + retrain):
  - 2025 matches: <count> rows, <N> tournaments
  - 2026 matches: <count> rows, <N> tournaments
  - Missing tournaments: <list, if any>
  - ATP retrain: accuracy=<X> logloss=<X> test_rows=<X>
  - WTA retrain: accuracy=<X> logloss=<X> test_rows=<X>
  - WTA weight sweep best: CB=<X> XGB=<X> acc=<X>

TASK 3 (tour-specific weights):
  - ATP weights: CB=<X> XGB=<X>
  - WTA weights: CB=<X> XGB=<X>
  - Config updated: YES/NO
  - Predictor updated: YES/NO
```

Do NOT commit or push. Report results only.

## IMPORTANT NOTES
- Always use `set PYTHONIOENCODING=utf-8` before running Python (Windows console has cp1252 issues with Cyrillic characters in paths)
- The dedup key uses `score` not `round` — don't change this
- If any step fails, report the error and continue to the next task
