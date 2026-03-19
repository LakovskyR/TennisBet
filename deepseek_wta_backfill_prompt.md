# CODEX/DEEPSEEK TASK: Fix WTA Backfill — Tournament Slugs + rtrvr.ai Fallback

## CONTEXT

WTA backfill script (`src/wta_backfill.py`) successfully scrapes Tennis Explorer via Firecrawl,
but **9 out of 32 tournaments return 0 matches** and the **rtrvr.ai fallback doesn't work**.

Current results: **1,337 matches** across 23 tournaments (2025) + **463 matches** across 8 tournaments (2026).

## ISSUE 1: WRONG TOURNAMENT SLUGS (ROOT CAUSE OF 0-MATCH TOURNAMENTS)

Tennis Explorer uses a **`-wta` suffix** for WTA tournament pages. Without it, the page
shows "Tournament does not exist." — which Firecrawl scrapes successfully but the parser
finds 0 matches.

### Verified Correct Slugs

| Tournament | OLD slug (WRONG) | NEW slug (CORRECT) | Verified |
|-----------|-----------------|-------------------|---------|
| Madrid | `mutua-madrid-open` | **`madrid-wta`** | ✅ 215 player links |
| Rome | `internazionali-bnl-d-italia` | **`rome-wta`** | ✅ 215 player links |
| Canada | `national-bank-open` | **`montreal-wta`** | ✅ 215 player links |
| Cincinnati | `western-and-southern-open` | **`cincinnati-wta`** | ✅ 215 player links |
| Abu Dhabi | `abu-dhabi` | **`abu-dhabi-wta`** | ✅ 79 player links |
| Seoul | `seoul` | **`seoul-wta`** | ✅ 79 player links |
| Singapore | `singapore` | **`singapore-wta`** | ✅ 87 player links |
| Austin | `austin` | **`austin-wta`** | ✅ 87 player links |
| Bad Homburg | `bad-homburg` | **`bad-homburg-wta`** | ✅ 79 player links |
| San Diego | `san-diego` | **NOT FOUND** on Tennis Explorer | ❌ Remove from registry |

### Fix Required

Update `WTA_2025_TOURNAMENTS` in `src/wta_backfill.py`:

```python
# Replace these slugs:
{"slug": "mutua-madrid-open", ...}        # -> "madrid-wta"
{"slug": "internazionali-bnl-d-italia", ...} # -> "rome-wta"
{"slug": "national-bank-open", ...}         # -> "montreal-wta"
{"slug": "western-and-southern-open", ...}  # -> "cincinnati-wta"
{"slug": "abu-dhabi", ...}                  # -> "abu-dhabi-wta"  (for BOTH 2025 and 2026!)
{"slug": "seoul", ...}                     # -> "seoul-wta"
{"slug": "singapore", ...}                # -> "singapore-wta"
{"slug": "austin", ...}                   # -> "austin-wta"
{"slug": "bad-homburg", ...}              # -> "bad-homburg-wta"

# REMOVE San Diego entirely (not on Tennis Explorer):
# {"slug": "san-diego", "name": "San Diego", ...}  # DELETE THIS ENTRY
```

Also update `WTA_2026_TOURNAMENTS`:
```python
# Abu Dhabi 2026:
{"slug": "abu-dhabi-wta", ...}   # was "abu-dhabi"
```

## ISSUE 2: rtrvr.ai FALLBACK BROKEN (TWO BUGS)

### Bug 2a: Wrong filename in `_get_rtrvr_key()`
The file is `rtvt.ai.txt` but the code looks for `rtrvr.ai.txt`:
```python
# CURRENT (WRONG):
for path in [Path("rtrvr.ai.txt"), Path(__file__).parent.parent / "rtrvr.ai.txt"]:

# FIX:
for path in [Path("rtvt.ai.txt"), Path(__file__).parent.parent / "rtvt.ai.txt"]:
```

**NOTE**: This fix was already applied to the code. Verify it's in place.

### Bug 2b: rtrvr.ai returns accessibility tree, not markdown
The `rtrvr_scrape()` function reads `tab.get("content", "")` but rtrvr.ai returns:
- **Default mode** (no `onlyTextContent`): `content` is EMPTY, data is in `tree` field (accessibility tree format)
- **`onlyTextContent: true`**: `content` has plain text (no HTML structure, no tables)

Neither format is compatible with the markdown table parser (`_parse_tennis_explorer_table()`).

### Fix Required for rtrvr.ai Integration

**Option A (Recommended)**: Don't use `onlyTextContent`. Parse the accessibility tree.

The tree format looks like this:
```
[title] Tennis Explorer: Australian Open
[document]
  [table]
    [row]
      [cell] 1R
      [cell] [link] Sabalenka A.
      [cell] 2
      [cell] 6
      [cell] 6
    [row]
      [cell]
      [cell] [link] Krueger E.
      [cell] 0
      [cell] 3
      [cell] 2
```

Write a new parser function `_parse_rtrvr_tree(tree_text, draw_size)` that extracts the same
match dicts as `_parse_tennis_explorer_table()`.

**Option B (Simpler, less reliable)**: Use `onlyTextContent: true` and write a text parser.
The text content is flat and doesn't preserve table structure well.

**Option C (Simplest)**: Since fixing the slugs solves the 0-match problem, and Firecrawl
handles all working tournaments, rtrvr.ai is only needed when Firecrawl credits are exhausted.
In that case, just fix bugs 2a/2b and note rtrvr.ai as a future enhancement.

### rtrvr.ai API Full Documentation

```
Endpoint:  POST https://api.rtrvr.ai/scrape
Auth:      Authorization: Bearer <key>
Key:       rtrvr_aGzcuAjQoySym4UtktIbCro1LgXgh24U2218MJEYzp4  (from rtvt.ai.txt)
Docs:      https://www.rtrvr.ai/docs/scrape

Request body:
{
  "urls": ["https://www.tennisexplorer.com/australian-open/2025/wta-women/"],
  "settings": {
    "extractionConfig": {
      "onlyTextContent": false,     // default; returns tree in "tree" field
      // OR
      "onlyTextContent": true       // returns plain text in "content" field
    }
  }
}

Response:
{
  "success": true,
  "tabs": [
    {
      "tabId": 123,
      "url": "...",
      "title": "Tennis Explorer: Australian Open",
      "status": "success",
      "content": "",             // empty unless onlyTextContent=true
      "tree": "[title]...",      // accessibility tree (default mode)
    }
  ],
  "usageData": {
    "totalCredits": 1,
    "totalUsd": 0.001
  }
}

Key differences from Firecrawl:
- Firecrawl returns markdown in response.data.markdown
- rtrvr.ai returns accessibility tree in response.tabs[0].tree (default)
- rtrvr.ai returns plain text in response.tabs[0].content (onlyTextContent=true)
- Neither rtrvr.ai format matches Firecrawl's markdown tables
```

### Updated `rtrvr_scrape()` function (fix for Bug 2b):
```python
def rtrvr_scrape(url: str, api_key: str) -> str | None:
    """Scrape a URL using rtrvr.ai API, returns content or tree text."""
    try:
        resp = requests.post(
            "https://api.rtrvr.ai/scrape",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"urls": [url]},  # default mode returns tree
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") and data.get("tabs"):
            tab = data["tabs"][0]
            # Try content first, fall back to tree
            return tab.get("content") or tab.get("tree") or None
        log.warning(f"  rtrvr.ai failed: {data.get('error', 'unknown')}")
        return None
    except Exception as e:
        log.warning(f"  rtrvr.ai error: {e}")
        return None
```

## EXECUTION STEPS

### Step 1: Fix Tournament Slugs
Update the 9 slugs in `WTA_2025_TOURNAMENTS` and 1 in `WTA_2026_TOURNAMENTS`.
Remove San Diego from both registries.

### Step 2: Fix rtrvr.ai Integration
1. Verify `_get_rtrvr_key()` looks for `rtvt.ai.txt` (not `rtrvr.ai.txt`)
2. Update `rtrvr_scrape()` to return `tree` field when `content` is empty
3. If implementing Option A: write `_parse_rtrvr_tree()` parser for accessibility tree format

### Step 3: Delete Old Files and Re-run
```bash
del data\raw\tennis_wta\wta_matches_2025.csv
del data\raw\tennis_wta\wta_matches_2026.csv
python -m src.wta_backfill --years 2025 2026
```

### Step 4: Validate
```bash
python -c "
import pandas as pd
for year in [2025, 2026]:
    path = f'data/raw/tennis_wta/wta_matches_{year}.csv'
    try:
        df = pd.read_csv(path)
        print(f'=== {year}: {len(df)} rows, {df[\"tourney_name\"].nunique()} tournaments ===')
        print(f'Rounds: {df[\"round\"].value_counts().sort_index().to_dict()}')

        # Verify fixed tournaments have data
        for name in ['Madrid', 'Rome', 'Canada', 'Cincinnati', 'Abu Dhabi', 'Seoul']:
            n = len(df[df['tourney_name'] == name])
            print(f'  {name}: {n} matches')

        # Verify no R128 in small-draw tournaments
        bad = df[(df['draw_size'] <= 32) & (df['round'] == 'R128')]
        print(f'R128 in small-draw: {len(bad)} (should be 0)')
    except FileNotFoundError:
        print(f'{year}: NOT FOUND')
"
```

**Expected after fixes:**
- 2025: ~1,800-2,200 matches across 31 tournaments (was 1,337 across 23)
- 2026: ~480-550 matches across 9 tournaments (was 463 across 8)
- Madrid, Rome, Canada, Cincinnati: ~55-95 matches each (was 0)
- Abu Dhabi, Seoul, Singapore, Austin, Bad Homburg: ~25-30 matches each (was 0)

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `src/wta_backfill.py` | Fix 9 tournament slugs (add `-wta`), remove San Diego, fix `rtrvr_scrape()` response parsing |

## IMPORTANT RULES

1. **Do NOT modify `src/data_updater.py`** — it's used by the daily pipeline.
2. **Do NOT modify any ATP files** — ATP data is complete and working.
3. **Do NOT change player resolution code** — it works perfectly (0 missing IDs).
4. **Do NOT change the dedup key** in `write_output()` — it's correct (uses `score`).
5. **Rate limit between scraping requests**: `time.sleep(2)` between tournaments.
6. **Delete old CSV files before running** — prevents stale data mixing.
7. **After fixing, run the full backfill and include validation output** in your response.
