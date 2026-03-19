from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROCESSED_DIR
from src.sqlite_storage import load_prediction_log_frame

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = PROCESSED_DIR / "performance_report.html"
TIER_ORDER = ["HIGH", "MEDIUM", "LOW"]
PROB_BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
PROB_BUCKET_LABELS = [
    "0.50-0.55",
    "0.55-0.60",
    "0.60-0.65",
    "0.65-0.70",
    "0.70-0.75",
    "0.75-0.80",
    "0.80-0.85",
    "0.85-0.90",
    "0.90-0.95",
    "0.95-1.00",
]


def _safe_read_prediction_log() -> pd.DataFrame:
    try:
        return load_prediction_log_frame(fallback_to_csv=False)
    except Exception as exc:
        logger.warning("Could not read prediction log from SQLite: %s", exc)
        return pd.DataFrame()


def _first_present(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_prediction_log(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "match_date",
                "p1_name",
                "p2_name",
                "bet_side",
                "recommended_stake",
                "selected_odds",
                "selected_edge",
                "selected_prob",
                "confidence_tier",
                "actual_winner",
                "result",
                "pnl",
                "status",
            ]
        )

    out = pd.DataFrame(index=df.index)
    out["match_date"] = pd.to_datetime(_first_present(df, ["match_date"]), errors="coerce")
    out["p1_name"] = _first_present(df, ["p1_name"]).astype("string").fillna("")
    out["p2_name"] = _first_present(df, ["p2_name"]).astype("string").fillna("")
    out["bet_side"] = _first_present(df, ["bet_side"]).astype("string").fillna("")
    out["recommended_stake"] = pd.to_numeric(
        _first_present(df, ["recommended_stake", "stake"]), errors="coerce"
    ).fillna(0.0)
    out["selected_odds"] = pd.to_numeric(
        _first_present(df, ["selected_odds", "odds"]), errors="coerce"
    )
    out["selected_edge"] = pd.to_numeric(
        _first_present(df, ["selected_edge", "edge"]), errors="coerce"
    )
    out["selected_prob"] = pd.to_numeric(
        _first_present(df, ["selected_prob", "probability"]), errors="coerce"
    )
    out["confidence_tier"] = _first_present(df, ["confidence_tier"]).astype("string").fillna("UNKNOWN")
    out["actual_winner"] = _first_present(df, ["actual_winner"]).astype("string").fillna("")
    out["result"] = _first_present(df, ["result"]).astype("string").fillna("")
    out["pnl"] = pd.to_numeric(_first_present(df, ["pnl"]), errors="coerce")
    out["status"] = _first_present(df, ["status"]).astype("string").fillna("")
    return out


def _non_blank_mask(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip() != ""


def _did_bet_win(row: pd.Series) -> bool:
    actual_winner = _normalize_text(row.get("actual_winner")).lower()
    bet_side = _normalize_text(row.get("bet_side")).lower()
    p1_name = _normalize_text(row.get("p1_name")).lower()
    p2_name = _normalize_text(row.get("p2_name")).lower()

    if actual_winner:
        if bet_side in {"p1", "1", "home"}:
            return actual_winner in {"p1", "1", "home", p1_name}
        if bet_side in {"p2", "2", "away"}:
            return actual_winner in {"p2", "2", "away", p2_name}

    result = _normalize_text(row.get("result")).lower()
    if result:
        return result in {"w", "win", "won", "true", "1"}

    pnl = row.get("pnl")
    return bool(pd.notna(pnl) and float(pnl) > 0.0)


def _compute_returns(settled_df: pd.DataFrame) -> pd.DataFrame:
    out = settled_df.copy()
    out["won"] = out.apply(_did_bet_win, axis=1)

    derived_return = out["recommended_stake"].where(~out["won"], out["recommended_stake"] * out["selected_odds"].fillna(0.0))
    out["total_return"] = (out["recommended_stake"] + out["pnl"]).where(out["pnl"].notna(), derived_return).fillna(0.0)
    out["bet_pnl"] = out["total_return"] - out["recommended_stake"]
    return out


def _build_overall_metrics(df: pd.DataFrame) -> dict[str, float]:
    total_bets = int(len(df))
    total_staked = float(df["recommended_stake"].sum())
    total_return = float(df["total_return"].sum())
    wins = int(df["won"].sum())
    return {
        "total_bets": total_bets,
        "win_rate": (wins / total_bets) if total_bets else 0.0,
        "total_staked": total_staked,
        "total_return": total_return,
        "roi": ((total_return - total_staked) / total_staked) if total_staked else 0.0,
    }


def _build_tier_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["confidence_tier", "count", "win_rate", "avg_edge", "roi"])

    grouped = (
        df.groupby("confidence_tier", dropna=False)
        .agg(
            count=("won", "size"),
            win_rate=("won", "mean"),
            avg_edge=("selected_edge", "mean"),
            total_staked=("recommended_stake", "sum"),
            total_return=("total_return", "sum"),
        )
        .reset_index()
    )
    grouped["roi"] = grouped.apply(
        lambda row: (
            float((row["total_return"] - row["total_staked"]) / row["total_staked"])
            if float(row["total_staked"]) > 0.0
            else 0.0
        ),
        axis=1,
    )
    grouped = grouped.drop(columns=["total_staked", "total_return"])
    grouped["confidence_tier"] = grouped["confidence_tier"].astype("string")
    grouped["sort_key"] = grouped["confidence_tier"].map({name: i for i, name in enumerate(TIER_ORDER)}).fillna(999)
    grouped = grouped.sort_values(["sort_key", "confidence_tier"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return grouped


def _build_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["bucket", "predicted_win_rate", "actual_win_rate", "count"])

    calib = df[df["selected_prob"].between(0.50, 1.00, inclusive="both")].copy()
    if calib.empty:
        return pd.DataFrame(columns=["bucket", "predicted_win_rate", "actual_win_rate", "count"])

    calib["bucket"] = pd.cut(
        calib["selected_prob"],
        bins=PROB_BUCKETS,
        labels=PROB_BUCKET_LABELS,
        include_lowest=True,
        right=True,
    )
    grouped = (
        calib.dropna(subset=["bucket"])
        .groupby("bucket", observed=True)
        .agg(
            predicted_win_rate=("selected_prob", "mean"),
            actual_win_rate=("won", "mean"),
            count=("won", "size"),
        )
        .reset_index()
    )
    return grouped


def _build_cumulative_pnl(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["match_date", "daily_pnl", "cumulative_pnl"])

    pnl = (
        df.sort_values(["match_date", "p1_name", "p2_name"])
        .groupby(df["match_date"].dt.strftime("%Y-%m-%d"), as_index=False)
        .agg(daily_pnl=("bet_pnl", "sum"))
    )
    pnl["cumulative_pnl"] = pnl["daily_pnl"].cumsum()
    return pnl


def _build_drift_metrics(df: pd.DataFrame, overall: dict[str, float]) -> dict[str, Any]:
    if df.empty:
        return {"recent_bets": 0, "recent_roi": 0.0, "drift_flag": False}

    anchor_date = df["match_date"].max()
    recent_cutoff = anchor_date - pd.Timedelta(days=29)
    recent = df[df["match_date"] >= recent_cutoff].copy()
    recent_staked = float(recent["recommended_stake"].sum())
    recent_return = float(recent["total_return"].sum())
    recent_roi = ((recent_return - recent_staked) / recent_staked) if recent_staked else 0.0
    return {
        "recent_bets": int(len(recent)),
        "recent_roi": recent_roi,
        "drift_flag": recent_roi <= (overall["roi"] - 0.10),
    }


def _summary_text(
    settled_df: pd.DataFrame,
    overall: dict[str, float],
    tier_stats: pd.DataFrame,
    drift: dict[str, Any],
    lookback_days: int,
) -> str:
    lines = [
        f"Performance report ({lookback_days}d lookback)",
        f"Settled bets: {overall['total_bets']}",
        f"Win rate: {overall['win_rate']:.1%}",
        f"Total staked: {overall['total_staked']:.2f}",
        f"Total return: {overall['total_return']:.2f}",
        f"ROI: {overall['roi']:.1%}",
    ]
    if not tier_stats.empty:
        lines.append("By tier:")
        for row in tier_stats.itertuples(index=False):
            lines.append(
                f"- {row.confidence_tier}: count={int(row.count)}, win_rate={float(row.win_rate):.1%}, "
                f"avg_edge={float(row.avg_edge):.1%}, roi={float(row.roi):.1%}"
            )
    lines.append(
        f"Recent 30d ROI: {float(drift.get('recent_roi', 0.0)):.1%} vs overall {overall['roi']:.1%}"
    )
    if drift.get("drift_flag"):
        lines.append("Potential model drift: recent ROI is at least 10 percentage points worse than overall.")
    if settled_df.empty:
        lines.append("Insufficient data: no settled bets found.")
    return "\n".join(lines)


def _minimal_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    .card {{ max-width: 760px; padding: 24px; border: 1px solid #d1d5db; border-radius: 12px; background: #f9fafb; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(body)}</p>
  </div>
</body>
</html>"""


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _build_html_report(
    settled_df: pd.DataFrame,
    overall: dict[str, float],
    tier_stats: pd.DataFrame,
    calibration: pd.DataFrame,
    cumulative_pnl: pd.DataFrame,
    drift: dict[str, Any],
    summary_text: str,
    lookback_days: int,
) -> str:
    calibration_labels = calibration["bucket"].astype(str).tolist()
    predicted_values = [round(float(v), 4) for v in calibration["predicted_win_rate"].tolist()]
    actual_values = [round(float(v), 4) for v in calibration["actual_win_rate"].tolist()]
    count_values = [int(v) for v in calibration["count"].tolist()]
    pnl_labels = cumulative_pnl["match_date"].astype(str).tolist()
    pnl_values = [round(float(v), 4) for v in cumulative_pnl["cumulative_pnl"].tolist()]

    tier_rows = []
    for row in tier_stats.itertuples(index=False):
        tier_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.confidence_tier))}</td>"
            f"<td>{int(row.count)}</td>"
            f"<td>{_format_pct(float(row.win_rate))}</td>"
            f"<td>{_format_pct(float(row.avg_edge))}</td>"
            f"<td>{_format_pct(float(row.roi))}</td>"
            "</tr>"
        )

    calibration_block = ""
    if len(calibration) >= 3:
        calibration_block = """
  <section class="panel">
    <h2>Calibration</h2>
    <canvas id="calibrationChart"></canvas>
  </section>
"""
    else:
        calibration_block = """
  <section class="panel">
    <h2>Calibration</h2>
    <p>Calibration chart skipped: fewer than 3 occupied probability buckets.</p>
  </section>
"""

    drift_notice = ""
    if drift.get("drift_flag"):
        drift_notice = "<p class=\"alert\">Potential model drift: recent 30-day ROI is at least 10 percentage points worse than overall.</p>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TennisBet Performance Report</title>
  <style>
    :root {{
      --bg: #f4efe5;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #d6cfc3;
      --accent: #0f766e;
      --accent-soft: #dff3f1;
      --warn: #b45309;
      --warn-soft: #fef3c7;
    }}
    body {{ font-family: Georgia, serif; background: linear-gradient(180deg, #efe7da 0%, var(--bg) 100%); color: var(--ink); margin: 0; }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }}
    .hero {{ margin-bottom: 24px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 36px; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px; box-shadow: 0 8px 24px rgba(31, 41, 55, 0.06); }}
    .metric-label {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ display: block; margin-top: 8px; font-size: 28px; font-weight: 700; }}
    .layout {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; }}
    .stack {{ display: grid; gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--border); padding: 10px 8px; text-align: left; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .summary {{ white-space: pre-wrap; background: #f8f4ed; border-radius: 12px; padding: 14px; }}
    .alert {{ background: var(--warn-soft); color: var(--warn); border-radius: 12px; padding: 12px; }}
    @media (max-width: 860px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Performance Dashboard</h1>
      <p>Settled bets from the last {lookback_days} days. Generated {html.escape(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))}.</p>
    </section>

    <section class="grid">
      <div class="card"><span class="metric-label">Settled Bets</span><span class="metric-value">{overall['total_bets']}</span></div>
      <div class="card"><span class="metric-label">Win Rate</span><span class="metric-value">{_format_pct(overall['win_rate'])}</span></div>
      <div class="card"><span class="metric-label">Total Staked</span><span class="metric-value">{overall['total_staked']:.2f}</span></div>
      <div class="card"><span class="metric-label">Total Return</span><span class="metric-value">{overall['total_return']:.2f}</span></div>
      <div class="card"><span class="metric-label">ROI</span><span class="metric-value">{_format_pct(overall['roi'])}</span></div>
      <div class="card"><span class="metric-label">Recent 30d ROI</span><span class="metric-value">{_format_pct(float(drift.get('recent_roi', 0.0)))}</span></div>
    </section>

    {drift_notice}

    <section class="layout">
      <div class="stack">
        {calibration_block}
        <section class="panel">
          <h2>Cumulative P&amp;L</h2>
          <canvas id="pnlChart"></canvas>
        </section>
      </div>
      <div class="stack">
        <section class="panel">
          <h2>By Confidence Tier</h2>
          <table>
            <thead>
              <tr>
                <th>Tier</th>
                <th>Count</th>
                <th>Win Rate</th>
                <th>Avg Edge</th>
                <th>ROI</th>
              </tr>
            </thead>
            <tbody>
              {''.join(tier_rows) or '<tr><td colspan="5">No tier data.</td></tr>'}
            </tbody>
          </table>
        </section>
        <section class="panel">
          <h2>Plain-Text Summary</h2>
          <div class="summary">{html.escape(summary_text)}</div>
        </section>
      </div>
    </section>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const calibrationLabels = {json.dumps(calibration_labels)};
    const predictedValues = {json.dumps(predicted_values)};
    const actualValues = {json.dumps(actual_values)};
    const bucketCounts = {json.dumps(count_values)};
    const pnlLabels = {json.dumps(pnl_labels)};
    const pnlValues = {json.dumps(pnl_values)};

    if (calibrationLabels.length >= 3) {{
      new Chart(document.getElementById('calibrationChart'), {{
        type: 'bar',
        data: {{
          labels: calibrationLabels,
          datasets: [
            {{
              label: 'Predicted win rate',
              data: predictedValues,
              backgroundColor: 'rgba(15, 118, 110, 0.75)'
            }},
            {{
              label: 'Actual win rate',
              data: actualValues,
              backgroundColor: 'rgba(180, 83, 9, 0.65)'
            }}
          ]
        }},
        options: {{
          responsive: true,
          interaction: {{ mode: 'index', intersect: false }},
          scales: {{
            y: {{
              beginAtZero: true,
              max: 1,
              ticks: {{
                callback: (value) => (value * 100).toFixed(0) + '%'
              }}
            }}
          }},
          plugins: {{
            tooltip: {{
              callbacks: {{
                afterBody: (items) => 'Count: ' + bucketCounts[items[0].dataIndex]
              }}
            }}
          }}
        }}
      }});
    }}

    new Chart(document.getElementById('pnlChart'), {{
      type: 'line',
      data: {{
        labels: pnlLabels,
        datasets: [{{
          label: 'Cumulative P&L',
          data: pnlValues,
          borderColor: '#0f766e',
          backgroundColor: 'rgba(15, 118, 110, 0.18)',
          fill: true,
          tension: 0.2
        }}]
      }},
      options: {{
        responsive: true,
        scales: {{
          y: {{
            ticks: {{
              callback: (value) => value.toFixed(2)
            }}
          }}
        }}
      }}
    }});
  </script>
</body>
</html>"""


def generate_performance_report(output_path: str | Path = DEFAULT_OUTPUT_PATH, lookback_days: int = 90) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_df = _safe_read_prediction_log()
    normalized = _normalize_prediction_log(raw_df)

    cutoff = datetime.now(UTC).date() - timedelta(days=lookback_days)
    window_df = normalized[normalized["match_date"].dt.date >= cutoff].copy() if not normalized.empty else normalized

    settled_mask = _non_blank_mask(window_df["actual_winner"])
    if "result" in window_df.columns:
        settled_mask = settled_mask | _non_blank_mask(window_df["result"])
    if "pnl" in window_df.columns:
        settled_mask = settled_mask | window_df["pnl"].notna()

    settled_df = _compute_returns(window_df[settled_mask].copy()) if not window_df.empty else window_df

    if len(settled_df) < 10:
        warning = f"Insufficient data: only {len(settled_df)} settled bets in the last {lookback_days} days."
        logger.warning(warning)
        summary_text = "\n".join(
            [
                f"Performance report ({lookback_days}d lookback)",
                f"Settled bets: {len(settled_df)}",
                "Insufficient data to compute stable performance metrics.",
            ]
        )
        output.write_text(_minimal_html("Insufficient data", warning), encoding="utf-8")
        print(summary_text)
        return {
            "status": "insufficient_data",
            "output_path": str(output),
            "settled_bets": int(len(settled_df)),
            "summary_text": summary_text,
            "lookback_days": lookback_days,
        }

    overall = _build_overall_metrics(settled_df)
    tier_stats = _build_tier_stats(settled_df)
    calibration = _build_calibration(settled_df)
    cumulative_pnl = _build_cumulative_pnl(settled_df)
    drift = _build_drift_metrics(settled_df, overall)
    summary_text = _summary_text(settled_df, overall, tier_stats, drift, lookback_days)

    html_body = _build_html_report(
        settled_df=settled_df,
        overall=overall,
        tier_stats=tier_stats,
        calibration=calibration,
        cumulative_pnl=cumulative_pnl,
        drift=drift,
        summary_text=summary_text,
        lookback_days=lookback_days,
    )
    output.write_text(html_body, encoding="utf-8")
    print(summary_text)
    return {
        "status": "ok",
        "output_path": str(output),
        "settled_bets": int(len(settled_df)),
        "overall": overall,
        "tier_stats": tier_stats.to_dict(orient="records"),
        "calibration_buckets": int(len(calibration)),
        "recent_roi": float(drift["recent_roi"]),
        "drift_flag": bool(drift["drift_flag"]),
        "summary_text": summary_text,
        "lookback_days": lookback_days,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tennis betting performance dashboard from prediction_log.csv")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output HTML file path",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Lookback window in days for settled bets",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    report = generate_performance_report(output_path=args.output, lookback_days=args.lookback_days)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
