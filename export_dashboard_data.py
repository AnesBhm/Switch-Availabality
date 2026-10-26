"""
export_dashboard_data.py
========================
Runs the Switch Silence Detection pipeline and exports all results as
CSV files ready to be uploaded to any SaaS dashboard tool
(Tableau, Power BI, Looker Studio, Retool, etc.).

Output files:
  1. export_kpis.csv            — Single-row KPI summary (Overview cards)
  2. export_summary.csv         — Daily SLA availability report (Daily Report tab)
  3. export_gaps.csv            — Detected silence gap events (Gap Log tab)
  4. export_profile.csv         — Hourly calibration profile (Profile tab)
  5. export_timeline.csv        — Per-transaction data (Timeline tab) — may be large
  6. export_timeline_hourly.csv — Hourly-aggregated timeline (lighter alternative)

Run:
  .venv\\Scripts\\python.exe export_dashboard_data.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_api import run_pipeline, profile_to_df

print("=" * 60)
print("  Switch Availability — CSV Data Exporter")
print("=" * 60)

# ── 1. Run pipeline ──────────────────────────────────────────────
print("\n[1/6] Running pipeline (loading ~554 MB CSV)…")
result   = run_pipeline()
profile  = result['profile']
scored   = result['scored_df'].copy()
gaps     = result['gaps_df'].copy()
summary  = result['summary_df'].copy()
cfg      = result['config']
fp       = result['fp_baseline']
prof_df  = profile_to_df(profile)

# ── 2. KPI summary (1 row) ───────────────────────────────────────
print("[2/6] Exporting export_kpis.csv …")
eval_df = summary[summary['Type'] == 'EVALUATION']
kpi = pd.DataFrame([{
    'avg_uptime_pct':          round(eval_df['Uptime (%)'].mean(), 4),
    'total_gaps_detected':     int(eval_df['Silent Gaps'].sum()),
    'avg_mttr_min':            round(eval_df['MTTR (min)'].mean(), 3),
    'total_transactions':      int(summary['Txn Count'].sum()),
    'baseline_false_positives': fp,
    'evaluation_days':         len(eval_df),
    'baseline_date':           str(cfg.real_date),
}])
kpi.to_csv('export_kpis.csv', index=False)

# ── 3. Daily SLA summary ─────────────────────────────────────────
print("[3/6] Exporting export_summary.csv …")
summary_out = summary.copy()
summary_out.columns = [
    'date', 'day_type', 'txn_count', 'silent_gaps',
    'downtime_min', 'mttr_min', 'max_gap_sec', 'uptime_pct'
]
summary_out.to_csv('export_summary.csv', index=False)

# ── 4. Gap event log ─────────────────────────────────────────────
print("[4/6] Exporting export_gaps.csv …")
gaps_out = gaps.copy()
gaps_out['date']     = gaps_out['date'].astype(str)
gaps_out['gap_start'] = pd.to_datetime(gaps_out['gap_start']).dt.strftime('%Y-%m-%d %H:%M:%S')
gaps_out['gap_end']   = pd.to_datetime(gaps_out['gap_end']).dt.strftime('%Y-%m-%d %H:%M:%S')
gaps_out['delta_t']          = gaps_out['delta_t'].round(2)
gaps_out['switch_threshold'] = gaps_out['switch_threshold'].round(2)
gaps_out['duration_minutes'] = gaps_out['duration_minutes'].round(3)
# add severity tag
def severity(d):
    if d >= 30: return 'CRITICAL'
    if d >= 10: return 'HIGH'
    if d >= 3:  return 'MEDIUM'
    return 'LOW'
gaps_out['severity'] = gaps_out['duration_minutes'].apply(severity)
gaps_out.to_csv('export_gaps.csv', index=False)

# ── 5. Calibration profile ───────────────────────────────────────
print("[5/6] Exporting export_profile.csv …")
ref = cfg.mentor_reference
prof_df['ref_z_score']    = [ref.get(h, (np.nan, np.nan))[0] for h in range(24)]
prof_df['ref_floor_sec']  = [ref.get(h, (np.nan, np.nan))[1] for h in range(24)]
prof_df['z_score']        = prof_df['z_score'].round(2)
prof_df['min_thresh_sec'] = prof_df['min_thresh_sec'].round(3)
prof_df['sigma_half']     = prof_df['sigma_half'].round(4)
prof_df['p99_gap']        = prof_df['p99_gap'].round(3)
prof_df.to_csv('export_profile.csv', index=False)

# ── 6. Per-transaction timeline (full) ───────────────────────────
print("[6/6] Exporting export_timeline.csv and export_timeline_hourly.csv …")
tl = scored[[
    'server_timestamp', 'date', 'hour',
    'delta_t', 'switch_threshold', 'is_silent_outlier', 'mu_half', 'sigma_half'
]].copy()
tl['date']            = tl['date'].astype(str)
tl['server_timestamp'] = tl['server_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
tl['delta_t']          = tl['delta_t'].round(3)
tl['switch_threshold'] = tl['switch_threshold'].round(3)
tl['mu_half']          = tl['mu_half'].round(3)
tl['sigma_half']       = tl['sigma_half'].round(4)
tl['is_silent_outlier'] = tl['is_silent_outlier'].astype(int)   # 0/1 for SaaS compat
tl.to_csv('export_timeline.csv', index=False)

# Hourly aggregation (lighter alternative — 192 rows max)
tl_h = scored.copy()
tl_h['date'] = tl_h['date'].astype(str)
hourly = tl_h.groupby(['date', 'hour']).agg(
    txn_count        = ('delta_t', 'count'),
    avg_gap_sec      = ('delta_t', 'mean'),
    max_gap_sec      = ('delta_t', 'max'),
    p99_gap_sec      = ('delta_t', lambda x: x.quantile(0.99)),
    avg_threshold    = ('switch_threshold', 'mean'),
    silent_outliers  = ('is_silent_outlier', 'sum'),
    downtime_min     = ('delta_t', lambda x: x[scored.loc[x.index, 'is_silent_outlier']].sum() / 60),
).reset_index()
hourly = hourly.round(3)
hourly['hour_label'] = hourly['hour'].apply(lambda h: f"{h:02d}:00")
hourly.to_csv('export_timeline_hourly.csv', index=False)

print("\n" + "=" * 60)
print("  Export complete! Files saved:")
files = {
    'export_kpis.csv':             f"1 row   — KPI cards",
    'export_summary.csv':          f"{len(summary_out)} rows  — Daily SLA report",
    'export_gaps.csv':             f"{len(gaps_out)} rows  — Silent gap events",
    'export_profile.csv':          f"24 rows  — Hourly calibration profile",
    'export_timeline.csv':         f"{len(tl):,} rows — Full per-transaction data",
    'export_timeline_hourly.csv':  f"{len(hourly)} rows  — Hourly aggregated timeline",
}
for fname, desc in files.items():
    size_kb = os.path.getsize(fname) / 1024
    print(f"  ✅  {fname:<35} {desc}  ({size_kb:.0f} KB)")
print("=" * 60)
