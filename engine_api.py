"""
engine_api.py
=============
Importable adapter around the Switch Silence Detection Engine.

- Fixes the data_path to use the local withdeltadf.csv
- Removes all matplotlib rendering (plt.show calls live in deployement.py)
- Exposes a single `run_pipeline()` function used by app.py
"""

import os
import sys

# Make sure deployement.py is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Re-import core classes from deployement.py (keeps a single source of truth)
# ---------------------------------------------------------------------------
from deployement import SilenceDetector, generate_sla_report

# ---------------------------------------------------------------------------
# We need to create an EngineConfig with the correct local data_path.
# EngineConfig is frozen, so we just pass the path at construction time.
# ---------------------------------------------------------------------------
from deployement import EngineConfig as _EngineConfig


_DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'withdeltadf.csv')


def build_config(data_path: str = None) -> _EngineConfig:
    """Return an EngineConfig pointed at the local CSV."""
    return _EngineConfig(data_path=data_path or _DEFAULT_CSV)


# ---------------------------------------------------------------------------
# Main pipeline entry-point
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str = None) -> dict:
    """
    Execute the full Switch Silence Detection pipeline.

    Returns
    -------
    dict with keys:
        profile     – Dict[int, Dict]  hourly calibration profile (24 slots)
        scored_df   – pd.DataFrame     all transactions with threshold & outlier flags
        gaps_df     – pd.DataFrame     detected silent gap events
        summary_df  – pd.DataFrame     per-day SLA availability summary
        config      – EngineConfig     the config used for this run
        fp_baseline – int              false positives on the baseline date
    """
    cfg = build_config(data_path)
    engine = SilenceDetector(cfg)

    print("[engine_api] Loading and preprocessing data …")
    raw_df = engine.load_and_preprocess()

    print(f"[engine_api] Calibrating baseline profile for {cfg.real_date} …")
    profile = engine.calibrate(raw_df)

    print("[engine_api] Scoring all target dates …")
    scored_df = engine.score(raw_df)

    fp_baseline = int(scored_df[scored_df['date'] == cfg.real_date]['is_silent_outlier'].sum())
    print(f"[engine_api] False positives on baseline date: {fp_baseline}")

    gaps_df, summary_df = generate_sla_report(scored_df, cfg)

    # Ensure date columns are strings for JSON-serialisability in Dash
    summary_df['Date'] = summary_df['Date'].astype(str)
    gaps_df['date']    = gaps_df['date'].astype(str)

    print("[engine_api] Pipeline complete.")
    return {
        'profile':     profile,
        'scored_df':   scored_df,
        'gaps_df':     gaps_df,
        'summary_df':  summary_df,
        'config':      cfg,
        'fp_baseline': fp_baseline,
    }


# ---------------------------------------------------------------------------
# Helper: build profile DataFrame for the dashboard Profile tab
# ---------------------------------------------------------------------------

def profile_to_df(profile: dict) -> pd.DataFrame:
    """Convert the hourly profile dict to a tidy DataFrame."""
    rows = []
    for h in range(24):
        p = profile[h]
        rows.append({
            'hour':           h,
            'label':          f"{h:02d}:00",
            'z_score':        p['z_score'],
            'min_thresh_sec': p['min_thresh_sec'],
            'sigma_half':     p['sigma_half'],
            'p99_gap':        p['p99_gap'],
            'n_transactions': p['n_transactions'],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    result = run_pipeline()
    print("\n=== SLA Summary ===")
    print(result['summary_df'].to_string(index=False))
    print(f"\nProfile slots calibrated: {len(result['profile'])}")
