# ============================================================================
# Switch Silence Detection — Hourly Half-Normal Engine (Calibrated + Z-Score)
# ============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# %% SECTION 1 — CONFIGURATION & PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class EngineConfig:
    """Centralized configuration for the Silence Detection Engine."""
    data_path: str = '/kaggle/input/datasets/aiden8teen/maje505/withdeltadf.csv'
    real_date: date = field(default_factory=lambda: datetime(2023, 10, 31).date())
    
    target_dates: List[date] = field(default_factory=lambda: [
        datetime(2023, 10, 31).date(),
        datetime(2023, 11, 2).date(),
        datetime(2023, 11, 3).date(),
        datetime(2023, 11, 4).date(),
        datetime(2023, 11, 5).date(),
        datetime(2023, 11, 6).date(),
        datetime(2023, 11, 7).date(),
        datetime(2023, 11, 8).date(),
    ])
    
    epsilon: float = 1e-4
    safety_margin: float = 1.15         # 15% headroom above maximum calibration gap
    constant_buffer_sec: float = 0     # 1-second physical transmission buffer
    max_calib_gap: float = 120.0        # Pre-filter cutoff to ignore macro outages during calibration
    
    # Mentor reference profile for validation benchmarking
    mentor_reference: Dict[int, Tuple[float, float]] = field(default_factory=lambda: {
        0: (7, 16.0), 1: (6, 25.0), 2: (5, 38.0), 3: (5, 50.0), 4: (5, 47.0),
        5: (6, 30.0), 6: (7, 11.0), 7: (12, 5.0), 8: (27, 2.0), 9: (3, 1.0),
        10: (5, 1.0), 11: (42, 1.0), 12: (61, 1.0), 13: (31, 1.0), 14: (10, 1.0),
        15: (32, 1.0), 16: (46, 1.0), 17: (63, 1.0), 18: (26, 6.0), 19: (19, 6.0),
        20: (4, 2.0), 21: (7, 3.0), 22: (8, 5.0), 23: (7, 10.0),
    })


# ============================================================================
# %% SECTION 2 — CORE DETECTION ENGINE
# ============================================================================

class SilenceDetector:
    """
    Half-Normal Anomaly Detection Engine for Payment Switch Silence Gaps.
    Uses hourly calibrated Z-scores, standard deviation parameters, and physical floors.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.hourly_profile: Dict[int, Dict] = {}
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads dataset, parses server timestamps, and computes arrival deltas."""
        df = pd.read_csv(self.config.data_path, parse_dates=['server_timestamp'])
        df = df.dropna(subset=['server_timestamp']).copy()
        
        df['date'] = df['server_timestamp'].dt.date
        df = df[df['date'].isin(self.config.target_dates)].copy()
        df = df.sort_values(['date', 'server_timestamp']).reset_index(drop=True)

        # Calculate delta_t per day partition
        df['delta_t'] = df.groupby('date')['server_timestamp'].diff().dt.total_seconds()
        df = df.dropna(subset=['delta_t']).copy()
        df['delta_t'] = np.maximum(df['delta_t'], self.config.epsilon)
        df['hour'] = df['server_timestamp'].dt.hour
        return df

    def calibrate(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Calibrates baseline Z-scores and floor parameters using clean day data.
        
        Mathematical Model:
        $$ \mu_{\text{half}} = \text{mean}(\Delta t), \quad \sigma_{\text{half}} = \frac{\mu_{\text{half}}}{\sqrt{2/\pi}} $$
        $$ Z_{\text{score}} = \left\lceil \frac{\text{SafetyMargin} \times \max(\Delta t)}{\sigma_{\text{half}}} \right\rceil $$
        """
        baseline_df = df[df['date'] == self.config.real_date]
        profile = {}
        
        for hour, grp in baseline_df.groupby('hour'):
            # Filter macro outages to isolate natural network queuing jitter
            clean_grp = grp[grp['delta_t'] < self.config.max_calib_gap]
            delta = clean_grp['delta_t'] if len(clean_grp) > 0 else grp['delta_t']
            
            mu_half = delta.mean()
            sigma_half = mu_half / np.sqrt(2.0 / np.pi)
            max_gap = delta.max()
            
            z_score = (
                float(np.ceil(self.config.safety_margin * max_gap / sigma_half))
                if sigma_half > 0 else 1.0
            )
                
            profile[hour] = {
                'z_score': z_score,
                'std_count': z_score,
                'min_thresh_sec': round(float(max_gap), 3),
                'n_transactions': len(grp),
                'p99_gap': round(float(delta.quantile(0.99)), 3),
                'sigma_half': round(float(sigma_half), 4)
            }
            
        # Impute any missing hours using nearest neighbor calibration
        for h in range(24):
            if h not in profile:
                nearest = min(profile.keys(), key=lambda x: abs(x - h)) if profile else None
                profile[h] = dict(profile[nearest]) if nearest is not None else {
                    'z_score': 1.0, 'std_count': 1.0, 'min_thresh_sec': self.config.epsilon,
                    'n_transactions': 0, 'p99_gap': 0.0, 'sigma_half': 0.0
                }
                
        self.hourly_profile = profile
        return profile

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates dynamic thresholds across target dates.
        
        Dynamic Threshold Equation:
        $$ \text{Threshold} = \max\left( (Z_{\text{score}} \times \sigma_{\text{half}}) + C_{\text{buffer}}, \, \text{Floor} \right) $$
        """
        z_map = {h: self.hourly_profile[h]['z_score'] for h in self.hourly_profile}
        floor_map = {h: self.hourly_profile[h]['min_thresh_sec'] for h in self.hourly_profile}

        processed = []
        for date_, group in df.groupby('date', sort=True):
            g = group.sort_values('server_timestamp').copy()

            g['mu_half'] = g.groupby('hour')['delta_t'].transform('mean')
            g['sigma_half'] = g['mu_half'] / np.sqrt(2.0 / np.pi)

            g['z_score'] = g['hour'].map(z_map)
            g['std_count'] = g['z_score']
            g['min_switch_thresh'] = g['hour'].map(floor_map)

            # Dynamic threshold formulation
            g['calc_thresh'] = (g['z_score'] * g['sigma_half']) + self.config.constant_buffer_sec
            g['switch_threshold'] = np.maximum(g['calc_thresh'], g['min_switch_thresh'])
            g['is_silent_outlier'] = g['delta_t'] > g['switch_threshold']

            processed.append(g)
            
        return pd.concat(processed, ignore_index=True)


# ============================================================================
# %% SECTION 3 — ANALYTICS & REPORTING
# ============================================================================

def generate_sla_report(scored_df: pd.DataFrame, config: EngineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates detailed outage gap logs and daily availability summaries."""
    silents = scored_df[scored_df['is_silent_outlier']].copy()
    
    if len(silents) > 0:
        silents['gap_start'] = silents['server_timestamp'] - pd.to_timedelta(silents['delta_t'], unit='s')
        silents = silents.rename(columns={'server_timestamp': 'gap_end'})
        silents['duration_minutes'] = silents['delta_t'] / 60.0
        cols = ['date', 'hour', 'gap_start', 'gap_end', 'delta_t', 'switch_threshold', 'duration_minutes']
        gaps_df = silents[cols].sort_values('duration_minutes', ascending=False).reset_index(drop=True)
    else:
        gaps_df = pd.DataFrame(columns=['date', 'hour', 'gap_start', 'gap_end', 'delta_t', 'switch_threshold', 'duration_minutes'])

    summary = []
    for d in sorted(scored_df['date'].unique()):
        grp = scored_df[scored_df['date'] == d]
        day_gaps = gaps_df[gaps_df['date'] == d] if len(gaps_df) else pd.DataFrame()

        gap_count = len(day_gaps)
        downtime_min = day_gaps['duration_minutes'].sum() if gap_count > 0 else 0.0
        uptime_pct = 100.0 * (1.0 - downtime_min / 1440.0)
        
        # Operational Reliability Metrics
        mttr_min = downtime_min / gap_count if gap_count > 0 else 0.0
        max_gap_sec = day_gaps['delta_t'].max() if gap_count > 0 else 0.0

        summary.append({
            'Date': d,
            'Type': 'BASELINE' if d == config.real_date else 'EVALUATION',
            'Txn Count': len(grp),
            'Silent Gaps': gap_count,
            'Downtime (min)': round(downtime_min, 2),
            'MTTR (min)': round(mttr_min, 2),
            'Max Gap (s)': round(max_gap_sec, 1),
            'Uptime (%)': round(uptime_pct, 4)
        })

    return gaps_df, pd.DataFrame(summary)


def print_profile_benchmark(calibrated: Dict[int, Dict], reference: Dict[int, Tuple[float, float]]):
    """Prints comparison table between runtime calibration and reference profile."""
    print(f"\n{'Hour':>4} | {'Txns':>7} | {'Calib Z':>8} {'Ref Z':>7} | {'Calib Floor(s)':>14} {'Ref Floor(s)':>12} | {'P99 Gap(s)':>10}")
    print("-" * 75)
    for h in range(24):
        c = calibrated[h]
        r = reference.get(h, (float('nan'), float('nan')))
        print(f"{h:>4} | {c['n_transactions']:>7} | "
              f"{c['z_score']:>8.0f} {r[0]:>7} | "
              f"{c['min_thresh_sec']:>14.2f} {r[1]:>12.1f} | "
              f"{c['p99_gap']:>10.2f}")


# ============================================================================
# %% SECTION 4 — VISUALIZATION SUITE
# ============================================================================

def plot_profile_diagnostics(calibrated: Dict[int, Dict], reference: Dict[int, Tuple[float, float]]):
    """Plots Z-scores and dynamic floors against reference parameters."""
    hours = list(range(24))
    calib_z = [calibrated[h]['z_score'] for h in hours]
    calib_floor = [calibrated[h]['min_thresh_sec'] for h in hours]
    ref_z = [reference.get(h, (np.nan, np.nan))[0] for h in hours]
    ref_floor = [reference.get(h, (np.nan, np.nan))[1] for h in hours]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), dpi=100)

    # Panel 1: Z-Score Multipliers
    axes[0].plot(hours, calib_z, 'o-', color='#1f77b4', lw=2, label='Calibrated Z-Score')
    axes[0].plot(hours, ref_z, 's--', color='#7f7f7f', alpha=0.7, label='Reference Baseline')
    axes[0].set_title('Hourly Z-Score Multiplier ($Z_h$)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Hour of Day (0–23)')
    axes[0].set_ylabel('Z-Score Value')
    axes[0].set_xticks(hours)
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(frameon=True)

    # Panel 2: Floor Thresholds
    axes[1].plot(hours, calib_floor, 'o-', color='#d62728', lw=2, label='Calibrated Floor')
    axes[1].plot(hours, ref_floor, 's--', color='#7f7f7f', alpha=0.7, label='Reference Floor')
    axes[1].set_title('Minimum Physical Floor ($\text{min\_thresh\_sec}$)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour of Day (0–23)')
    axes[1].set_ylabel('Threshold (seconds, log-scale)')
    axes[1].set_yscale('log')
    axes[1].set_xticks(hours)
    axes[1].grid(True, which='both', linestyle='--', alpha=0.4)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.show()


def plot_timeline_step_charts(scored_df: pd.DataFrame, dates_to_plot: List[date], config: EngineConfig):
    """Plots silence timeline graphs overlaid with dynamic dynamic threshold boundaries."""
    fig, axes = plt.subplots(len(dates_to_plot), 1, figsize=(15, 3.8 * len(dates_to_plot)), dpi=100)
    if len(dates_to_plot) == 1:
        axes = [axes]

    for ax, d in zip(axes, dates_to_plot):
        day_df = scored_df[scored_df['date'] == d].copy()
        
        # Plot Arrival Gaps and Threshold Line
        ax.plot(day_df['server_timestamp'], day_df['delta_t'],
                color='#2b5c8f', alpha=0.6, lw=0.9, label='Silence Gap ($\Delta t$)')
        ax.plot(day_df['server_timestamp'], day_df['switch_threshold'],
                color='#1b9e77', lw=2.0, label='Switch Threshold ($Z \cdot \sigma + 1.0\text{s}$)')

        # Highlight Anomaly Outliers
        outliers = day_df[day_df['is_silent_outlier']]
        if len(outliers) > 0:
            ax.scatter(outliers['server_timestamp'], outliers['delta_t'],
                       color='#e41a1c', s=45, zorder=5, label='Detected Unavailability', edgecolors='black', linewidth=0.5)

        is_base = (d == config.real_date)
        tag = "CALIBRATION BASELINE" if is_base else "EVALUATION DATE"
        ax.set_title(f"Date: {d} [{tag}] — Dynamic Silence Threshold vs. Arrival Gaps", fontweight='bold', fontsize=11)
        ax.set_ylabel('Silence (seconds)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()


# ============================================================================
# %% SECTION 5 — MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    # 1. Initialize Configuration
    cfg = EngineConfig()
    engine = SilenceDetector(cfg)
    
    # 2. Preprocess & Calibrate
    print("Loading switch telemetry dataset...")
    raw_df = engine.load_and_preprocess()
    
    print(f"Calibrating baseline hourly profile using date: {cfg.real_date}...")
    profile = engine.calibrate(raw_df)
    
    # 3. Print Profile Benchmarks
    print_profile_benchmark(profile, cfg.mentor_reference)
    
    # 4. Score Dataset Across Target Dates
    print("\nEvaluating dynamic thresholds across target dates...")
    scored_df = engine.score(raw_df)
    
    # 5. Check False Positives on Baseline
    fp_baseline = scored_df[scored_df['date'] == cfg.real_date]['is_silent_outlier'].sum()
    print(f"False Positives on Baseline Date ({cfg.real_date}): {fp_baseline}")

    # 6. SLA Availability Summary
    gaps_df, summary_df = generate_sla_report(scored_df, cfg)
    
    print("\n" + "=" * 90)
    print("      SWITCH SILENCE DETECTION ENGINE — SLA AVAILABILITY & RELIABILITY REPORT")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)

    # 7. Diagnostics & Timeline Charts
    plot_profile_diagnostics(profile, cfg.mentor_reference)
    plot_timeline_step_charts(scored_df, [cfg.real_date, datetime(2023, 11, 2).date(), datetime(2023, 11, 6).date()], cfg)

    # 8. Export Outputs
    gaps_df.to_csv('silent_gaps_calibrated.csv', index=False)
    summary_df.to_csv('availability_summary_calibrated.csv', index=False)
    pd.DataFrame([{'hour': h, **profile[h]} for h in range(24)]).to_csv('calibrated_hourly_profile.csv', index=False)
    print("\nArtifacts exported: 'silent_gaps_calibrated.csv', 'availability_summary_calibrated.csv', 'calibrated_hourly_profile.csv'")

if __name__ == '__main__':
    main()