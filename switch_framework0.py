
import pandas as pd
import numpy as np
import math
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from datetime import timedelta
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.offline as pyo

pyo.init_notebook_mode(connected=True)
file_path = "/kaggle/input/datasets/anesbhm/data-set/withdeltadf.csv"
df = pd.read_csv(file_path, low_memory=False)
#df = df.sort_values('server_timestamp')
# 3. Convert to DateTime and set as index
#df['server_timestamp'] = pd.to_datetime(df['server_timestamp'])

#df.set_index('server_timestamp', inplace=True)

# ---------------------------------------------------------
# 1. PREPARATION & FEATURE ENGINEERING
# ---------------------------------------------------------
# Ensure timestamp is datetime and sort chronologically
df['server_timestamp'] = pd.to_datetime(df['server_timestamp'])
start_time = pd.to_datetime('2023-10-31 00:04:00')
end_time = pd.to_datetime('2023-11-01 00:00:00')

df = df[(df['server_timestamp'] >= start_time) & (df['server_timestamp'] <= end_time)].copy()
df = df.sort_values('server_timestamp').reset_index(drop=True)


# Extract temporal features for hourly grouping and daily SLA aggregation
#df['hour'] = df['server_timestamp'].dt.hour
df['hour'] = df['server_timestamp'].dt.floor('hour')
df['date'] = df['server_timestamp'].dt.date


# ---------------------------------------------------------
# 2. STATISTICAL MODELING (Half-Normal to Normal Transformation)
# ---------------------------------------------------------
# Calculate the Half-Normal Mean (μ_half) and the 99th percentile for each hour
hourly_stats = df.groupby('hour')['delta_t_seconds'].agg(
    mu_half='mean',
    p99=lambda x: x.quantile(0.99)
).reset_index()

# Transform to Normal Standard Deviation (σ) using the MajestEye constant
# σ = μ_half / sqrt(2/π)
hourly_stats['sigma'] = hourly_stats['mu_half'] / math.sqrt(2 / math.pi)

# ---------------------------------------------------------
# 3. DYNAMIC HOURLY THRESHOLD CALIBRATION
# ---------------------------------------------------------
# Instead of hardcoding k, we dynamically calculate the threshold.
# We set the baseline threshold to bound the 99th percentile of traffic, 
# ensuring quiet hours and busy hours scale their k_h multiplier perfectly.

hourly_stats['k_base'] = hourly_stats['p99'] / hourly_stats['sigma']
#hourly_stats['k_dynamic'] = np.maximum(3.0, hourly_stats['k_base'] * 1.3)
hourly_stats['k_dynamic'] =hourly_stats['k_base'] * 1.2
#MIN_DOWNTIME_FLOOR = 2.0
#hourly_stats['dynamic_threshold'] = np.maximum( hourly_stats['k_dynamic'] * hourly_stats['sigma'],MIN_DOWNTIME_FLOOR )
hourly_stats['dynamic_threshold'] = (hourly_stats['k_dynamic'] * hourly_stats['sigma']) + hourly_stats["p99"]
#hourly_stats['dynamic_threshold'] = np.maximum(6 * hourly_stats['sigma'], hourly_stats['p99'] * 1.5)
print("--- DYNAMIC HOURLY CALIBRATION ---")
print(hourly_stats[['hour', 'mu_half', 'sigma', 'k_dynamic', 'dynamic_threshold']].round(3))


# Back-calculate the standard deviation count (k_h) for transparency
hourly_stats['k_h'] = hourly_stats['dynamic_threshold'] / hourly_stats['sigma']

# Map the computed thresholds back to every single transaction event
df = df.merge(hourly_stats[['hour', 'sigma', 'k_h', 'dynamic_threshold']], on='hour', how='left')

# ---------------------------------------------------------
# 4. OUTLIER CLASSIFICATION & UNAVAILABILITY CALCULATION
# ---------------------------------------------------------
# Flag events where the silence exceeds the dynamic hourly threshold

df['z_score'] = df['delta_t_seconds'] / df['sigma']
df['is_outlier'] = df['z_score'] > df['k_h']

# If an outlier is detected, the entirety of that silence is classified as downtime
df['unavailability_seconds'] = np.where(df['is_outlier'], df['delta_t_seconds'], 0)

# ---------------------------------------------------------
# 5. SLA METRICS & DELIVERABLES
# ---------------------------------------------------------
total_downtime_sec = df['unavailability_seconds'].sum()
total_downtime_min = total_downtime_sec / 60

# Calculate Total Evaluated Time (End Timestamp - Start Timestamp)
total_time_sec = (df['server_timestamp'].max() - df['server_timestamp'].min()).total_seconds()
availability_pct = (1 - (total_downtime_sec / total_time_sec)) * 100

print("=========================================")
print(" MAJESTEYE PIPELINE: SWITCH AVAILABILITY ")
print("=========================================")
print(f"Total Evaluated Time    : {total_time_sec / 3600:.2f} hours")
print(f"Total Unavailability    : {total_downtime_min:,.0f} minutes")
print(f"Switch Availability     : {availability_pct:.4f}%")
print(f"SLA Adherence Target    : 99.99%")
print("=========================================\n")

# Top Worst Days Table
worst_days = df.groupby('date')['unavailability_seconds'].sum() / 60
worst_days = worst_days.sort_values(ascending=False).reset_index()
worst_days.columns = ['Date', 'Unavailability (Minutes)']

print("--- TOP WORST DAYS ---")
print(worst_days[worst_days['Unavailability (Minutes)'] > 0].head(10).to_string(index=False))


# ---------------------------------------------------------
# 6. INTERACTIVE VISUALIZATION (Kaggle/Notebook Fix)
# ---------------------------------------------------------
import plotly.io as pio

# 1. Quick Data Check: Ensure the filter didn't accidentally empty the DataFrame
print(f"Data rows ready for plotting: {len(df)}")

if len(df) == 0:
    print("WARNING: The DataFrame is empty. Check if your timestamps have a timezone (e.g., UTC) that is throwing off the filter.")
else:
    # 2. Force the notebook to use the iframe renderer (Highly reliable for Kaggle)
    pio.renderers.default = "iframe"
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['server_timestamp'], 
        y=df['delta_t_seconds'],
        mode='lines',
        name='Silence (Δt)',
        line=dict(color='#A3B1C6', width=1),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=df['server_timestamp'], 
        y=df['dynamic_threshold'],
        mode='lines',
        name='Dynamic Threshold',
        line=dict(color='#00529B', width=2, shape='hv') 
    ))

    outliers = df[df['is_outlier'] == True]

    fig.add_trace(go.Scatter(
        x=outliers['server_timestamp'], 
        y=outliers['delta_t_seconds'],
        mode='markers',
        name='Flagged Downtime',
        marker=dict(
        color='#E37222',  # Highly visible alert color
        size=6, 
        symbol='circle',
        line=dict(color='red', width=1)
        )
        ))
    
    fig.update_layout(
        title='MajestEye Methodology: Oct 31 Isolated Evaluation',
        xaxis_title='Server Time',
        yaxis_title='Silence (Seconds)',
        hovermode='x unified',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#EAEAEA'),
        yaxis=dict(showgrid=True, gridcolor='#EAEAEA')
    )

    # 3. Show the plot using the forced iframe renderer
    fig.show()