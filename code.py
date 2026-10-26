%pyspark
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


df = spark.sql("SELECT * FROM traffic_summary").toPandas()

# Convert time column to actual Datetime and set it as the index
df['time_bin'] = pd.to_datetime(df['time_bin'])
df.set_index('time_bin', inplace=True)

# This resamples the data to exact 1-minute intervals and forces empty minutes to be 0
tpm = df['transaction_count'].resample('1min').sum().fillna(0).reset_index()
tpm.columns = ['timestamp', 'actual_volume']
tpm.set_index('timestamp', inplace=True)



# What was the median traffic over the last 10 minutes?
tpm['plateau_context'] = tpm['actual_volume'].rolling(window=10, min_periods=1).median().shift(1).fillna(0)

# Did we just drop suddenly from the previous minute?
tpm['sudden_drop'] = tpm['actual_volume'].diff().fillna(0)



# THE AI MODEL

features = ['actual_volume', 'plateau_context', 'sudden_drop']
X = tpm[features].fillna(0)

# Train the model (contamination=0.01 because micro-outages are rare)
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
tpm['anomaly_flag'] = iso_forest.fit_predict(X)


# We only want to see true crashes: Flagged by AI AND volume hit 0 AND it was previously busy
micro_outages = tpm[
    (tpm['anomaly_flag'] == -1) & 
    (tpm['actual_volume'] == 0) & 
    (tpm['plateau_context'] > 10) # Using 10 as a noise threshold so it ignores dead nights
]

print("--- DETECTED MIDDAY MICRO-OUTAGES ---")
print(f"Total isolated drops found: {len(micro_outages)}\n")
print(micro_outages[['actual_volume', 'plateau_context', 'sudden_drop']])