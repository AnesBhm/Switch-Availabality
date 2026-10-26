# Switch Availability & Silence Detection Platform ⚡

A high-performance anomaly detection engine and interactive **Plotly Dash** dashboard designed to detect **micro-outages, silent gaps, and SLA breaches** in electronic payment switches (SVFE / ATM / POS networks).

Traditional monitoring relies on HTTP error codes (e.g., 5xx, timeouts) or system health metrics. However, payment switch failures often manifest as **unannounced silence periods**—where transaction traffic suddenly drops to zero without explicit errors. This platform utilizes an **Hourly Half-Normal distribution model with dynamic Z-scores and physical safety floors** to isolate true silent outages from natural traffic variations.

---

## 🌟 Key Features

- **Hourly Half-Normal Anomaly Engine**: Calibrates inter-arrival transaction time ($\Delta t$) against a clean benchmark day across 24 hourly buckets.
- **Dynamic Adaptive Thresholds**: Automatically adjusts detection thresholds based on time-of-day traffic density:
  $$\text{Threshold}_h = \max\left(\mu_{\text{half}, h} + Z_h \cdot \sigma_{\text{half}, h} + \text{Buffer}, \text{MinFloor}_h\right)$$
- **Zero-Code SLA Calculation**: Translates detected silent outage durations into contract SLA availability metrics (e.g., $99.95\%$ uptime target, total downtime in minutes, MTTR).
- **Interactive Dark Glassmorphic Dashboard**:
  - 🏠 **Overview**: Real-time KPI summary cards (Uptime %, Outage Count, MTTR, Processed Transactions).
  - 📅 **Daily SLA Report**: Interactive availability breakdown per date with pass/fail SLA contract indicators.
  - 🔕 **Gap Log**: Filterable and sortable log of all individual outage events with duration and excess multipliers.
  - 📈 **Timeline**: High-resolution time-series chart showing inter-arrival $\Delta t$ against the dynamic threshold line.
  - 🎛️ **Profile**: Hourly calibration curves displaying $Z$-scores, standard deviations, and floor parameters.
  - 🔴 **Live Monitor Simulator**: Optional real-time streaming simulator evaluating synthetic transactions on the fly.
- **Distributed Big Data Engine**: Includes Apache Spark (Scala) source code (`src/`) for scalable batch processing on enterprise clusters.

---

## 📁 Repository Structure

```text
├── app.py                     # Plotly Dash web application and reactive UI
├── deployement.py             # Core Silence Detection engine & mathematical model
├── engine_api.py              # Clean API bridge between dashboard and engine
├── run_dashboard.bat          # 1-click Windows starter script (auto venv & deps)
├── requirements.txt           # Python dependency specifications
├── sample_data.csv.example    # Reference CSV template showing expected data format
├── assets/
│   └── custom.css             # Glassmorphism styling, animations, and color tokens
├── src/main/scala/...         # Apache Spark / Scala production pipeline
├── switch_framework.py        # Extended evaluation framework and batch metrics
├── evaluate_rules.py          # Rule calibration and scoring script
└── .gitignore                 # Configured to exclude heavy datasets (>100MB)
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- Git

### 2. Installation

Clone this repository:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Switch Availabality"
```

Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launching the Dashboard

#### On Windows:
Double-click `run_dashboard.bat` or run:
```bat
run_dashboard.bat
```

#### On Linux / macOS / Any OS:
```bash
python app.py
```

Once initialized, open your browser and navigate to:
```
http://127.0.0.1:8050
```

---

## 🧪 Testing with Your Own Data

The platform is designed to run directly on any transaction stream. Large raw dataset CSV files (`*.csv`) are excluded from Git to prevent repository bloat and respect GitHub file size limits.

### 1. Expected Data Format
Place your transaction data in the project root directory as **`withdeltadf.csv`**.

Only **one column is strictly required**:
- **`server_timestamp`**: Timestamp of each transaction in `YYYY-MM-DD HH:MM:SS` (or standard ISO-8601 format).

*(Optional standard switch columns: `system_trace`, `terminal_type`, `acquiring_inst`, `issuing_inst`, `delta_t_seconds`)*.

#### Quick Reference Template:
Inspect `sample_data.csv.example` for reference:
```csv
server_timestamp,terminal_type,acquiring_inst
2023-10-31 08:00:05,EPOS,9968
2023-10-31 08:00:10,ATM,198
2023-10-31 08:00:12,EPOS,196
```

To quickly test the dashboard with the sample template:
```bash
# Windows (cmd)
copy sample_data.csv.example withdeltadf.csv

# Linux / macOS
cp sample_data.csv.example withdeltadf.csv
```

### 2. Custom Date Configuration
By default, the engine calibrates on baseline date `2023-10-31` and evaluates target dates in November 2023. If your dataset contains different dates, you can adjust them in `deployement.py` inside `EngineConfig`:

```python
@dataclass(frozen=True)
class EngineConfig:
    data_path: str = "withdeltadf.csv"
    real_date: date = datetime(2024, 1, 15).date()    # Your clean baseline date
    target_dates: List[date] = field(default_factory=lambda: [
        datetime(2024, 1, 15).date(),
        datetime(2024, 1, 16).date(),
        # add other evaluation dates...
    ])
```

---

## 📤 Pushing to GitHub (Manual Guide)

If you are setting up the remote repository for the first time:

1. **Verify Git status**:
   Ensure all `.csv` files are ignored:
   ```bash
   git status
   ```

2. **Stage and commit the files**:
   ```bash
   git add .
   git commit -m "feat: initial commit for Switch Availability & Silence Detection Dashboard"
   ```

3. **Link to your GitHub repository**:
   Create a new empty repository on [GitHub](https://github.com/new), then run:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
   git push -u origin main
   ```

*(All heavy CSVs and temporary artifacts are automatically ignored by `.gitignore`)*.

---

## 🛠️ Tech Stack

- **Frontend / Visualization**: [Plotly Dash](https://dash.plotly.com/), Dash Bootstrap Components, Plotly Graph Objects
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Distributed Big Data**: Apache Spark / Scala (`SVFEAvailabilityPipeline.scala`)
- **Theme & Design**: Custom dark glassmorphism, Google Fonts (`Inter` & `Outfit`)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
