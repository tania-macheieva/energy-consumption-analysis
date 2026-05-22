# Energy Data Pipeline

A simple Python ETL (Extract, Transform, Load/Visualize) pipeline that collects real hourly weather data, simulates household energy consumption based on temperature and time, and generates visual analytics and reports.

## Features

- **Collect:** Fetches historical weather data from the open-meteo API.
- **Clean & Model:** Handles missing values and models a realistic hourly energy consumption curve.
- **Visualize:** Generates charts (trends, histograms, volatility, heatmaps).
- **Report:** Compiles an automated Markdown report with data insights.

## Prerequisites

- Python 3.8+
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `requests`

You can install the dependencies using:
```bash
pip install pandas numpy matplotlib requests

```

## Usage

The project uses a command-line interface (CLI). To run the full pipeline, execute the following commands in order:

### 1. Collect Data

Fetches the weather data from the API and saves it as a raw JSON file.

```bash
python main.py collect

```

### 2. Clean and Process

Cleans the raw data and generates the modeled electricity consumption dataset.

```bash
python main.py clean --infile ./data/raw/energy_consumption.json

```

### 3. Generate Visualizations

Creates analytical plots (scatter plots, heatmaps, time-series lines).

```bash
python main.py viz --infile ./data/processed/clean.csv

```

### 4. Generate Report

Compiles a final markdown report with references to the generated charts.

```bash
python main.py report --raw ./data/raw/energy_consumption.json --clean ./data/processed/clean.csv --cleaninfo ./artifacts/reports/clean_info.json

```

## Project Structure

After running the pipeline, your directory will look like this:

```text
├── main.py                 # The main script
├── README.md               # This file
├── data/
│   ├── raw/                # Raw API responses
│   └── processed/          # Cleaned CSV files
└── artifacts/
    ├── figures/            # Generated charts (.png)
    └── reports/            # Markdown reports and metadata
