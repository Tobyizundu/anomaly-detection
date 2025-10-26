# anomaly-detection
A professional web-based application for real-time anomaly detection on streaming time-series data. Built with Python, Streamlit, and advanced statistical algorithms.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Features

- **Real-Time Data Streaming**: Simulates continuous data feeds (stock prices, sensor readings)
- **Multiple Detection Algorithms**:
  - Z-Score Method (Statistical)
  - IQR Method (Interquartile Range)
  - Moving Average Method (Trend-based)
- **Interactive Visualization**: Beautiful Plotly charts with real-time updates
- **Configurable Parameters**: Adjust detection sensitivity and data characteristics
- **Professional UI/UX**: Responsive design with gradient cards and intuitive controls
- **Live Statistics**: Real-time metrics and anomaly tracking


Requirements

- Python 3.9 or higher
- Libraries (see `requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - plotly
  - scipy

Installation & Local Setup

1. Clone the Repository
```bash
git clone https://github.com/yourusername/anomaly-detection-app.git
cd anomaly-detection-app
```

2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser 
How It Works

### Data Generation

The app simulates two types of data streams:

1. **Stock Price Data**: Mimics stock market behavior with trends, volatility, and random spikes
2. **Sensor Reading Data**: Simulates IoT sensor data (e.g., temperature) with periodic patterns

### Anomaly Detection Algorithms

 1. Z-Score Method
- Detects data points that deviate significantly from the mean
- Formula: `z = (x - μ) / σ`
- Anomaly if `|z| > threshold` (default: 3)
- **Best for**: Normally distributed data

2. IQR (Interquartile Range) Method
- Uses quartiles to identify outliers
- Anomaly if `x < Q1 - k*IQR` or `x > Q3 + k*IQR`
- **Best for**: Skewed distributions, robust to outliers

3. Moving Average Method
- Compares current value to recent trend
- Detects sudden deviations from moving average
- **Best for**: Time-series with trends and seasonality

Architecture
```
app.py
├── DataGenerator: Simulates streaming data
├── AnomalyDetector: Implements detection algorithms
├── UI Components: Streamlit interface
└── Visualization: Plotly charts
```

## Usage Guide

### Starting the Stream

1. Select your **Data Type** (Stock Price or Sensor Reading)
2. Choose a **Detection Method**
3. Adjust **Algorithm Parameters** for sensitivity
4. Click ** Start** to begin streaming
5. Watch real-time anomaly detection in action!

### Customization

- **Algorithm Parameters**: Fine-tune detection sensitivity
- **Data Parameters**: Adjust base values, volatility, and anomaly injection rate
- **Pause/Resume**: Control the data stream
- **Reset**: Clear all data and start fresh




## Testing

The app includes:
- Built-in data validation
- Error handling for edge cases
- Responsive UI testing across devices
- Real-time performance optimization

## UI/UX Features

- **Gradient Cards**: Beautiful visual statistics
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Color-Coded Anomalies**: Red markers for easy identification
- **Interactive Charts**: Hover for detailed information
- **Intuitive Controls**: Easy-to-use sidebar configuration
- **Real-Time Updates**: Smooth, continuous streaming

## Configuration Options

### Data Types
- Stock Price: Simulates financial time-series
- Sensor Reading: Simulates IoT device data

### Detection Methods
- Z-Score: Threshold (1.0 - 5.0)
- IQR: Multiplier (0.5 - 3.0)
- Moving Average: Window size (5-50), Threshold (1.0-5.0)

### Data Parameters
- Base values
- Volatility/Noise level
- Anomaly injection rate

##  Performance

- Handles 200+ data points efficiently
- Real-time chart updates (10 FPS)
- Optimized memory usage with rolling window
- Responsive UI with < 100ms latency


