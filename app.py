import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from scipy import stats
import queue
import threading

# Page configuration
st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .normal-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    .stSelectbox, .stSlider {
        padding: 0.5rem 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Anomaly Detection Classes
class AnomalyDetector:
    """Base class for anomaly detection methods"""
    
    @staticmethod
    def z_score_detection(data, threshold=3):
        """Detect anomalies using Z-Score method"""
        if len(data) < 2:
            return np.array([False] * len(data))
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.array([False] * len(data))
        
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold
    
    @staticmethod
    def iqr_detection(data, multiplier=1.5):
        """Detect anomalies using IQR method"""
        if len(data) < 4:
            return np.array([False] * len(data))
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def moving_average_detection(data, window=20, threshold=2):
        """Detect anomalies using Moving Average method"""
        if len(data) < window:
            return np.array([False] * len(data))
        
        # Calculate moving average and standard deviation
        df = pd.DataFrame({'value': data})
        df['ma'] = df['value'].rolling(window=window, min_periods=1).mean()
        df['std'] = df['value'].rolling(window=window, min_periods=1).std()
        
        # Replace NaN std with 0
        df['std'].fillna(0, inplace=True)
        
        # Calculate deviation
        df['deviation'] = np.abs(df['value'] - df['ma'])
        
        # Detect anomalies
        anomalies = df['deviation'] > (threshold * df['std'])
        
        return anomalies.values

class DataGenerator:
    """Generate realistic time-series data with anomalies"""
    
    @staticmethod
    def generate_stock_data(timestamp, base_price=100, volatility=2, anomaly_prob=0.03):
        """Generate stock-like data with trend and volatility"""
        # Base trend with some random walk
        trend = np.sin(timestamp / 50) * 5
        noise = np.random.normal(0, volatility)
        
        value = base_price + trend + noise
        
        # Inject anomalies randomly
        if np.random.random() < anomaly_prob:
            spike = np.random.choice([-1, 1]) * np.random.uniform(10, 20)
            value += spike
        
        return max(value, 0.1)  # Ensure positive values
    
    @staticmethod
    def generate_sensor_data(timestamp, base_value=25, noise_level=0.5, anomaly_prob=0.05):
        """Generate sensor reading data (e.g., temperature)"""
        # Simulate daily cycle
        daily_cycle = np.sin(timestamp / 30) * 3
        noise = np.random.normal(0, noise_level)
        
        value = base_value + daily_cycle + noise
        
        # Inject anomalies
        if np.random.random() < anomaly_prob:
            spike = np.random.choice([-1, 1]) * np.random.uniform(5, 15)
            value += spike
        
        return value

# Initialize session state
def initialize_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame({
            'timestamp': [],
            'value': [],
            'anomaly': []
        })
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'data_type' not in st.session_state:
        st.session_state.data_type = "Stock Price"
    
    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = "Z-Score"
    
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    
    if 'total_points' not in st.session_state:
        st.session_state.total_points = 0
    
    if 'anomaly_count' not in st.session_state:
        st.session_state.anomaly_count = 0

def generate_new_datapoint():
    """Generate a new data point based on selected data type"""
    timestamp = st.session_state.counter
    
    if st.session_state.data_type == "Stock Price":
        value = DataGenerator.generate_stock_data(
            timestamp,
            base_price=st.session_state.get('base_price', 100),
            volatility=st.session_state.get('volatility', 2),
            anomaly_prob=st.session_state.get('anomaly_prob', 0.03)
        )
    else:  # Sensor Reading
        value = DataGenerator.generate_sensor_data(
            timestamp,
            base_value=st.session_state.get('base_value', 25),
            noise_level=st.session_state.get('noise_level', 0.5),
            anomaly_prob=st.session_state.get('anomaly_prob', 0.05)
        )
    
    st.session_state.counter += 1
    return timestamp, value

def detect_anomalies(values, method):
    """Detect anomalies using the selected method"""
    values_array = np.array(values)
    
    if method == "Z-Score":
        threshold = st.session_state.get('z_threshold', 3)
        return AnomalyDetector.z_score_detection(values_array, threshold)
    elif method == "IQR":
        multiplier = st.session_state.get('iqr_multiplier', 1.5)
        return AnomalyDetector.iqr_detection(values_array, multiplier)
    else:  # Moving Average
        window = st.session_state.get('ma_window', 20)
        threshold = st.session_state.get('ma_threshold', 2)
        return AnomalyDetector.moving_average_detection(values_array, window, threshold)

def create_chart(df):
    """Create an interactive Plotly chart"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Real-Time Data Stream with Anomaly Detection',)
    )
    
    if len(df) > 0:
        # Normal points
        normal_df = df[~df['anomaly']]
        fig.add_trace(
            go.Scatter(
                x=normal_df['timestamp'],
                y=normal_df['value'],
                mode='lines+markers',
                name='Normal Data',
                line=dict(color='#4facfe', width=2),
                marker=dict(size=6, color='#4facfe')
            )
        )
        
        # Anomaly points
        anomaly_df = df[df['anomaly']]
        if len(anomaly_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_df['timestamp'],
                    y=anomaly_df['value'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        size=12,
                        color='#f5576c',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                )
            )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Main Application
def main():
    initialize_session_state()
    
    # Header
    st.title("📊 Real-Time Anomaly Detection System")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Type Selection
        st.subheader("Data Source")
        data_type = st.selectbox(
            "Select Data Type",
            ["Stock Price", "Sensor Reading"],
            key="data_type_selector"
        )
        if data_type != st.session_state.data_type:
            st.session_state.data_type = data_type
        
        # Detection Method Selection
        st.subheader("Detection Method")
        method = st.selectbox(
            "Select Algorithm",
            ["Z-Score", "IQR", "Moving Average"],
            key="method_selector"
        )
        if method != st.session_state.detection_method:
            st.session_state.detection_method = method
        
        # Method-specific parameters
        st.subheader("Algorithm Parameters")
        
        if st.session_state.detection_method == "Z-Score":
            st.session_state.z_threshold = st.slider(
                "Z-Score Threshold",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Higher values = fewer anomalies detected"
            )
            st.info("**Z-Score Method**: Detects points that deviate significantly from the mean.")
        
        elif st.session_state.detection_method == "IQR":
            st.session_state.iqr_multiplier = st.slider(
                "IQR Multiplier",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Higher values = fewer anomalies detected"
            )
            st.info("**IQR Method**: Detects points outside the interquartile range.")
        
        else:  # Moving Average
            st.session_state.ma_window = st.slider(
                "Window Size",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="Number of points for moving average"
            )
            st.session_state.ma_threshold = st.slider(
                "Deviation Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Higher values = fewer anomalies detected"
            )
            st.info("**Moving Average**: Detects points deviating from recent trend.")
        
        # Data Generation Parameters
        st.subheader("Data Parameters")
        
        if st.session_state.data_type == "Stock Price":
            st.session_state.base_price = st.slider(
                "Base Price ($)",
                min_value=50,
                max_value=200,
                value=100,
                step=10
            )
            st.session_state.volatility = st.slider(
                "Volatility",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        else:
            st.session_state.base_value = st.slider(
                "Base Value (°C)",
                min_value=15,
                max_value=35,
                value=25,
                step=1
            )
            st.session_state.noise_level = st.slider(
                "Noise Level",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1
            )
        
        st.session_state.anomaly_prob = st.slider(
            "Anomaly Injection Rate",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Probability of injecting an anomaly"
        )
        
        st.markdown("---")
        
        # Control Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start" if not st.session_state.is_running else "⏸️ Pause", 
                        type="primary", use_container_width=True):
                st.session_state.is_running = not st.session_state.is_running
        
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.data = pd.DataFrame({
                    'timestamp': [],
                    'value': [],
                    'anomaly': []
                })
                st.session_state.counter = 0
                st.session_state.total_points = 0
                st.session_state.anomaly_count = 0
                st.session_state.is_running = False
                st.rerun()
        
        # About Section
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **Real-Time Anomaly Detection System**
            
            This application demonstrates real-time anomaly detection on streaming data using three different algorithms:
            
            - **Z-Score**: Statistical method using standard deviation
            - **IQR**: Interquartile range method
            - **Moving Average**: Trend-based detection
            
            Adjust parameters to see how they affect detection sensitivity.
            """)
    
    # Main Content Area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Total Points</h3>
            <h2>{st.session_state.total_points}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="anomaly-card">
            <h3>⚠️ Anomalies</h3>
            <h2>{st.session_state.anomaly_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        normal_count = st.session_state.total_points - st.session_state.anomaly_count
        st.markdown(f"""
        <div class="normal-card">
            <h3>✅ Normal Points</h3>
            <h2>{normal_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        anomaly_rate = (st.session_state.anomaly_count / st.session_state.total_points * 100) if st.session_state.total_points > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Anomaly Rate</h3>
            <h2>{anomaly_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart Container
    chart_placeholder = st.empty()
    
    # Data stream simulation
    if st.session_state.is_running:
        # Generate new data point
        timestamp, value = generate_new_datapoint()
        
        # Add to dataframe
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'value': [value],
            'anomaly': [False]  # Will be updated
        })
        
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        
        # Keep only last 200 points for performance
        if len(st.session_state.data) > 200:
            st.session_state.data = st.session_state.data.iloc[-200:].reset_index(drop=True)
        
        # Detect anomalies on all data
        if len(st.session_state.data) > 1:
            anomalies = detect_anomalies(
                st.session_state.data['value'].values,
                st.session_state.detection_method
            )
            st.session_state.data['anomaly'] = anomalies
        
        # Update statistics
        st.session_state.total_points += 1
        st.session_state.anomaly_count = st.session_state.data['anomaly'].sum()
        
        # Auto-rerun for continuous streaming
        time.sleep(0.1)  # Control speed
        st.rerun()
    
    # Display chart
    if len(st.session_state.data) > 0:
        fig = create_chart(st.session_state.data)
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        chart_placeholder.info("👆 Click 'Start' to begin data streaming...")
    
    # Recent Anomalies Table
    if st.session_state.anomaly_count > 0:
        st.markdown("### 🔴 Recent Anomalies")
        recent_anomalies = st.session_state.data[st.session_state.data['anomaly']].tail(10)
        
        if len(recent_anomalies) > 0:
            display_df = recent_anomalies[['timestamp', 'value']].copy()
            display_df['value'] = display_df['value'].round(2)
            display_df = display_df.rename(columns={
                'timestamp': 'Time Index',
                'value': 'Value'
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
