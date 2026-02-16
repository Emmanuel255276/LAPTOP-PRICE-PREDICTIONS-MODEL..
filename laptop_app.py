# laptop_app.py
# Laptop Price Predictor - DataPredict Exact Style

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import base64
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="DataPredict - Laptop Price Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS - Exact DataPredict Style
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Main Container */
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Header */
    .main-title {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title h1 {
        color: #1e293b;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-title h2 {
        color: #64748b;
        font-size: 1.2rem;
        font-weight: 400;
        margin-top: 0;
    }
    
    /* Dashboard Tabs - Exactly as in image */
    .dashboard-tabs {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    .tab-btn {
        background: none;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        color: #64748b;
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .tab-btn:hover {
        color: #2563eb;
        background-color: #f1f5f9;
    }
    
    .tab-btn.active {
        color: #2563eb;
        background-color: #e6f0ff;
    }
    
    /* Content Card */
    .content-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }
    
    /* Section Titles */
    .section-title {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Form Styling */
    .form-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .form-field {
        margin-bottom: 1rem;
    }
    
    .form-label {
        display: block;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Override Streamlit default styling */
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 0.5rem;
    }
    
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Predict Button - Exactly as in image */
    .predict-btn {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        width: 100%;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin: 1rem 0;
    }
    
    .predict-btn:hover {
        background-color: #1d4ed8;
    }
    
    /* Result Card - Exactly as in image */
    .result-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .result-label {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        color: #1e293b;
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-currency {
        color: #64748b;
        font-size: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        padding: 2rem 0;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    
    /* Analytics Cards */
    .stat-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
@st.cache_data
def load_data():
    """Load laptop data from CSV"""
    try:
        df = pd.read_csv('laptop_prices.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        saved_objects = joblib.load('laptop_price_model.pkl')
        
        dt_model = saved_objects.get('model')
        scaler = saved_objects.get('scaler')
        label_encoders = saved_objects.get('label_encoders', {})
        
        categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
        numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
        boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
        
        df = load_data()
        lr_model = None
        lr_r2 = 0
        lr_rmse = 0
        
        if df is not None and dt_model is not None:
            try:
                X_processed = []
                for idx, row in df.iterrows():
                    features = []
                    for feat in categorical_features:
                        if feat in label_encoders:
                            le = label_encoders[feat]
                            val = str(row[feat])
                            if val in le.classes_:
                                features.append(le.transform([val])[0])
                            else:
                                features.append(-1)
                    for feat in numerical_features:
                        features.append(row[feat])
                    for feat in boolean_features:
                        features.append(1 if row[feat] == 'Yes' else 0)
                    X_processed.append(features)
                
                X = np.array(X_processed)
                y = df['Price_Tsh'].values
                X_scaled = scaler.transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                
                y_pred = lr_model.predict(X_test)
                lr_r2 = r2_score(y_test, y_pred)
                lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            except Exception as e:
                pass
        
        dt_r2 = saved_objects.get('test_r2_score', 0)
        dt_rmse = saved_objects.get('test_rmse', 0)
        
        return {
            'decision_tree': dt_model,
            'linear_regression': lr_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'boolean_features': boolean_features,
            'dt_r2': dt_r2,
            'dt_rmse': dt_rmse,
            'lr_r2': lr_r2,
            'lr_rmse': lr_rmse
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def prepare_features(input_data, models):
    """Prepare features for prediction"""
    features = []
    for feature in models['categorical_features']:
        if feature in models['label_encoders']:
            le = models['label_encoders'][feature]
            val = str(input_data[feature].iloc[0])
            try:
                if val in le.classes_:
                    features.append(le.transform([val])[0])
                else:
                    features.append(-1)
            except:
                features.append(-1)
    
    for feature in models['numerical_features']:
        features.append(float(input_data[feature].iloc[0]))
    
    for feature in models['boolean_features']:
        features.append(1 if input_data[feature].iloc[0] == 'Yes' else 0)
    
    X = np.array(features).reshape(1, -1)
    X_scaled = models['scaler'].transform(X)
    return X_scaled

# =============================================================================
# INITIALIZATION
# =============================================================================
df = load_data()
models = load_models()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Predictions'

# =============================================================================
# MAIN CONTAINER
# =============================================================================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# =============================================================================
# HEADER - Exactly as in image
# =============================================================================
st.markdown("""
<div class="main-title">
    <h1>DataPredict</h1>
    <h2>Price Prediction of Laptop</h2>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# DASHBOARD TABS - Exactly as in image
# =============================================================================
col1, col2, col3, col4 = st.columns([1,1,1,4])
with col1:
    if st.button("📊 Predictions", use_container_width=True):
        st.session_state.current_tab = 'Predictions'
with col2:
    if st.button("📈 Analytics", use_container_width=True):
        st.session_state.current_tab = 'Analytics'
with col3:
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.current_tab = 'Settings'

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# PREDICTIONS TAB - Exactly as in image
# =============================================================================
if st.session_state.current_tab == 'Predictions':
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Laptop Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Enter the details to predict the price of a laptop:</p>', unsafe_allow_html=True)
    
    if df is not None:
        with st.form("prediction_form"):
            # Row 1: Brand and Processor
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="form-label">Brand</div>', unsafe_allow_html=True)
                company = st.selectbox("", sorted(df['Company'].unique()), label_visibility="collapsed")
            with col2:
                st.markdown('<div class="form-label">Processor</div>', unsafe_allow_html=True)
                cpu_company = st.selectbox("", sorted(df['CPU_company'].unique()), label_visibility="collapsed")
            
            # Row 2: RAM and Storage
            col3, col4 = st.columns(2)
            with col3:
                st.markdown('<div class="form-label">RAM (GB)</div>', unsafe_allow_html=True)
                ram = st.selectbox("", [2, 4, 6, 8, 12, 16, 24, 32, 64], label_visibility="collapsed")
            with col4:
                st.markdown('<div class="form-label">Storage (GB)</div>', unsafe_allow_html=True)
                primary_storage = st.selectbox("", [32, 64, 128, 256, 512, 1024, 2048], label_visibility="collapsed")
            
            # Row 3: Screen Size and Type
            col5, col6 = st.columns(2)
            with col5:
                st.markdown('<div class="form-label">Screen Size (Inches)</div>', unsafe_allow_html=True)
                inches = st.slider("", 10.0, 18.0, 15.6, 0.1, label_visibility="collapsed")
            with col6:
                st.markdown('<div class="form-label">Laptop Type</div>', unsafe_allow_html=True)
                typename = st.selectbox("", sorted(df['TypeName'].unique()), label_visibility="collapsed")
            
            # Row 4: Operating System
            st.markdown('<div class="form-label">Operating System</div>', unsafe_allow_html=True)
            os_type = st.selectbox("", sorted(df['OS'].unique()), label_visibility="collapsed")
            
            # Additional required features (hidden from UI but needed for model)
            weight = 2.0
            cpu_freq = 2.5
            gpu_company = st.selectbox("Graphics", sorted(df['GPU_company'].unique()), key="gpu")
            touchscreen = 'No'
            ips = 'No'
            retina = 'No'
            
            # Predict button
            submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)
        
        if submitted and models is not None:
            # Prepare input data
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [typename],
                'Inches': [inches],
                'Ram': [ram],
                'OS': [os_type],
                'Weight': [weight],
                'CPU_company': [cpu_company],
                'CPU_freq': [cpu_freq],
                'PrimaryStorage': [primary_storage],
                'GPU_company': [gpu_company],
                'Touchscreen': [touchscreen],
                'IPSpanel': [ips],
                'RetinaDisplay': [retina]
            })
            
            # Make prediction
            X_scaled = prepare_features(input_data, models)
            dt_pred = models['decision_tree'].predict(X_scaled)[0]
            
            # Display result - Exactly as in image (TZS instead of KES)
            st.markdown("""
            <div class="result-card">
                <div class="result-label">Estimated Price</div>
                <div class="result-value">{:,.0f}</div>
                <div class="result-currency">Tanzanian Shillings (TZS)</div>
            </div>
            """.format(dt_pred), unsafe_allow_html=True)
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'brand': company,
                'specs': f"{typename} | {ram}GB RAM | {primary_storage}GB",
                'price': f"{dt_pred:,.0f} TZS"
            }
            st.session_state.prediction_history.append(history_entry)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# ANALYTICS TAB
# =============================================================================
elif st.session_state.current_tab == 'Analytics' and df is not None:
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Analytics</div>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Laptops</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{df['Company'].nunique()}</div>
            <div class="stat-label">Brands</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_price = df['Price_Tsh'].mean() / 1_000_000
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{avg_price:.1f}M</div>
            <div class="stat-label">Avg Price</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{df['Ram'].mode()[0]}</div>
            <div class="stat-label">Common RAM</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price Distribution
    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.histogram(df, x='Price_Tsh', nbins=50, title='Price Distribution')
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SETTINGS TAB
# =============================================================================
elif st.session_state.current_tab == 'Settings' and models is not None:
    
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Settings & Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        st.metric("Decision Tree R²", f"{models['dt_r2']:.3f}")
        st.metric("Linear Regression R²", f"{models['lr_r2']:.3f}")
    
    with col2:
        st.markdown("### About")
        st.info(
            "This model predicts laptop prices based on specifications.\n\n"
            "• Trained on laptop_prices.csv\n"
            "• Uses Decision Tree algorithm\n"
            "• 13 features considered\n"
            "• Predictions in Tanzanian Shillings"
        )
    
    # History preview
    st.markdown("### Recent Predictions")
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history[-5:])
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("📥 Download Full History"):
            full_history = pd.DataFrame(st.session_state.prediction_history)
            csv = full_history.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Click to Download</a>'
            st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No predictions yet")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER - Exactly as in image
# =============================================================================
st.markdown("""
<div class="footer">
    © 2025 DataPredict. All rights reserved.
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close main-container






