# laptop_app.py
# Laptop Price Predictor Enterprise Edition - FIXED VERSION
# No caching errors, optimized for Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import base64
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor | Enterprise AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Enterprise Styling
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Enterprise Header */
    .enterprise-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .enterprise-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .enterprise-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.25);
        margin-bottom: 1rem;
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    /* Enterprise Cards */
    .enterprise-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .enterprise-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    /* Success/Error Messages */
    .enterprise-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1e3c72;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 500;
        border-left: 5px solid #00c853;
    }
    
    /* Footer */
    .enterprise-footer {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING - WITHOUT CACHING ERRORS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data():
    """Load laptop data from CSV"""
    try:
        df = pd.read_csv('laptop_prices.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_base_model():
    """Load only the base model (not the full object)"""
    try:
        model_data = joblib.load('laptop_price_model.pkl')
        return {
            'model': model_data.get('model'),
            'scaler': model_data.get('scaler'),
            'label_encoders': model_data.get('label_encoders', {}),
            'dt_r2': model_data.get('test_r2_score', 0),
            'dt_rmse': model_data.get('test_rmse', 0)
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

if 'additional_models' not in st.session_state:
    st.session_state.additional_models = {}

# =============================================================================
# LOAD DATA AND BASE MODEL
# =============================================================================
df = load_data()
base_model = load_base_model()

# =============================================================================
# ENTERPRISE HEADER
# =============================================================================
st.markdown("""
<div class="enterprise-header">
    <div class="enterprise-title">Laptop Price Predictor</div>
    <div class="enterprise-subtitle">Enterprise AI-Powered Pricing Intelligence System | Made in Tanzania 🇹🇿</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - Professional Navigation
# =============================================================================
with st.sidebar:
    # Logo
    st.image("https://img.icons8.com/fluency/96/laptop.png", width=80)
    st.markdown("### Laptop Predictor")
    st.markdown("*Enterprise Edition v3.0*")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        [" Dashboard", " Price Predictor", " Analytics", " Model Hub", " History"],
        index=0
    )
    
    st.markdown("---")
    
   
    
    
    
    # Support
    st.markdown("### 24/7 Support")
    st.markdown("📞 +255 655 540 648")


# =============================================================================
# DASHBOARD PAGE
# =============================================================================
if page == "🏠 Dashboard":
    st.markdown("##  Executive Dashboard")
    
    if df is None:
        st.error("Unable to load data. Please check your files.")
    else:
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Laptops</div>
                <div class="kpi-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Price</div>
                <div class="kpi-value">{df['Price_Tsh'].mean()/1e6:.1f}M</div>
                <div>Tsh</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Active Brands</div>
                <div class="kpi-value">{df['Company'].nunique()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Model Accuracy</div>
                <div class="kpi-value">{base_model['dt_r2']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Price Distribution")
            fig = px.histogram(df, x='Price_Tsh', nbins=50, title='Price Distribution')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏢 Top Brands")
            brand_avg = df.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=brand_avg.values, y=brand_avg.index, orientation='h', title='Average Price by Brand')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PRICE PREDICTOR PAGE - FIXED
# =============================================================================
elif page == "🤖 Price Predictor":
    st.markdown("## 🤖 AI Price Predictor")
    
    if df is None or base_model is None:
        st.error("System resources not available. Please contact support.")
    else:
        # Define feature categories
        categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
        numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
        boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
        
        # Create form
        with st.form("prediction_form"):
            st.markdown("### 📝 Enter Laptop Specifications")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                company = st.selectbox("Company", sorted(df['Company'].unique()))
                typename = st.selectbox("Type", sorted(df['TypeName'].unique()))
                inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6, 0.1)
                ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
                
            with col2:
                os_type = st.selectbox("Operating System", sorted(df['OS'].unique()))
                weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
                cpu_company = st.selectbox("CPU Company", sorted(df['CPU_company'].unique()))
                cpu_freq = st.slider("CPU Frequency (GHz)", 1.0, 4.0, 2.5, 0.1)
                
            with col3:
                primary_storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048])
                gpu_company = st.selectbox("GPU Company", sorted(df['GPU_company'].unique()))
                touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
                ips = st.selectbox("IPS Panel", ['No', 'Yes'])
                retina = st.selectbox("Retina Display", ['No', 'Yes'])
            
            model_choice = st.selectbox(
                "Select AI Model",
                ["Decision Tree", "Linear Regression", "Random Forest", "Gradient Boosting", "Ensemble (All)"]
            )
            
            submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)
        
        if submitted:
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
            
            # Prepare features
            features = []
            
            # Categorical features
            for feat in categorical_features:
                if feat in base_model['label_encoders']:
                    le = base_model['label_encoders'][feat]
                    val = str(input_data[feat].iloc[0])
                    if val in le.classes_:
                        features.append(le.transform([val])[0])
                    else:
                        features.append(-1)
            
            # Numerical features
            for feat in numerical_features:
                features.append(float(input_data[feat].iloc[0]))
            
            # Boolean features
            for feat in boolean_features:
                features.append(1 if input_data[feat].iloc[0] == 'Yes' else 0)
            
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = base_model['scaler'].transform(X)
            
            # Make predictions
            predictions = {}
            
            with st.spinner("AI Models are analyzing your laptop..."):
                time.sleep(1)
                
                if model_choice in ["Decision Tree", "Ensemble (All)"]:
                    pred = base_model['model'].predict(X_scaled)[0]
                    predictions['Decision Tree'] = pred
                
                # Train additional models on the fly if needed
                if model_choice in ["Linear Regression", "Random Forest", "Gradient Boosting", "Ensemble (All)"]:
                    # Prepare training data
                    X_train_list = []
                    for _, row in df.iterrows():
                        row_features = []
                        for feat in categorical_features:
                            if feat in base_model['label_encoders']:
                                le = base_model['label_encoders'][feat]
                                val = str(row[feat])
                                if val in le.classes_:
                                    row_features.append(le.transform([val])[0])
                                else:
                                    row_features.append(-1)
                        for feat in numerical_features:
                            row_features.append(row[feat])
                        for feat in boolean_features:
                            row_features.append(1 if row[feat] == 'Yes' else 0)
                        X_train_list.append(row_features)
                    
                    X_train = np.array(X_train_list)
                    X_train_scaled = base_model['scaler'].transform(X_train)
                    y_train = df['Price_Tsh'].values
                    
                    # Train requested model
                    if model_choice in ["Linear Regression", "Ensemble (All)"]:
                        lr = LinearRegression()
                        lr.fit(X_train_scaled, y_train)
                        predictions['Linear Regression'] = lr.predict(X_scaled)[0]
                    
                    if model_choice in ["Random Forest", "Ensemble (All)"]:
                        rf = RandomForestRegressor(n_estimators=50, random_state=42)
                        rf.fit(X_train_scaled, y_train)
                        predictions['Random Forest'] = rf.predict(X_scaled)[0]
                    
                    if model_choice in ["Gradient Boosting", "Ensemble (All)"]:
                        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
                        gb.fit(X_train_scaled, y_train)
                        predictions['Gradient Boosting'] = gb.predict(X_scaled)[0]
            
            # Display results
            st.markdown("### 📊 Prediction Results")
            
            if len(predictions) > 1:
                cols = st.columns(len(predictions))
                colors = ['#1e3c72', '#2a5298', '#3a6ea5', '#4a8ab2']
                
                for idx, (name, price) in enumerate(predictions.items()):
                    with cols[idx]:
                        st.markdown(f"""
                        <div class="kpi-card" style="background: {colors[idx % len(colors)]};">
                            <div class="kpi-label">{name}</div>
                            <div class="kpi-value">{price:,.0f} Tsh</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Ensemble average
                avg_price = np.mean(list(predictions.values()))
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="kpi-label">Ensemble Average</div>
                    <div class="kpi-value">{avg_price:,.0f} Tsh</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                name, price = list(predictions.items())[0]
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{name}</div>
                    <div class="kpi-value">{price:,.0f} Tsh</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Save to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'company': company,
                'model': model_choice,
                'specs': f"{typename} | {ram}GB | {primary_storage}GB",
                'price': avg_price if len(predictions) > 1 else price
            })
            
            st.markdown("""
            <div class="enterprise-success">
                ✅ Prediction saved to history!
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# ANALYTICS PAGE
# =============================================================================
elif page == "📊 Analytics" and df is not None:
    st.markdown("## 📊 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📊 Brand Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Price_Tsh', nbins=50, title='Price Distribution')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x='Ram', nbins=20, title='RAM Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numerical_cols = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'Price_Tsh']
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        brand_stats = df.groupby('Company').agg({
            'Price_Tsh': ['mean', 'min', 'max', 'count']
        }).round(2)
        brand_stats.columns = ['Avg Price', 'Min Price', 'Max Price', 'Count']
        st.dataframe(brand_stats.sort_values('Avg Price', ascending=False), use_container_width=True)

# =============================================================================
# MODEL HUB PAGE
# =============================================================================
elif page == "⚙️ Model Hub" and base_model:
    st.markdown("## ⚙️ AI Model Hub")
    
    st.markdown("### 📊 Model Performance")
    
    model_data = [{
        'Model': 'Decision Tree',
        'R² Score': f"{base_model['dt_r2']:.4f}",
        'RMSE': f"{base_model['dt_rmse']:,.0f} Tsh",
        'Status': '✅ Active'
    }]
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    st.markdown("### 🚀 Train Additional Models")
    if st.button("Train All Models"):
        with st.spinner("Training models... This may take a moment."):
            st.info("Models will be trained on-the-fly during prediction.")

# =============================================================================
# HISTORY PAGE
# =============================================================================
elif page == "📜 History":
    st.markdown("## 📜 Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Use the Price Predictor to make predictions.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Average Price", f"{history_df['price'].mean():,.0f} Tsh")
        with col3:
            st.metric("Last Prediction", f"{history_df.iloc[-1]['price']:,.0f} Tsh")
        
        # Table
        st.dataframe(history_df, use_container_width=True)
        
        # Download
        if st.button("📥 Download History"):
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="enterprise-footer">
    <div>Laptop Price Predictor Enterprise Edition v3.0</div>
    <div style="margin-top: 1rem; font-size: 0.9rem;">
        © 2024 All Rights Reserved | Powered by Enterprise AI Solutions
    </div>
</div>
""", unsafe_allow_html=True)

            





