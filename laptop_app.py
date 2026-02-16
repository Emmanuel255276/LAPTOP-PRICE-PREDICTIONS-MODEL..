# laptop_app.py
# Laptop Price Predictor Enterprise Edition - NO EMOJIS, PROFESSIONAL CLEAN VERSION

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
    page_icon=None,
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
    
    /* Enterprise Header - Poa Sana */
    .enterprise-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 30px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .enterprise-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .enterprise-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
        border-bottom: 2px solid #ffd700;
        display: inline-block;
        padding-bottom: 0.5rem;
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
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
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
        margin: 1rem 0;
    }
    
    /* Footer */
    .enterprise-footer {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 -10px 30px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .sidebar-logo img {
        width: 70px;
        height: 70px;
        filter: brightness(0) invert(1);
    }
    
    .sidebar-logo h3 {
        color: white;
        margin-top: 0.5rem;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Sidebar info cards - Blue Style */
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-info .label {
        font-size: 0.85rem;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 0.3rem;
    }
    
    .sidebar-info .value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .sidebar-info .small {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Sidebar phone - Gold Style */
    .sidebar-phone {
        background: #ffd700;
        color: #1e3c72;
        border-radius: 50px;
        padding: 0.8rem 1rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        font-size: 1.3rem;
        letter-spacing: 1px;
        box-shadow: 0 5px 15px rgba(255,215,0,0.3);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Navigation radio styling */
    .stRadio > div {
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stRadio label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 10px;
    }
    
    .stSlider > div > div > div {
        color: #1e3c72;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(26, 115, 232, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
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
    """Load only the base model"""
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

# =============================================================================
# LOAD DATA AND BASE MODEL
# =============================================================================
df = load_data()
base_model = load_base_model()

# =============================================================================
# ENTERPRISE HEADER - POA SANA
# =============================================================================
st.markdown("""
<div class="enterprise-header">
    <div class="enterprise-title">LAPTOP PREDICTOR</div>
    <div class="enterprise-subtitle">Enterprise AI-Powered Pricing Intelligence System | Made in Tanzania</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - Professional Navigation
# =============================================================================
with st.sidebar:
    # LOGO - Kutumia picha yako
    try:
        with open("pic.jpg", "rb") as f:  # Badilisha hapa
            logo_base64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 15px; margin-bottom: 1.5rem;">
            <img src="data:image/jpg;base64,{logo_base64}" width="80">
            <h3 style="color: white; margin-top: 0.5rem;">LAPTOP PREDICTOR</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">Enterprise Edition</p>
        </div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Picha haipatikani. Angalia kama faili ipo kwenye folder.")

    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Dashboard", "Price Predictor", "Analytics", "Model Hub", "History"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Information Section - Brands na Total Laptops (Blue Cards)
    if df is not None:
        total_brands = df['Company'].nunique()
        total_laptops = len(df)
        
        st.markdown(f"""
        <div class="sidebar-info">
            <div class="label">Total Laptop Brands</div>
            <div class="value">{total_brands}</div>
            <div class="small">manufacturers worldwide</div>
        </div>
        
        <div class="sidebar-info">
            <div class="label">Total Laptops in Database</div>
            <div class="value">{total_laptops:,}</div>
            <div class="small">unique models</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
 
    
    # Contact - Phone Number tu (0655540648)
    st.markdown("### 24/7 Support")
    st.markdown("""
    <div class="sidebar-phone">
        0655 540 648
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# DASHBOARD PAGE
# =============================================================================
if page == "Dashboard":
    st.markdown("## Executive Dashboard")
    
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
            avg_price = df['Price_Tsh'].mean() / 1_000_000
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Price</div>
                <div class="kpi-value">{avg_price:.1f}M</div>
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
            st.markdown("### Price Distribution")
            fig = px.histogram(df, x='Price_Tsh', nbins=50, title='Price Distribution')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Top Brands by Average Price")
            brand_avg = df.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=brand_avg.values, y=brand_avg.index, orientation='h', title='Average Price by Brand')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Data
        st.markdown("### Recent Laptop Models")
        st.dataframe(
            df[['Company', 'TypeName', 'Ram', 'PrimaryStorage', 'Price_Tsh']].head(10),
            use_container_width=True
        )

# =============================================================================
# PRICE PREDICTOR PAGE
# =============================================================================
elif page == "Price Predictor":
    st.markdown("## AI Price Predictor")
    
    if df is None or base_model is None:
        st.error("System resources not available. Please contact support.")
    else:
        # Define feature categories
        categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
        numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
        boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
        
        # Create form
        with st.form("prediction_form"):
            st.markdown("### Enter Laptop Specifications")
            
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
            
            submitted = st.form_submit_button("Predict Price", use_container_width=True)
        
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
            st.markdown("### Prediction Results")
            
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
            final_price = avg_price if len(predictions) > 1 else price
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'company': company,
                'model': model_choice,
                'specs': f"{typename} | {ram}GB | {primary_storage}GB",
                'price': final_price
            })
            
            st.markdown("""
            <div class="enterprise-success">
                Prediction saved to history.
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# ANALYTICS PAGE
# =============================================================================
elif page == "Analytics" and df is not None:
    st.markdown("## Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Brand Analysis"])
    
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
elif page == "Model Hub" and base_model:
    st.markdown("## AI Model Hub")
    
    st.markdown("### Model Performance")
    
    model_data = [{
        'Model': 'Decision Tree',
        'R² Score': f"{base_model['dt_r2']:.4f}",
        'RMSE': f"{base_model['dt_rmse']:,.0f} Tsh",
        'Status': 'Active'
    }]
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True)
    
    st.markdown("### Train Additional Models")
    if st.button("Train All Models"):
        with st.spinner("Training models... This may take a moment."):
            st.info("Models will be trained on-the-fly during prediction.")

# =============================================================================
# HISTORY PAGE
# =============================================================================
elif page == "History":
    st.markdown("## Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Use the Price Predictor to make predictions.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            avg_price = history_df['price'].mean()
            st.metric("Average Price", f"{avg_price:,.0f} Tsh")
        with col3:
            last_price = history_df.iloc[-1]['price']
            st.metric("Last Prediction", f"{last_price:,.0f} Tsh")
        
        # Table
        st.dataframe(history_df, use_container_width=True)
        
        # Download
        if st.button("Download History"):
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