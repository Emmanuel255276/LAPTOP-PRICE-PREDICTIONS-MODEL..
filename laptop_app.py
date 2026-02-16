# laptop_app.py
# Laptop Price Predictor - Professional Edition (DataPredict Style)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
# PAGE CONFIGURATION - DataPredict Style
# =============================================================================
st.set_page_config(
    page_title="DataPredict - Laptop Price Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS - DataPredict Modern Design
# =============================================================================
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: #f5f7fb;
    }
    
    /* Header */
    .header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 0 0 2rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header p {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Dashboard Cards */
    .dashboard-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.03);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-title i {
        font-size: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        flex: 1;
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Navigation Tabs */
    .nav-tabs {
        display: flex;
        gap: 0.5rem;
        background: white;
        padding: 0.5rem;
        border-radius: 50px;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .nav-tab {
        flex: 1;
        padding: 0.8rem;
        text-align: center;
        border-radius: 50px;
        color: #666;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-tab.active {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    
    /* Form Styling */
    .form-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    
    .form-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1.5rem;
    }
    
    .form-group {
        margin-bottom: 1.2rem;
    }
    
    .form-label {
        font-weight: 500;
        color: #444;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    /* Button */
    .predict-btn {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .predict-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(30, 60, 114, 0.3);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin-top: 1.5rem;
    }
    
    .result-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-currency {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 0.9rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        margin-top: 3rem;
    }
    
    /* Sidebar (hidden by default) */
    .sidebar-content {
        background: white;
        padding: 1rem;
        border-radius: 10px;
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
        # Load the saved model
        saved_objects = joblib.load('laptop_price_model.pkl')
        
        # Extract components
        dt_model = saved_objects.get('model')
        scaler = saved_objects.get('scaler')
        label_encoders = saved_objects.get('label_encoders', {})
        
        # Define feature categories - MUST MATCH EXACT ORDER FROM TRAINING
        categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
        numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
        boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
        
        all_features = categorical_features + numerical_features + boolean_features
        
        # Load data for Linear Regression training
        df = load_data()
        lr_model = None
        lr_r2 = 0
        lr_rmse = 0
        
        if df is not None and dt_model is not None:
            try:
                # Prepare features exactly as model expects
                X_processed = []
                for idx, row in df.iterrows():
                    features = []
                    
                    # Categorical features
                    for feat in categorical_features:
                        if feat in label_encoders:
                            le = label_encoders[feat]
                            val = str(row[feat])
                            if val in le.classes_:
                                features.append(le.transform([val])[0])
                            else:
                                features.append(-1)
                    
                    # Numerical features
                    for feat in numerical_features:
                        features.append(row[feat])
                    
                    # Boolean features
                    for feat in boolean_features:
                        features.append(1 if row[feat] == 'Yes' else 0)
                    
                    X_processed.append(features)
                
                X = np.array(X_processed)
                y = df['Price_Tsh'].values
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Train Linear Regression
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                lr_model = LinearRegression()
                lr_model.fit(X_train, y_train)
                
                # Calculate metrics
                y_pred = lr_model.predict(X_test)
                lr_r2 = r2_score(y_test, y_pred)
                lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
            except Exception as e:
                st.warning(f"Could not train Linear Regression: {e}")
        
        # Get Decision Tree metrics
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
            'all_features': all_features,
            'dt_r2': dt_r2,
            'dt_rmse': dt_rmse,
            'lr_r2': lr_r2,
            'lr_rmse': lr_rmse
        }
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def prepare_prediction_features(input_data, models):
    """Prepare features for prediction in EXACT order expected by model"""
    
    features = []
    
    # 1. Categorical features (in correct order)
    for feature in models['categorical_features']:
        if feature in models['label_encoders']:
            le = models['label_encoders'][feature]
            val = str(input_data[feature].iloc[0])
            try:
                if val in le.classes_:
                    encoded = le.transform([val])[0]
                else:
                    encoded = -1
            except:
                encoded = -1
            features.append(encoded)
    
    # 2. Numerical features (in correct order)
    for feature in models['numerical_features']:
        features.append(float(input_data[feature].iloc[0]))
    
    # 3. Boolean features (in correct order)
    for feature in models['boolean_features']:
        val = 1 if input_data[feature].iloc[0] == 'Yes' else 0
        features.append(val)
    
    # Convert to numpy array and reshape for single prediction
    X = np.array(features).reshape(1, -1)
    
    # Scale features
    X_scaled = models['scaler'].transform(X)
    
    return X_scaled

# =============================================================================
# INITIALIZATION
# =============================================================================
df = load_data()
models = load_models()

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Decision Tree'

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Predictions'

# =============================================================================
# HEADER - DataPredict Style
# =============================================================================
st.markdown("""
<div class="header">
    <h1>📊 DataPredict</h1>
    <p>Price Prediction of Laptop | Advanced AI-Powered Analytics</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# NAVIGATION TABS - DataPredict Style
# =============================================================================
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("📊 Predictions", use_container_width=True):
        st.session_state.current_page = 'Predictions'
with col2:
    if st.button("📈 Analytics", use_container_width=True):
        st.session_state.current_page = 'Analytics'
with col3:
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.current_page = 'Settings'
with col4:
    if st.button("📜 History", use_container_width=True):
        st.session_state.current_page = 'History'

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# =============================================================================
# PREDICTIONS PAGE - DataPredict Style (Main Focus)
# =============================================================================
if st.session_state.current_page == 'Predictions':
    
    # Model Selection Row
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        model_option = st.selectbox(
            "🤖 Select Model",
            ["Decision Tree", "Linear Regression", "Compare Both"],
            index=0
        )
        st.session_state.selected_model = model_option
    
    with col2:
        if models is not None:
            st.info(f"📊 Model R²: {models['dt_r2']:.3f} (DT) | {models['lr_r2']:.3f} (LR)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two Column Layout - Form and Result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📝 Enter Laptop Specifications</div>', unsafe_allow_html=True)
        
        if df is not None:
            with st.form("prediction_form"):
                # Row 1: Brand and Processor
                col_a, col_b = st.columns(2)
                with col_a:
                    company = st.selectbox("Brand", sorted(df['Company'].unique()))
                with col_b:
                    cpu_company = st.selectbox("Processor", sorted(df['CPU_company'].unique()))
                
                # Row 2: RAM and Storage
                col_c, col_d = st.columns(2)
                with col_c:
                    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
                with col_d:
                    primary_storage = st.selectbox("Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048])
                
                # Row 3: Screen Size and Type
                col_e, col_f = st.columns(2)
                with col_e:
                    inches = st.slider("Screen Size (Inches)", 10.0, 18.0, 15.6, 0.1)
                with col_f:
                    typename = st.selectbox("Laptop Type", sorted(df['TypeName'].unique()))
                
                # Row 4: OS and Weight
                col_g, col_h = st.columns(2)
                with col_g:
                    os_type = st.selectbox("Operating System", sorted(df['OS'].unique()))
                with col_h:
                    weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
                
                # Row 5: GPU and CPU Frequency
                col_i, col_j = st.columns(2)
                with col_i:
                    gpu_company = st.selectbox("Graphics Card", sorted(df['GPU_company'].unique()))
                with col_j:
                    cpu_freq = st.slider("CPU Frequency (GHz)", 1.0, 4.0, 2.5, 0.1)
                
                # Row 6: Display Features
                col_k, col_l, col_m = st.columns(3)
                with col_k:
                    touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
                with col_l:
                    ips = st.selectbox("IPS Panel", ['No', 'Yes'])
                with col_m:
                    retina = st.selectbox("Retina Display", ['No', 'Yes'])
                
                # Submit Button
                submitted = st.form_submit_button("🚀 Predict Price", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)
        
        if submitted and models is not None and df is not None:
            # Create input dataframe
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
            X_scaled = prepare_prediction_features(input_data, models)
            
            # Make predictions
            predictions = {}
            
            if st.session_state.selected_model in ['Decision Tree', 'Compare Both']:
                if models['decision_tree'] is not None:
                    dt_pred = models['decision_tree'].predict(X_scaled)[0]
                    predictions['Decision Tree'] = dt_pred
            
            if st.session_state.selected_model in ['Linear Regression', 'Compare Both']:
                if models['linear_regression'] is not None:
                    lr_pred = models['linear_regression'].predict(X_scaled)[0]
                    predictions['Linear Regression'] = lr_pred
            
            # Display predictions in DataPredict style
            if st.session_state.selected_model == 'Compare Both' and len(predictions) == 2:
                # Two predictions
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                border-radius: 15px; padding: 1.5rem; color: white; text-align: center;">
                        <div style="font-size: 0.9rem; opacity: 0.9;">Decision Tree</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{predictions['Decision Tree']/1000:.0f}K</div>
                        <div style="font-size: 0.8rem;">Tsh</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_y:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                border-radius: 15px; padding: 1.5rem; color: white; text-align: center;">
                        <div style="font-size: 0.9rem; opacity: 0.9;">Linear Regression</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{predictions['Linear Regression']/1000:.0f}K</div>
                        <div style="font-size: 0.8rem;">Tsh</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                avg_price = np.mean(list(predictions.values()))
                st.markdown(f"""
                <div class="result-card" style="margin-top: 1rem;">
                    <div class="result-label">Estimated Price (Average)</div>
                    <div class="result-value">{avg_price:,.0f}</div>
                    <div class="result-currency">Tanzanian Shillings</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                model_name = list(predictions.keys())[0]
                price = predictions[model_name]
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Estimated Price ({model_name})</div>
                    <div class="result-value">{price:,.0f}</div>
                    <div class="result-currency">Tanzanian Shillings</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'company': company,
                'model': st.session_state.selected_model,
                'specs': f"{typename} | {ram}GB RAM | {primary_storage}GB SSD",
                'avg_price': np.mean(list(predictions.values())) if predictions else 0,
                'price_tz': f"{np.mean(list(predictions.values())):,.0f} Tsh"
            }
            st.session_state.prediction_history.append(history_entry)
            
            st.success("✅ Prediction saved to history!")
        
        elif not submitted:
            st.info("👆 Fill in the laptop specifications and click 'Predict Price'")
        
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# ANALYTICS PAGE
# =============================================================================
elif st.session_state.current_page == 'Analytics' and df is not None:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Data Analytics Dashboard</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📉 Correlations", "🏷️ Brand Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, x='Price_Tsh', nbins=50,
                title='Price Distribution',
                color_discrete_sequence=['#1e3c72']
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='Ram', nbins=20,
                title='RAM Distribution',
                color_discrete_sequence=['#2a5298']
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        numerical_cols = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'Price_Tsh']
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        brand_avg = df.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=brand_avg.values,
            y=brand_avg.index,
            orientation='h',
            title='Average Price by Brand',
            color=brand_avg.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SETTINGS PAGE
# =============================================================================
elif st.session_state.current_page == 'Settings' and models is not None:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙️ Settings & Model Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Model Performance")
        metrics_df = pd.DataFrame({
            'Model': ['Decision Tree', 'Linear Regression'],
            'R² Score': [f"{models['dt_r2']:.3f}", f"{models['lr_r2']:.3f}"],
            'RMSE (Tsh)': [f"{models['dt_rmse']:,.0f}", f"{models['lr_rmse']:,.0f}"]
        })
        st.dataframe(metrics_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Dataset Statistics")
        if df is not None:
            stats_df = pd.DataFrame({
                'Metric': ['Total Laptops', 'Average Price', 'Min Price', 'Max Price', 'Number of Brands'],
                'Value': [
                    len(df),
                    f"{df['Price_Tsh'].mean():,.0f} Tsh",
                    f"{df['Price_Tsh'].min():,.0f} Tsh",
                    f"{df['Price_Tsh'].max():,.0f} Tsh",
                    df['Company'].nunique()
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("### 🎯 Feature Importance (Decision Tree)")
    if models['decision_tree'] is not None:
        importance = models['decision_tree'].feature_importances_
        feature_names = (models['categorical_features'] + 
                        models['numerical_features'] + 
                        models['boolean_features'])
        
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            imp_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# HISTORY PAGE
# =============================================================================
elif st.session_state.current_page == 'History':
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📜 Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Go to the Predictions page to make predictions!")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        display_df = history_df[['timestamp', 'company', 'model', 'specs', 'price_tz']].copy()
        display_df.columns = ['Timestamp', 'Brand', 'Model Used', 'Specifications', 'Predicted Price']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">📥 Download History (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# FOOTER - DataPredict Style
# =============================================================================
st.markdown("""
<div class="footer">
    © 2025 DataPredict. All rights reserved. | Powered by Machine Learning | Made in Tanzania 🇹🇿
</div>
""", unsafe_allow_html=True)


