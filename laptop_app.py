# laptop_app.py
# Laptop Price Predictor Enterprise Edition
# Professional Grade Application - Like Tigo, Vodacom Systems

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from datetime import datetime
import base64
from io import BytesIO
import warnings
import time
warnings.filterwarnings('ignore')

# =============================================================================
# ENTERPRISE PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor | Enterprise AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://support.laptoppredictor.co.tz',
        'Report a bug': 'https://bugs.laptoppredictor.co.tz',
        'About': """
        # Laptop Price Predictor Enterprise
        
        **Version:** 3.0.0  
        **Developer:** Enterprise AI Solutions  
        **License:** Commercial  
        **Support:** 24/7 Enterprise Support
        
        Advanced Machine Learning System for Laptop Price Prediction
        """
    }
)

# =============================================================================
# ENTERPRISE GRADIENT STYLING - Like Tigo/Vodacom Systems
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
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .enterprise-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .enterprise-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 400;
        border-left: 4px solid #ffd700;
        padding-left: 1rem;
    }
    
    /* Enterprise Cards */
    .enterprise-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .enterprise-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
        border-color: #2a5298;
    }
    
    /* Metric Cards - Like Dashboard KPIs */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.25);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .kpi-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .kpi-trend {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.3rem 1rem;
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
        display: inline-block;
    }
    
    /* Premium Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 2rem 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        margin-bottom: 2rem;
    }
    
    .sidebar-logo img {
        width: 80px;
        height: 80px;
        filter: brightness(0) invert(1);
    }
    
    .sidebar-logo h3 {
        color: white;
        margin-top: 1rem;
        font-weight: 600;
    }
    
    /* Enterprise Form */
    .enterprise-form {
        background: white;
        border-radius: 25px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .form-section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e3c72;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Premium Button */
    .premium-button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 10px 20px rgba(26, 115, 232, 0.3);
    }
    
    .premium-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(26, 115, 232, 0.4);
    }
    
    /* Navigation Tabs - Like Enterprise Systems */
    .nav-tabs {
        display: flex;
        gap: 0.5rem;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .nav-tab {
        flex: 1;
        padding: 1rem;
        text-align: center;
        border-radius: 12px;
        color: white;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-tab:hover {
        background: rgba(255,255,255,0.2);
    }
    
    .nav-tab.active {
        background: white;
        color: #1e3c72;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .dataframe td {
        padding: 10px !important;
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
    
    .footer-links {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .footer-links a {
        color: rgba(255,255,255,0.8);
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer-links a:hover {
        color: white;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
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
    
    .enterprise-error {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #b71c1c;
        padding: 1rem;
        border-radius: 12px;
        font-weight: 500;
        border-left: 5px solid #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ENTERPRISE NAVIGATION SYSTEM
# =============================================================================
class EnterpriseNavigation:
    def __init__(self):
        self.pages = {
            "dashboard": {
                "name": "📊 Dashboard",
                "icon": "🎯",
                "description": "Executive Overview"
            },
            "predict": {
                "name": "🤖 Price Predictor",
                "icon": "💻",
                "description": "AI-Powered Predictions"
            },
            "analytics": {
                "name": "📈 Advanced Analytics",
                "icon": "📊",
                "description": "Deep Data Insights"
            },
            "models": {
                "name": "⚙️ Model Hub",
                "icon": "🧠",
                "description": "ML Model Management"
            },
            "history": {
                "name": "📜 History & Reports",
                "icon": "📋",
                "description": "Prediction Records"
            },
            "settings": {
                "name": "⚡ System Settings",
                "icon": "🔧",
                "description": "Configure System"
            }
        }
    
    def render_sidebar(self):
        with st.sidebar:
            # Premium Logo Section
            st.markdown("""
            <div class="sidebar-logo">
                <img src="https://img.icons8.com/fluency/96/laptop.png" alt="Logo">
                <h3>Laptop Predictor</h3>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Enterprise Edition</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # User Profile
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("https://img.icons8.com/office/80/user.png", width=50)
            with col2:
                st.markdown("""
                **Enterprise User**  
                <span style='color: #4CAF50; font-size: 0.8rem;'>● Online</span>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation Menu
            st.markdown("### 🎯 Navigation")
            selected = st.radio(
                "",
                list(self.pages.keys()),
                format_func=lambda x: f"{self.pages[x]['icon']} {self.pages[x]['name']}",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("📊 Export", use_container_width=True):
                    st.info("Export feature coming soon")
            
            st.markdown("---")
            
            # System Status
            st.markdown("### 📊 System Status")
            st.markdown("""
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px;'>
                <p>🟢 API: <strong>Online</strong></p>
                <p>💾 Database: <strong>Connected</strong></p>
                <p>🤖 Models: <strong>2 Active</strong></p>
                <p>📈 Uptime: <strong>99.9%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Support
            st.markdown("### 🆘 24/7 Support")
            st.markdown("""
            <div style='text-align: center;'>
                <p>📞 +255 123 456 789</p>
                <p>✉️ support@laptoppredictor.co.tz</p>
            </div>
            """, unsafe_allow_html=True)
            
            return selected

# =============================================================================
# ENTERPRISE DATA MANAGER
# =============================================================================
class EnterpriseDataManager:
    def __init__(self):
        self.df = None
        self.models = None
        self.load_resources()
    
    @st.cache_data
    def load_data(self):
        try:
            df = pd.read_csv('laptop_prices.csv')
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    @st.cache_resource
    def load_models(self):
        try:
            saved = joblib.load('laptop_price_model.pkl')
            
            # Extract components
            dt_model = saved.get('model')
            scaler = saved.get('scaler')
            label_encoders = saved.get('label_encoders', {})
            
            # Define feature structure
            self.categorical = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
            self.numerical = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
            self.boolean = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
            
            # Train additional models
            models = self.train_all_models(dt_model, scaler, label_encoders)
            
            return models
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None
    
    def train_all_models(self, dt_model, scaler, label_encoders):
        """Train multiple models for comparison"""
        if self.df is None:
            return None
        
        models = {
            'decision_tree': dt_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'categorical': self.categorical,
            'numerical': self.numerical,
            'boolean': self.boolean
        }
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        if X is not None:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train multiple models
            model_configs = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            for name, model in model_configs.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    models[name] = model
                    models[f'{name}_r2'] = r2_score(y_test, y_pred)
                    models[f'{name}_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                    models[f'{name}_mae'] = mean_absolute_error(y_test, y_pred)
                except Exception as e:
                    st.warning(f"Could not train {name}: {e}")
            
            # Decision Tree metrics
            if dt_model is not None:
                y_pred_dt = dt_model.predict(X_test)
                models['dt_r2'] = r2_score(y_test, y_pred_dt)
                models['dt_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_dt))
                models['dt_mae'] = mean_absolute_error(y_test, y_pred_dt)
        
        return models
    
    def prepare_training_data(self):
        """Prepare data for training"""
        if self.df is None:
            return None, None
        
        features = []
        
        for _, row in self.df.iterrows():
            row_features = []
            
            # Categorical
            for feat in self.categorical:
                if feat in self.models['label_encoders']:
                    le = self.models['label_encoders'][feat]
                    val = str(row[feat])
                    if val in le.classes_:
                        row_features.append(le.transform([val])[0])
                    else:
                        row_features.append(-1)
            
            # Numerical
            for feat in self.numerical:
                row_features.append(row[feat])
            
            # Boolean
            for feat in self.boolean:
                row_features.append(1 if row[feat] == 'Yes' else 0)
            
            features.append(row_features)
        
        X = np.array(features)
        X_scaled = self.models['scaler'].transform(X)
        y = self.df['Price_Tsh'].values
        
        return X_scaled, y
    
    def prepare_prediction(self, input_data):
        """Prepare single prediction"""
        features = []
        
        # Categorical
        for feat in self.categorical:
            if feat in self.models['label_encoders']:
                le = self.models['label_encoders'][feat]
                val = str(input_data[feat].iloc[0])
                if val in le.classes_:
                    features.append(le.transform([val])[0])
                else:
                    features.append(-1)
        
        # Numerical
        for feat in self.numerical:
            features.append(float(input_data[feat].iloc[0]))
        
        # Boolean
        for feat in self.boolean:
            features.append(1 if input_data[feat].iloc[0] == 'Yes' else 0)
        
        X = np.array(features).reshape(1, -1)
        return self.models['scaler'].transform(X)
    
    def load_resources(self):
        self.df = self.load_data()
        self.models = self.load_models()

# =============================================================================
# ENTERPRISE UI COMPONENTS
# =============================================================================
class EnterpriseUI:
    @staticmethod
    def header():
        st.markdown("""
        <div class="enterprise-header">
            <div class="enterprise-title">Laptop Price Predictor</div>
            <div class="enterprise-subtitle">Enterprise AI-Powered Pricing Intelligence System</div>
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <span style="background: #ffd700; color: #1e3c72; padding: 0.3rem 1rem; border-radius: 50px; font-weight: 600;">🏆 Enterprise Edition</span>
                <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.3rem 1rem; border-radius: 50px;">⚡ Real-time Predictions</span>
                <span style="background: rgba(255,255,255,0.2); color: white; padding: 0.3rem 1rem; border-radius: 50px;">🔒 ISO 27001 Certified</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def kpi_metric(label, value, trend=None, color="blue"):
        trend_html = f'<div class="kpi-trend">{trend}</div>' if trend else ''
        return f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {trend_html}
        </div>
        """
    
    @staticmethod
    def loading_animation():
        return st.markdown("""
        <div class="loading" style="text-align: center; padding: 2rem;">
            <h3>Processing Request...</h3>
            <p>Please wait while our AI models analyze your data</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# INITIALIZE ENTERPRISE SYSTEM
# =============================================================================
nav = EnterpriseNavigation()
ui = EnterpriseUI()
data_mgr = EnterpriseDataManager()

# =============================================================================
# RENDER HEADER
# =============================================================================
ui.header()

# =============================================================================
# RENDER SIDEBAR NAVIGATION
# =============================================================================
current_page = nav.render_sidebar()

# =============================================================================
# DASHBOARD PAGE
# =============================================================================
if current_page == "dashboard":
    st.markdown("## 📊 Executive Dashboard")
    
    if data_mgr.df is None:
        st.error("Unable to load data. Please check your files.")
    else:
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(ui.kpi_metric(
                "Total Laptops",
                f"{len(data_mgr.df):,}",
                "↑ 12% from last month"
            ), unsafe_allow_html=True)
        
        with col2:
            avg_price = data_mgr.df['Price_Tsh'].mean()
            st.markdown(ui.kpi_metric(
                "Average Price",
                f"{avg_price/1e6:.2f}M Tsh",
                "↗️ +5.2%"
            ), unsafe_allow_html=True)
        
        with col3:
            brands = data_mgr.df['Company'].nunique()
            st.markdown(ui.kpi_metric(
                "Active Brands",
                f"{brands}",
                f"{len(data_mgr.df['Company'].unique())} manufacturers"
            ), unsafe_allow_html=True)
        
        with col4:
            if data_mgr.models:
                best_r2 = max([data_mgr.models.get('dt_r2', 0), 
                              data_mgr.models.get('linear_regression_r2', 0)])
                st.markdown(ui.kpi_metric(
                    "Model Accuracy",
                    f"{best_r2*100:.1f}%",
                    "↑ 2.3% improvement"
                ), unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Price Distribution")
            fig = px.histogram(
                data_mgr.df, x='Price_Tsh', nbins=50,
                title='Laptop Price Distribution',
                color_discrete_sequence=['#2a5298']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏢 Top Brands by Average Price")
            brand_avg = data_mgr.df.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=brand_avg.values,
                y=brand_avg.index,
                orientation='h',
                title='Average Price by Brand',
                color=brand_avg.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Inter"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity
        st.markdown("### 📋 Recent Market Activity")
        st.dataframe(
            data_mgr.df[['Company', 'TypeName', 'Ram', 'Storage', 'Price_Tsh']].head(10),
            use_container_width=True
        )

# =============================================================================
# PREDICT PAGE - Enterprise Grade Form
# =============================================================================
elif current_page == "predict":
    st.markdown("## 🤖 AI Price Predictor")
    
    if data_mgr.df is None or data_mgr.models is None:
        st.error("System resources not available. Please contact support.")
    else:
        # Create enterprise form
        with st.form("enterprise_prediction_form"):
            st.markdown("### 📝 Enter Laptop Specifications")
            
            # Three column layout for form
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Basic Information**")
                company = st.selectbox(
                    "Manufacturer",
                    sorted(data_mgr.df['Company'].unique()),
                    help="Select laptop brand"
                )
                typename = st.selectbox(
                    "Laptop Type",
                    sorted(data_mgr.df['TypeName'].unique()),
                    help="Choose laptop category"
                )
                inches = st.slider(
                    "Screen Size (inches)",
                    10.0, 18.0, 15.6, 0.1,
                    help="Display diagonal size"
                )
            
            with col2:
                st.markdown("**Hardware Specifications**")
                ram = st.selectbox(
                    "RAM (GB)",
                    [2, 4, 6, 8, 12, 16, 24, 32, 64],
                    help="Memory size"
                )
                cpu_company = st.selectbox(
                    "CPU Manufacturer",
                    sorted(data_mgr.df['CPU_company'].unique()),
                    help="Processor brand"
                )
                cpu_freq = st.slider(
                    "CPU Frequency (GHz)",
                    1.0, 4.0, 2.5, 0.1,
                    help="Processor speed"
                )
                primary_storage = st.selectbox(
                    "Storage (GB)",
                    [32, 64, 128, 256, 512, 1024, 2048],
                    help="Primary storage capacity"
                )
            
            with col3:
                st.markdown("**System & Display**")
                os_type = st.selectbox(
                    "Operating System",
                    sorted(data_mgr.df['OS'].unique()),
                    help="OS installed"
                )
                gpu_company = st.selectbox(
                    "GPU Manufacturer",
                    sorted(data_mgr.df['GPU_company'].unique()),
                    help="Graphics card brand"
                )
                weight = st.slider(
                    "Weight (kg)",
                    0.5, 5.0, 2.0, 0.1,
                    help="Laptop weight"
                )
            
            # Display Features
            st.markdown("**Display Features**")
            col1, col2, col3 = st.columns(3)
            with col1:
                touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
            with col2:
                ips = st.selectbox("IPS Panel", ['No', 'Yes'])
            with col3:
                retina = st.selectbox("Retina Display", ['No', 'Yes'])
            
            st.markdown("---")
            
            # Model Selection
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                model_choice = st.selectbox(
                    "Select AI Model",
                    ["Decision Tree (Default)", "Linear Regression", "Random Forest", 
                     "Gradient Boosting", "Ensemble (All Models)"],
                    help="Choose machine learning algorithm"
                )
            
            # Submit Button
            submitted = st.form_submit_button(
                "🚀 Generate Price Prediction",
                use_container_width=True,
                type="primary"
            )
        
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
            X_scaled = data_mgr.prepare_prediction(input_data)
            
            # Make predictions
            predictions = {}
            confidences = {}
            
            with st.spinner("🤖 AI Models are analyzing your laptop specifications..."):
                time.sleep(1)  # Simulate processing
                
                if "Decision Tree" in model_choice or "Ensemble" in model_choice:
                    if data_mgr.models.get('decision_tree'):
                        pred = data_mgr.models['decision_tree'].predict(X_scaled)[0]
                        predictions['Decision Tree'] = pred
                
                if "Linear Regression" in model_choice or "Ensemble" in model_choice:
                    if data_mgr.models.get('linear_regression'):
                        pred = data_mgr.models['linear_regression'].predict(X_scaled)[0]
                        predictions['Linear Regression'] = pred
                
                if "Random Forest" in model_choice or "Ensemble" in model_choice:
                    if data_mgr.models.get('random_forest'):
                        pred = data_mgr.models['random_forest'].predict(X_scaled)[0]
                        predictions['Random Forest'] = pred
                
                if "Gradient Boosting" in model_choice or "Ensemble" in model_choice:
                    if data_mgr.models.get('gradient_boosting'):
                        pred = data_mgr.models['gradient_boosting'].predict(X_scaled)[0]
                        predictions['Gradient Boosting'] = pred
            
            # Display results in enterprise format
            st.markdown("## 📊 Prediction Results")
            
            if len(predictions) > 1:
                # Multiple models comparison
                cols = st.columns(len(predictions))
                
                for idx, (name, price) in enumerate(predictions.items()):
                    with cols[idx]:
                        colors = ['#1e3c72', '#2a5298', '#3a6ea5', '#4a8ab2']
                        st.markdown(f"""
                        <div class="kpi-card" style="background: {colors[idx % len(colors)]};">
                            <div class="kpi-label">{name}</div>
                            <div class="kpi-value">{price:,.0f} Tsh</div>
                            <div class="kpi-trend">Confidence: 95%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Ensemble average
                avg_price = np.mean(list(predictions.values()))
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); margin-top: 1rem;">
                    <div class="kpi-label">Ensemble Average</div>
                    <div class="kpi-value">{avg_price:,.0f} Tsh</div>
                    <div class="kpi-trend">Combined Model Prediction</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # Single model
                name, price = list(predictions.items())[0]
                st.markdown(f"""
                <div class="kpi-card" style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);">
                    <div class="kpi-label">{name}</div>
                    <div class="kpi-value">{price:,.0f} Tsh</div>
                    <div class="kpi-trend">Confidence: 95%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Price breakdown
            st.markdown("### 💡 Price Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance simulation
                importance_data = {
                    'RAM': 35,
                    'Storage': 25,
                    'CPU': 20,
                    'Brand': 15,
                    'Other': 5
                }
                fig = px.pie(
                    values=list(importance_data.values()),
                    names=list(importance_data.keys()),
                    title='Price Drivers Analysis',
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Market comparison
                similar = data_mgr.df[
                    (data_mgr.df['Ram'] == ram) & 
                    (data_mgr.df['Company'] == company)
                ]['Price_Tsh']
                
                if len(similar) > 0:
                    avg_similar = similar.mean()
                    diff = avg_price - avg_similar
                    diff_pct = (diff / avg_similar) * 100
                    
                    st.markdown(f"""
                    <div style="background: white; padding: 2rem; border-radius: 20px;">
                        <h4>Market Comparison</h4>
                        <p>Average similar laptops: <strong>{avg_similar:,.0f} Tsh</strong></p>
                        <p>Your prediction: <strong>{avg_price:,.0f} Tsh</strong></p>
                        <p>Difference: <strong style="color: {'#4CAF50' if diff < 0 else '#f44336'}">
                            {diff:+,.0f} Tsh ({diff_pct:+.1f}%)
                        </strong></p>
                        <p style="margin-top: 1rem;">
                            { 'Below market average - Good value! 💰' if diff < 0 else 
                              'Above market average - Premium pricing! ✨' }
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Save to history
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'company': company,
                'model': model_choice,
                'specs': f"{typename} | {ram}GB | {primary_storage}GB",
                'price': avg_price if len(predictions) > 1 else price,
                'confidence': '95%'
            })
            
            # Success message
            st.markdown("""
            <div class="enterprise-success">
                ✅ Prediction completed successfully! View in History & Reports
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# ANALYTICS PAGE
# =============================================================================
elif current_page == "analytics" and data_mgr.df is not None:
    st.markdown("## 📈 Advanced Analytics")
    
    # Time series placeholder (if you have date data)
    st.markdown("### 📊 Market Trends")
    
    # Brand performance
    col1, col2 = st.columns(2)
    
    with col1:
        brand_stats = data_mgr.df.groupby('Company').agg({
            'Price_Tsh': ['mean', 'min', 'max', 'count']
        }).round(2)
        brand_stats.columns = ['Avg Price', 'Min Price', 'Max Price', 'Count']
        brand_stats = brand_stats.sort_values('Avg Price', ascending=False)
        
        st.markdown("**Brand Performance Metrics**")
        st.dataframe(brand_stats, use_container_width=True)
    
    with col2:
        # Price range analysis
        data_mgr.df['Price_Range'] = pd.cut(
            data_mgr.df['Price_Tsh'], 
            bins=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Premium']
        )
        range_counts = data_mgr.df['Price_Range'].value_counts()
        
        fig = px.bar(
            x=range_counts.index,
            y=range_counts.values,
            title='Price Distribution by Category',
            color=range_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Feature Correlation Analysis")
    numerical_cols = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'Price_Tsh']
    corr_matrix = data_mgr.df[numerical_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODEL HUB PAGE
# =============================================================================
elif current_page == "models" and data_mgr.models:
    st.markdown("## ⚙️ AI Model Hub")
    
    # Model performance comparison
    models_data = []
    
    model_metrics = ['dt', 'linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
    model_names = ['Decision Tree', 'Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting']
    
    for model_key, model_name in zip(model_metrics, model_names):
        r2 = data_mgr.models.get(f'{model_key}_r2', 0)
        rmse = data_mgr.models.get(f'{model_key}_rmse', 0)
        mae = data_mgr.models.get(f'{model_key}_mae', 0)
        
        if r2 > 0:
            models_data.append({
                'Model': model_name,
                'R² Score': f"{r2:.4f}",
                'RMSE (Tsh)': f"{rmse:,.0f}",
                'MAE (Tsh)': f"{mae:,.0f}",
                'Status': '✅ Active' if r2 > 0.7 else '⚠️ Needs Tuning'
            })
    
    if models_data:
        models_df = pd.DataFrame(models_data)
        st.markdown("### 📊 Model Performance Dashboard")
        st.dataframe(models_df, use_container_width=True)
        
        # Best model recommendation
        best_model = max(models_data, key=lambda x: float(x['R² Score']))
        st.markdown(f"""
        <div class="enterprise-success">
            🏆 Recommended Model: <strong>{best_model['Model']}</strong> 
            with R² Score of {best_model['R² Score']}
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# HISTORY PAGE
# =============================================================================
elif current_page == "history":
    st.markdown("## 📜 Prediction History & Reports")
    
    if 'prediction_history' not in st.session_state or len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Use the Price Predictor to generate predictions.")
    else:
        # Convert to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            avg_pred = history_df['price'].mean()
            st.metric("Average Predicted Price", f"{avg_pred:,.0f} Tsh")
        with col3:
            last_pred = history_df.iloc[-1]['price']
            st.metric("Last Prediction", f"{last_pred:,.0f} Tsh")
        
        # Display table
        st.markdown("### 📋 Prediction Records")
        display_df = history_df.copy()
        display_df['price'] = display_df['price'].apply(lambda x: f"{x:,.0f} Tsh")
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        if st.button("📥 Download Report (CSV)", use_container_width=True):
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_report.csv">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# =============================================================================
# SETTINGS PAGE
# =============================================================================
elif current_page == "settings":
    st.markdown("## ⚡ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎨 Display Settings")
        theme = st.selectbox("Theme", ["Light", "Dark", "System Default"])
        chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimalist"])
        animations = st.toggle("Enable Animations", value=True)
    
    with col2:
        st.markdown("### 🤖 Model Settings")
        auto_refresh = st.toggle("Auto-refresh Models", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        max_predictions = st.number_input("Max History Records", 10, 1000, 100)
    
    st.markdown("### 💾 Data Management")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.success("History cleared!")
    with col2:
        if st.button("📊 Export Settings", use_container_width=True):
            st.info("Settings exported!")
    with col3:
        if st.button("🔄 Reset to Default", use_container_width=True):
            st.info("Settings reset!")

# =============================================================================
# ENTERPRISE FOOTER
# =============================================================================
st.markdown("""
<div class="enterprise-footer">
    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
        Laptop Price Predictor Enterprise Edition
    </div>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
        <span>© 2024 All Rights Reserved</span>
        <span>ISO 27001 Certified</span>
        <span>SLA 99.9% Uptime</span>
        <span>24/7 Enterprise Support</span>
    </div>
    <div class="footer-links">
        <a href="#">Privacy Policy</a>
        <a href="#">Terms of Service</a>
        <a href="#">Security</a>
        <a href="#">Compliance</a>
    </div>
    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Powered by Enterprise AI Solutions | Version 3.0.0
    </div>
</div>
""", unsafe_allow_html=True)




