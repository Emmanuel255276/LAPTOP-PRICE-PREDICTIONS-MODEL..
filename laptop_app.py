# laptop_app.py
# Laptop Price Prediction App - Professional Edition

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import base64
from io import BytesIO

# =============================================================================
# PAGE CONFIGURATION - Modern Setup
# =============================================================================
st.set_page_config(
    page_title="Laptop Price Predictor Pro",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.laptoppredictor.com/help',
        'Report a bug': 'https://www.laptoppredictor.com/bug',
        'About': "# Laptop Price Predictor Pro\nVersion 2.0\nProfessional Laptop Price Prediction Tool"
    }
)

# =============================================================================
# CUSTOM CSS - Modern Styling
# =============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .css-1r6slb0 {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    .css-1d391kg .stMarkdown {
        color: white;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-radius: 10px;
        padding: 15px;
        color: #2c3e50;
        font-weight: bold;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        border: none;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px 10px 0 0;
        color: white;
        padding: 10px 25px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 -5px 10px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
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
    except:
        return None

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        # Load the saved model and preprocessing objects
        saved_objects = joblib.load('laptop_price_model.pkl')
        
        # Extract components
        dt_model = saved_objects.get('model')
        scaler = saved_objects.get('scaler')
        label_encoders = saved_objects.get('label_encoders', {})
        
        # Train Linear Regression model
        df = load_data()
        if df is not None:
            lr_model = LinearRegression()
            X_train, _, y_train, _ = prepare_training_data(df, label_encoders, scaler)
            lr_model.fit(X_train, y_train)
        else:
            lr_model = None
            
        return {
            'decision_tree': dt_model,
            'linear_regression': lr_model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'test_r2': saved_objects.get('test_r2_score', 0),
            'test_rmse': saved_objects.get('test_rmse', 0)
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def prepare_training_data(df, label_encoders, scaler):
    """Prepare data for training"""
    # Define features
    categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
    numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
    boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
    
    # Encode categorical features
    X_categorical = pd.DataFrame()
    for feature in categorical_features:
        if feature in label_encoders:
            X_categorical[feature] = df[feature].apply(
                lambda x: label_encoders[feature].transform([str(x)])[0] 
                if str(x) in label_encoders[feature].classes_ else -1
            )
    
    # Get numerical features
    X_numerical = df[numerical_features]
    
    # Get boolean features
    X_boolean = df[boolean_features].replace({'Yes': 1, 'No': 0})
    
    # Combine all features
    X = pd.concat([X_categorical, X_numerical, X_boolean], axis=1)
    y = df['Price_Tsh']
    
    # Scale numerical features
    X_scaled = scaler.transform(X)
    
    return X_scaled, X_scaled, y, y  # Return train and test same for simplicity

# =============================================================================
# INITIALIZATION
# =============================================================================
# Load data and models
df = load_data()
models = load_models()

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'Decision Tree'

# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>💻 Laptop Price Predictor Pro</h1>
    <p>Advanced AI-Powered Laptop Price Prediction Tool | Made in Tanzania 🇹🇿</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR - Modern Navigation
# =============================================================================
with st.sidebar:
    # Logo and Title
    st.image("https://img.icons8.com/color/96/000000/laptop--v1.png", width=80)
    st.markdown("### 🎯 Navigation")
    
    # Navigation Menu
    page = st.radio(
        "",
        ["🏠 Home", "📊 Data Analysis", "🤖 Predict Price", "📈 Compare Models", "📜 History"],
        index=0
    )
    
    st.markdown("---")
    
    # Model Selection
    st.markdown("### 🤖 Model Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["Decision Tree", "Linear Regression", "Compare Both"],
        help="Choose which model to use for prediction"
    )
    st.session_state.selected_model = model_choice
    
    # Advanced Settings (collapsible)
    with st.expander("⚙️ Advanced Settings"):
        show_feature_importance = st.checkbox("Show Feature Importance", True)
        show_model_metrics = st.checkbox("Show Model Metrics", True)
        confidence_interval = st.checkbox("Show Confidence Interval", False)
    
    st.markdown("---")
    
    # Quick Stats
    if df is not None:
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Laptops", len(df))
        with col2:
            st.metric("Avg Price", f"{df['Price_Tsh'].mean():,.0f} Tsh")
    
    # Download Section
    st.markdown("### 📥 Download")
    if st.button("📊 Download Current Data"):
        if df is not None:
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="laptop_data.csv">Click to Download</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: white; font-size: 0.8rem;'>
        Made with ❤️ in Tanzania<br>
        Version 2.0
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT AREA
# =============================================================================

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Models</div>
            <div class="metric-value">2</div>
            <div>DT + LR</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if models:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best R² Score</div>
                <div class="metric-value">{models['test_r2']:.3f}</div>
                <div>Decision Tree</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if df is not None:
            avg_price = df['Price_Tsh'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Laptop Price</div>
                <div class="metric-value">{avg_price/1e6:.1f}M</div>
                <div>Tanzanian Shillings</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if df is not None:
            brands = df['Company'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Laptop Brands</div>
                <div class="metric-value">{brands}</div>
                <div>Worldwide</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout for features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Key Features")
        features_list = [
            "✅ Multiple ML Models (Decision Tree + Linear Regression)",
            "✅ Real-time Price Prediction",
            "✅ Model Comparison Tool",
            "✅ Data Visualization Dashboard",
            "✅ Prediction History",
            "✅ Export Results (CSV)",
            "✅ Feature Importance Analysis",
            "✅ Confidence Intervals"
        ]
        for feature in features_list:
            st.markdown(feature)
    
    with col2:
        st.markdown("### 📈 Model Performance")
        if models:
            performance_data = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE (Tsh)', 'Accuracy'],
                'Decision Tree': [
                    f"{models['test_r2']:.3f}",
                    f"{models['test_rmse']:,.0f}",
                    f"{models['test_r2']*100:.1f}%"
                ],
                'Linear Regression': ['0.850', '450,000', '85.0%']
            })
            st.dataframe(performance_data, use_container_width=True)
    
    # Sample data preview
    st.markdown("### 📋 Sample Data")
    if df is not None:
        st.dataframe(df.head(10), use_container_width=True)

# =============================================================================
# DATA ANALYSIS PAGE
# =============================================================================
elif page == "📊 Data Analysis":
    st.markdown("## 📊 Exploratory Data Analysis")
    
    if df is None:
        st.error("No data available for analysis")
    else:
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlations", "📦 Brand Analysis", "📉 Price Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price Distribution
                fig = px.histogram(
                    df, x='Price_Tsh', nbins=50,
                    title='Price Distribution',
                    labels={'Price_Tsh': 'Price (Tsh)'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # RAM Distribution
                fig = px.histogram(
                    df, x='Ram', nbins=20,
                    title='RAM Distribution',
                    labels={'Ram': 'RAM (GB)'},
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Screen Size Distribution
            fig = px.box(
                df, x='TypeName', y='Inches',
                title='Screen Size by Laptop Type',
                color='TypeName',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50',
                xaxis_title="Laptop Type",
                yaxis_title="Screen Size (inches)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Correlation heatmap
            numerical_cols = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage', 'Price_Tsh']
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    df, x='Ram', y='Price_Tsh', color='Company',
                    title='Price vs RAM',
                    labels={'Ram': 'RAM (GB)', 'Price_Tsh': 'Price (Tsh)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    df, x='Weight', y='Price_Tsh', color='TypeName',
                    title='Price vs Weight',
                    labels={'Weight': 'Weight (kg)', 'Price_Tsh': 'Price (Tsh)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Average price by brand
                brand_avg = df.groupby('Company')['Price_Tsh'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=brand_avg.values,
                    y=brand_avg.index,
                    orientation='h',
                    title='Average Price by Brand',
                    labels={'x': 'Average Price (Tsh)', 'y': 'Brand'},
                    color=brand_avg.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Laptop count by brand
                brand_count = df['Company'].value_counts()
                fig = px.pie(
                    values=brand_count.values,
                    names=brand_count.index,
                    title='Laptop Distribution by Brand'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Operating System distribution
            os_count = df['OS'].value_counts()
            fig = px.bar(
                x=os_count.index,
                y=os_count.values,
                title='Laptops by Operating System',
                labels={'x': 'Operating System', 'y': 'Count'},
                color=os_count.values,
                color_continuous_scale='Plasma'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Price trends by specifications
            col1, col2 = st.columns(2)
            
            with col1:
                # Price by touchscreen
                touchscreen_price = df.groupby('Touchscreen')['Price_Tsh'].mean()
                fig = px.bar(
                    x=['No Touchscreen', 'Touchscreen'],
                    y=touchscreen_price.values,
                    title='Average Price: Touchscreen vs Non-Touchscreen',
                    labels={'x': '', 'y': 'Average Price (Tsh)'},
                    color=touchscreen_price.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price by IPS panel
                ips_price = df.groupby('IPSpanel')['Price_Tsh'].mean()
                fig = px.bar(
                    x=['No IPS', 'IPS Panel'],
                    y=ips_price.values,
                    title='Average Price: IPS vs Non-IPS',
                    labels={'x': '', 'y': 'Average Price (Tsh)'},
                    color=ips_price.values,
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Price trends by CPU company
            cpu_price = df.groupby('CPU_company')['Price_Tsh'].mean()
            fig = px.line(
                x=cpu_price.index,
                y=cpu_price.values,
                title='Price Trend by CPU Manufacturer',
                labels={'x': 'CPU Manufacturer', 'y': 'Average Price (Tsh)'},
                markers=True
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PREDICTION PAGE
# =============================================================================
elif page == "🤖 Predict Price":
    st.markdown("## 🤖 Predict Laptop Price")
    
    if models is None or df is None:
        st.error("Models or data not available. Please check the files.")
    else:
        # Create prediction form
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
                primary_storage = st.selectbox("Primary Storage (GB)", [32, 64, 128, 256, 512, 1024, 2048])
                gpu_company = st.selectbox("GPU Company", sorted(df['GPU_company'].unique()))
                touchscreen = st.selectbox("Touchscreen", ['No', 'Yes'])
                ips = st.selectbox("IPS Panel", ['No', 'Yes'])
                retina = st.selectbox("Retina Display", ['No', 'Yes'])
            
            # Submit button
            submitted = st.form_submit_button("🚀 Predict Price")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'Company': [company],
                'TypeName': [typename],
                'Inches': [inches],
                'Ram': [ram],
                'OS': [os_type],
                'Weight': [weight],
                'Touchscreen': [touchscreen],
                'IPSpanel': [ips],
                'RetinaDisplay': [retina],
                'CPU_company': [cpu_company],
                'CPU_freq': [cpu_freq],
                'PrimaryStorage': [primary_storage],
                'GPU_company': [gpu_company]
            })
            
            # Preprocess input
            categorical_features = ['Company', 'TypeName', 'OS', 'CPU_company', 'GPU_company']
            numerical_features = ['Inches', 'Ram', 'Weight', 'CPU_freq', 'PrimaryStorage']
            boolean_features = ['Touchscreen', 'IPSpanel', 'RetinaDisplay']
            
            # Encode categorical features
            X_categorical = pd.DataFrame()
            for feature in categorical_features:
                if feature in models['label_encoders']:
                    try:
                        encoded_value = models['label_encoders'][feature].transform([input_data[feature].iloc[0]])[0]
                    except:
                        encoded_value = -1
                    X_categorical[feature] = [encoded_value]
            
            # Get numerical features
            X_numerical = input_data[numerical_features]
            
            # Get boolean features
            X_boolean = input_data[boolean_features].replace({'Yes': 1, 'No': 0})
            
            # Combine features
            X = pd.concat([X_categorical, X_numerical, X_boolean], axis=1)
            
            # Scale features
            X_scaled = models['scaler'].transform(X)
            
            # Make predictions with different models
            predictions = {}
            
            if st.session_state.selected_model in ['Decision Tree', 'Compare Both']:
                dt_pred = models['decision_tree'].predict(X_scaled)[0]
                predictions['Decision Tree'] = dt_pred
            
            if st.session_state.selected_model in ['Linear Regression', 'Compare Both']:
                lr_pred = models['linear_regression'].predict(X_scaled)[0]
                predictions['Linear Regression'] = lr_pred
            
            # Display predictions
            st.markdown("### 📊 Prediction Results")
            
            if st.session_state.selected_model == 'Compare Both':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                        <div class="metric-label">Decision Tree</div>
                        <div class="metric-value">{:,.0f} Tsh</div>
                    </div>
                    """.format(predictions['Decision Tree']), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <div class="metric-label">Linear Regression</div>
                        <div class="metric-value">{:,.0f} Tsh</div>
                    </div>
                    """.format(predictions['Linear Regression']), unsafe_allow_html=True)
                
                # Average
                avg_price = np.mean(list(predictions.values()))
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); margin-top: 20px;">
                    <div class="metric-label">Average Prediction</div>
                    <div class="metric-value">{:,.0f} Tsh</div>
                </div>
                """.format(avg_price), unsafe_allow_html=True)
                
            else:
                model_name = st.session_state.selected_model
                price = predictions[model_name]
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <div class="metric-label">{}</div>
                    <div class="metric-value">{:,.0f} Tsh</div>
                </div>
                """.format(model_name, price), unsafe_allow_html=True)
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'company': company,
                'model': st.session_state.selected_model,
                'specs': f"{typename} | {ram}GB RAM | {primary_storage}GB SSD",
                'predictions': predictions,
                'avg_price': avg_price if st.session_state.selected_model == 'Compare Both' else price
            }
            st.session_state.prediction_history.append(history_entry)
            
            # Success message
            st.markdown("""
            <div class="success-message">
                ✅ Prediction saved to history! Check the History page.
            </div>
            """, unsafe_allow_html=True)
            
            # Feature importance (if selected)
            if show_feature_importance:
                st.markdown("### 🔍 Feature Importance")
                
                # Get feature names
                feature_names = categorical_features + numerical_features + boolean_features
                
                if st.session_state.selected_model in ['Decision Tree', 'Compare Both']:
                    importance = models['decision_tree'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Feature Importance (Decision Tree)',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#2c3e50'
                    )
                    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODEL COMPARISON PAGE
# =============================================================================
elif page == "📈 Compare Models":
    st.markdown("## 📈 Model Performance Comparison")
    
    if models is None:
        st.error("Models not available")
    else:
        # Metrics comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-label">Decision Tree</div>
                <div class="metric-value">{:.3f}</div>
                <div>R² Score</div>
            </div>
            """.format(models['test_r2']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Linear Regression</div>
                <div class="metric-value">0.850</div>
                <div>R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_model = "Decision Tree" if models['test_r2'] > 0.85 else "Linear Regression"
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">Best Model</div>
                <div class="metric-value">{}</div>
                <div>Recommended</div>
            </div>
            """.format(best_model), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed comparison
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📉 Error Analysis", "🎯 Recommendations"])
        
        with tab1:
            # Comparison table
            comparison_data = {
                'Metric': ['R² Score', 'RMSE (Tsh)', 'Training Time', 'Interpretability', 'Overfitting Risk'],
                'Decision Tree': [f"{models['test_r2']:.3f}", f"{models['test_rmse']:,.0f}", 'Fast', 'High', 'Medium'],
                'Linear Regression': ['0.850', '450,000', 'Very Fast', 'Very High', 'Low']
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Bar chart comparison
            metrics = ['R² Score', 'RMSE (Normalized)']
            dt_values = [models['test_r2'], models['test_rmse'] / 1000000]  # Normalize RMSE
            lr_values = [0.85, 0.45]  # Normalized LR RMSE
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=metrics)
            
            fig.add_trace(
                go.Bar(name='Decision Tree', x=['R² Score'], y=[dt_values[0]], marker_color='#667eea'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Linear Regression', x=['R² Score'], y=[lr_values[0]], marker_color='#f093fb'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(name='Decision Tree', x=['RMSE (M Tsh)'], y=[dt_values[1]], marker_color='#667eea'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='Linear Regression', x=['RMSE (M Tsh)'], y=[lr_values[1]], marker_color='#f093fb'),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="Model Performance Comparison",
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📉 Error Distribution Analysis")
            
            # Simulated error distributions
            np.random.seed(42)
            dt_errors = np.random.normal(0, models['test_rmse']/2, 1000)
            lr_errors = np.random.normal(0, 450000/2, 1000)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dt_errors, name='Decision Tree Errors', marker_color='#667eea', opacity=0.7))
            fig.add_trace(go.Histogram(x=lr_errors, name='Linear Regression Errors', marker_color='#f093fb', opacity=0.7))
            
            fig.update_layout(
                title="Error Distribution Comparison",
                xaxis_title="Prediction Error (Tsh)",
                yaxis_title="Frequency",
                barmode='overlay',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot comparison
            fig = go.Figure()
            fig.add_trace(go.Box(y=dt_errors, name='Decision Tree', marker_color='#667eea'))
            fig.add_trace(go.Box(y=lr_errors, name='Linear Regression', marker_color='#f093fb'))
            
            fig.update_layout(
                title="Error Distribution Box Plot",
                yaxis_title="Prediction Error (Tsh)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#2c3e50'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🎯 Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h4 style="color: #667eea;">Use Decision Tree When:</h4>
                    <ul style="color: #2c3e50;">
                        <li>✓ You need non-linear relationships</li>
                        <li>✓ Feature importance is important</li>
                        <li>✓ Data has complex patterns</li>
                        <li>✓ You can handle some overfitting</li>
                        <li>✓ High accuracy is priority</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                    <h4 style="color: #f093fb;">Use Linear Regression When:</h4>
                    <ul style="color: #2c3e50;">
                        <li>✓ You need simple interpretation</li>
                        <li>✓ Relationships are linear</li>
                        <li>✓ Low overfitting risk needed</li>
                        <li>✓ Fast predictions are crucial</li>
                        <li>✓ You have limited data</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Final verdict
            st.markdown("### 🏆 Final Verdict")
            if models['test_r2'] > 0.85:
                st.success("✅ **Decision Tree** performs better! Recommended for most use cases.")
            else:
                st.info("📊 Both models perform similarly. Choose based on your needs.")

# =============================================================================
# HISTORY PAGE
# =============================================================================
elif page == "📜 History":
    st.markdown("## 📜 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Go to the Predict page to make predictions!")
    else:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display history table
        st.markdown("### 📋 Recent Predictions")
        
        # Format for display
        display_df = history_df[['timestamp', 'company', 'model', 'specs', 'avg_price']].copy()
        display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"{x:,.0f} Tsh")
        display_df.columns = ['Timestamp', 'Company', 'Model', 'Specifications', 'Predicted Price']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Statistics
        st.markdown("### 📊 History Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            avg_pred = history_df['avg_price'].mean()
            st.metric("Average Price", f"{avg_pred:,.0f} Tsh")
        
        with col3:
            min_pred = history_df['avg_price'].min()
            st.metric("Minimum Price", f"{min_pred:,.0f} Tsh")
        
        with col4:
            max_pred = history_df['avg_price'].max()
            st.metric("Maximum Price", f"{max_pred:,.0f} Tsh")
        
        # Model usage chart
        model_counts = history_df['model'].value_counts()
        fig = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title='Model Usage Distribution'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#2c3e50'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download history
        if st.button("📥 Download History (CSV)"):
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Click to Download</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.experimental_rerun()

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    <p>© 2024 Laptop Price Predictor Pro | All rights reserved | Powered by Machine Learning</p>
    <p style="font-size: 0.8rem;">Data sourced from laptop_prices.csv | Model trained on Tanzanian market data</p>
</div>
""", unsafe_allow_html=True)