"""
Bank Customer Churn Prediction Dashboard
Author: Deep Learning Developer
Description: A production-ready Streamlit application that evaluates customer 
             attrition risk using an optimized multi-layer Artificial Neural Network.
"""

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================================
# 1. CORE PIPELINE & MODEL EXECUTION ENGINE
# =====================================================================

@st.cache_resource
def initialize_and_train_pipeline():
    """
    Executes the complete data engineering pipeline and trains the deep learning model.
    Results are cached natively by Streamlit to optimize server performance.
    
    Returns:
        scaler (StandardScaler): Fitted parameters for structural normalization.
        model (Sequential): Trained Keras Deep Neural Network configuration state.
    """
    # Load raw telemetry data
    df = pd.read_csv("Bank_Churn.csv")
    
    # Drop structural identifiers with zero feature importance variance
    df_clean = df.drop(['CustomerId', 'Surname'], axis=1)
    
    # Explicit mapping transformation for binary string metrics
    df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
    
    # One-Hot encoding evaluation to handle geographical categorical metrics safely
    df_numeric = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)

    # Segregate dependent target arrays and independent vectors
    X = df_numeric.drop('Exited', axis=1).astype(np.float32)
    y = df_numeric['Exited'].values
    num_features = X.shape[1]

    # Partition historical profiles into training and validation segments
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit standard scaling transforms against training distribution variance
    scaler_instance = StandardScaler()
    X_train_scaled = scaler_instance.fit_transform(X_train)

    # Multi-Layer Deep Perceptron structural implementation with dropout regularizations
    model = Sequential([
        # Hidden Layer 1: Enhanced feature extraction mapping
        Dense(units=64, activation='relu', input_shape=(num_features,)),
        Dropout(0.2),
        
        # Hidden Layer 2: Abstract representation mapping
        Dense(units=32, activation='relu'),
        Dropout(0.2),
        
        # Hidden Layer 3: Convergence architecture layer
        Dense(units=16, activation='relu'),
        
        # Output Layer: Continuous bounded probability distribution mapping
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Configure criteria to halt gradient descent updates upon loss stabilization
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )

    # Model training interface execution phase (Suppressed console verbosity logs)
    model.fit(
        X_train_scaled, 
        y_train, 
        epochs=50, 
        batch_size=64, 
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0  
    )
    
    return scaler_instance, model

# Execute runtime pipeline initialization on software deployment startup
with st.spinner("Processing structural data pipeline and initializing neural network updates..."):
    scaler, model_tuned = initialize_and_train_pipeline()

# =====================================================================
# 2. APPLICATION USER INTERFACE ARCHITECTURE
# =====================================================================

st.set_page_config(page_title="Bank Churn AI Dashboard", layout="centered")

st.title("Bank Customer Retention Predictive Analytics Dashboard")
st.write("Live risk assessment framework powered by a structurally tuned Artificial Neural Network.")
st.markdown("---")

# Layout segmentation using structural alignment matrices
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=40)
    tenure = st.slider("Tenure Duration (Years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Account Balance Amount ($)", min_value=0.0, value=60000.0)
    num_products = st.selectbox("Number of Enrolled Products", options=[1, 2, 3, 4], index=1)

with col2:
    has_cr_card = st.selectbox("Active Credit Card Holder?", options=["No", "Yes"], index=1)
    is_active = st.selectbox("Active Engagement Profile?", options=["No", "Yes"], index=1)
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, value=80000.0)
    gender = st.selectbox("Customer Demographics: Gender", options=["Female", "Male"], index=1)
    geography = st.selectbox("Customer Demographics: Region", options=["France", "Germany", "Spain"], index=1)

# =====================================================================
# 3. VECTOR ENCODING AND INTERFACE FEATURE ALIGNMENT
# =====================================================================

# Re-map operational parameters to historical tracking system configurations
gender_val = 1 if gender == "Male" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0

# Reproduce the exact alphabetical one-hot encoding array mapping structure
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# Enforce explicit array sequence ordering: CreditScore, Gender, Age, Tenure, 
# Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Germany, Spain
input_vector = np.array([[
    credit_score,
    gender_val,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card_val,
    is_active_val,
    salary,
    geo_germany,
    geo_spain
]], dtype=np.float32)

# =====================================================================
# 4. INFERENCE ENGINE & CLASSIFICATION EVALUATIONS
# =====================================================================

# Business operational constraint classification threshold setting
CLASSIFICATION_THRESHOLD = 0.35

st.markdown("---")
if st.button("Evaluate Customer Churn Risk Index", use_container_width=True):
    # Scale spatial input matrix to match training distribution parameters
    input_scaled = scaler.transform(input_vector)
    
    # Compute operational feedforward prediction tensor transformations
    prediction_prob = float(model_tuned.predict(input_scaled)[0][0])
    
    st.subheader(f"Calculated Churn Risk Factor: {prediction_prob * 100:.2f}%")
    
    # Evaluate risk vectors against the target optimization threshold bounds
    if prediction_prob > CLASSIFICATION_THRESHOLD:
        st.error("CRITICAL ALERT: Customer exhibits extreme attrition risk profiling (High Risk Churn)")
    else:
        st.success("VERDICT STABLE: Customer displays standard retention attributes (Low Risk Safe)")