import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diamonds.csv")  
    return df

df = load_data()

# Define features & target variable
X = df.drop(columns=['price'])  
y = df['price']  

# Identify categorical & numerical columns
categorical_features = ['cut', 'color', 'clarity']
numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical_features),  
    ("num", "passthrough", numerical_features)  
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, min_samples_split=5)
}

# Streamlit UI
st.title("💎 Diamond Price Prediction App")
st.write("Select a model and enter diamond features to predict the price.")

# Model selection
selected_model = st.selectbox("Select Model", list(models.keys()))

# Input fields for user to enter values
carat = st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01)
cut = st.selectbox("Cut", df['cut'].unique())
color = st.selectbox("Color", df['color'].unique())
clarity = st.selectbox("Clarity", df['clarity'].unique())
depth = st.number_input("Depth", min_value=50.0, max_value=80.0, step=0.1)
table = st.number_input("Table", min_value=50.0, max_value=80.0, step=0.1)
x = st.number_input("X (Length)", min_value=0.0, max_value=10.0, step=0.01)
y = st.number_input("Y (Width)", min_value=0.0, max_value=10.0, step=0.01)
z = st.number_input("Z (Depth)", min_value=0.0, max_value=10.0, step=0.01)

# Convert input into DataFrame
input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]], 
                          columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])

# Train selected model
pipeline = make_pipeline(preprocessor, models[selected_model])
pipeline.fit(X_train, y_train)

# Predict price
if st.button("Predict Price"):
    predicted_price = pipeline.predict(input_data)[0]
    st.success(f"💰 Predicted Diamond Price: ${predicted_price:.2f}")
