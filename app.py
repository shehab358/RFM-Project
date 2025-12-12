import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model.h5")  

st.title("Customer Frequency Prediction App")
st.write("Predict whether a customer is frequent (Frequency > 1) based on Recency, Monetary, and AvgQuantity.")

st.subheader("Enter Customer Features")

recency = st.number_input("Recency (days since last purchase)", min_value=0)
monetary = st.number_input("Monetary (total spent)", min_value=0.0)
avg_quantity = st.number_input("Average Quantity per order", min_value=0.0)

if st.button("Predict"):
    input_array = np.array([[recency, monetary, avg_quantity]], dtype=np.float32)
    
    probability = model.predict(input_array)[0][0]  
    
    prediction = 1 if probability >= 0.5 else 0
    
    st.subheader("Prediction Result")
    st.write("Frequent Customer?", "Yes ✅" if prediction == 1 else "No ❌")
    st.write(f"Probability of being frequent: {probability*100:.2f}%")
