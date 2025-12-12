import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite

st.title("Customer Frequency Prediction App")
st.write("Predict whether a customer is frequent (Frequency > 1) based on Recency, Monetary, and AvgQuantity.")

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

st.subheader("Enter Customer Features")

recency = st.number_input("Recency (days since last purchase)", min_value=0)
monetary = st.number_input("Monetary (total spent)", min_value=0.0)
avg_quantity = st.number_input("Average Quantity per order", min_value=0.0)

if st.button("Predict"):
    input_array = np.array([[recency, monetary, avg_quantity]], dtype=np.float32)

    interpreter.set_tensor(input_index, input_array)
    interpreter.invoke()
    probability = interpreter.get_tensor(output_index)[0][0]

    prediction = 1 if probability >= 0.5 else 0

    st.subheader("Prediction Result")
    st.write("Frequent Customer?", "Yes ✅" if prediction == 1 else "No ❌")
    st.write(f"Probability of being frequent: {probability*100:.2f}%")
