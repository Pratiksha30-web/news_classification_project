import streamlit as st
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

st.title("PolicyPal Model Deployment")

st.write("This app preprocesses data, trains the model, and evaluates performance.")

if st.button("Run Pipeline"):

    st.write("Step 1: Preprocessing Data...")
    preprocess_data()
    st.success("Data Preprocessing Completed")

    st.write("Step 2: Training Model...")
    X_test, y_test = train_model()
    st.success("Model Training Completed")

    st.write("Step 3: Evaluating Model...")
    results = evaluate_model(X_test, y_test)

    st.success("Model Evaluation Completed")

    st.write("Evaluation Results:")
    st.write(results)