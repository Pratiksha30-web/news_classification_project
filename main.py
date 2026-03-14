import streamlit as st
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model

st.set_page_config(page_title="News Classification System", layout="wide")

st.title("📰 News Classification Project")
st.write("Machine Learning pipeline for News Category Classification")

# Sidebar controls
st.sidebar.header("Pipeline Controls")

run_preprocess = st.sidebar.button("Run Data Preprocessing")
run_train = st.sidebar.button("Train Model")
run_evaluate = st.sidebar.button("Evaluate Model")

# Preprocessing
if run_preprocess:
    st.subheader("Data Preprocessing")
    with st.spinner("Processing dataset..."):
        preprocess_data()
    st.success("Data preprocessing completed")

# Training
if run_train:
    st.subheader("Model Training")
    with st.spinner("Training model..."):
        X_test, y_test = train_model()
    st.success("Model training completed")
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

# Evaluation
if run_evaluate:
    st.subheader("Model Evaluation")

    if "X_test" in st.session_state:
        evaluate_model(
            st.session_state["X_test"],
            st.session_state["y_test"]
        )
        st.success("Evaluation completed")
    else:
        st.warning("Please train the model first.")