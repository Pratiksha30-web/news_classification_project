import streamlit as st
import joblib
import os
import pandas as pd
from src.data_preprocessing import preprocess_data, clean_text
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import MODEL_PATH, METRICS_PATH

st.set_page_config(
    page_title="News Classification Hub",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📰 News Classification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Seamlessly train a machine learning pipeline and classify news articles instantly.</p>", unsafe_allow_html=True)
st.divider()

# Create Tabs
tab_predict, tab_pipeline, tab_metrics = st.tabs(["🔮 Classify News", "⚙️ Pipeline Management", "📊 Model Metrics"])

with tab_predict:
    st.header("Predict News Category")
    st.markdown("Enter a news snippet below, and our trained model will classify its category instantly.")
    
    user_input = st.text_area("News Content", height=200, placeholder="Paste the news article text here...")
    
    if st.button("Classify Text", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to classify.")
        else:
            if not os.path.exists(MODEL_PATH):
                st.error("🚨 Model not found! Please go to the **Pipeline Management** tab and train the model first.")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        # Load model & vectorizer
                        model, vectorizer = joblib.load(MODEL_PATH)
                        
                        # Preprocess text (using the imported clean_text function if it's available)
                        cleaned_text = clean_text(user_input)
                        vec_text = vectorizer.transform([cleaned_text])
                        
                        prediction = model.predict(vec_text)[0]
                        
                        st.success("✅ Classification Successful!")
                        st.metric(label="Predicted Category", value=str(prediction).upper())
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

with tab_pipeline:
    st.header("Pipeline Controls")
    st.markdown("Run individual steps of the machine learning pipeline.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**1. Data Preprocessing**\nClean text, remove stopwords, and prepare dataset.")
        if st.button("Run Preprocessing"):
            with st.spinner("Processing dataset..."):
                try:
                    preprocess_data()
                    st.success("✨ Data preprocessing completed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.warning("**2. Model Training**\nTrain a LinearSVC model on the processed data.")
        if st.button("Train Model"):
            with st.spinner("Training model... this might take a minute."):
                try:
                    X_test, y_test = train_model()
                    st.session_state["X_test"] = X_test
                    st.session_state["y_test"] = y_test
                    st.success("🎯 Model training completed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col3:
        st.success("**3. Model Evaluation**\nEvaluate the model and generate metrics.")
        if st.button("Evaluate Model"):
            if "X_test" in st.session_state and "y_test" in st.session_state:
                with st.spinner("Evaluating model..."):
                    try:
                        evaluate_model(st.session_state["X_test"], st.session_state["y_test"])
                        st.success("📈 Evaluation completed! Check the **Model Metrics** tab.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("⚠️ Please train the model first to generate test data.")

with tab_metrics:
    st.header("Model Performance")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_content = f.read()
            st.code(metrics_content, language="text")
    else:
        st.info("No metrics found. Please train and evaluate the model to generate metrics.")

# Sidebar info
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2965/2965306.png", width=100)
st.sidebar.title("About")
st.sidebar.info(
    "This application classifies news articles into distinct categories using Natural Language Processing (NLP) "
    "and a Support Vector Machine (LinearSVC) model."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io/)")