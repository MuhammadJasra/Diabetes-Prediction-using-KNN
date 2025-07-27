import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("your_model.pkl")

model = load_model()
n_features = model.n_features_in_
class_labels = model.classes_

# Page settings
st.set_page_config(
    page_title="KNN Classifier App",
    layout="wide",
    page_icon="🔍"
)

# Header
st.title("🔍 KNN Classifier App")
st.caption("Built with Streamlit — Predict outcomes using a trained K-Nearest Neighbors model.")

st.divider()

# Sidebar
with st.sidebar:
    st.header("📘 Model Info")
    st.write(f"**Model Type:** {model.__class__.__name__}")
    st.write(f"**Features Expected:** {n_features}")
    st.write("**Class Labels:**")
    for cls in class_labels:
        st.write(f"- {cls}")
    st.markdown("---")
    st.write("🧠 Powered by `scikit-learn`, `Streamlit`, and `joblib`")

# Input & Output layout
col1, col2 = st.columns([1.5, 2])

# Input section
with col1:
    st.subheader("🔢 Input Features")
    user_input = []
    for i in range(n_features):
        val = st.number_input(f"Feature {i + 1}", step=0.01, key=f"f_{i}")
        user_input.append(val)

    predict_btn = st.button("🔮 Predict")

# Output section
with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        input_array = np.array([user_input])
        prediction = model.predict(input_array)[0]
        probs = model.predict_proba(input_array)[0]

        st.success(f"🎯 Predicted Class: `{prediction}`")

        st.markdown("### Confidence Scores:")
        for cls, prob in zip(class_labels, probs):
            st.progress(prob)
            st.markdown(f"**{cls}** — `{prob:.2%}`")

        st.balloons()
    else:
        st.info("👈 Enter input values on the left and click Predict.")

