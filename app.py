import streamlit as st
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("Knn_model.pkl")

model = load_model()
n_features = model.n_features_in_
class_labels = model.classes_

# Page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered",
    page_icon="🩺"
)

# App title
st.title("🩺 Diabetes Prediction App")
st.caption("Predict the likelihood of diabetes using a trained KNN model.")

st.divider()

# Sidebar: model details
with st.sidebar:
    st.header("📘 Model Info")
    st.write(f"**Model Type:** {model.__class__.__name__}")
    st.write(f"**Features Expected:** {n_features}")
    st.write("**Prediction Labels:**")
    for cls in class_labels:
        st.write(f"- {cls}")
    st.markdown("---")
    st.markdown("🔬 *Built with scikit-learn, joblib & Streamlit*")

# Input and output sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔢 Input Patient Data")
    user_input = []
    for i in range(n_features):
        val = st.number_input(f"Feature {i+1}", step=0.01, key=f"f{i}")
        user_input.append(val)

    predict_btn = st.button("🔮 Predict Diabetes")

with col2:
    st.subheader("📊 Prediction Result")

    if predict_btn:
        input_array = np.array([user_input])
        prediction = model.predict(input_array)[0]
        probs = model.predict_proba(input_array)[0]

        st.success(f"🎯 Prediction: `{prediction}`")

        st.markdown("### Confidence Scores:")
        for cls, prob in zip(class_labels, probs):
            st.progress(prob)
            st.markdown(f"**{cls}** — `{prob:.2%}`")

        st.balloons()
    else:
        st.info("👈 Enter values and click Predict.")

