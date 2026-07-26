import streamlit as st
from PIL import Image
import io

from src.inference import Predictor

# Configure the Streamlit page
st.set_page_config(
    page_title="PneumoNet Demo",
    page_icon="🫁",
    layout="wide",
)

# Initialize Predictor lazily
@st.cache_resource
def get_predictor():
    return Predictor()

predictor = get_predictor()

# ─── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🫁 PneumoNet")
st.sidebar.markdown(
    """
    **Pneumonia Detection from Chest X-rays**
    
    This is an interactive demo for the PneumoNet project.
    
    ### Architecture
    - **Model:** Sequential CNN
    - **Input:** 150x150 Grayscale
    - **Blocks:** Conv2D → BatchNorm → MaxPool → Dropout
    - **Output:** Dense(1, Sigmoid)
    
    ### Performance
    - **Train Accuracy:** 98.29%
    - **Test Accuracy:** 95.90%
    """
)
st.sidebar.image("assets/PneumoNet_Architecture_Colored.png", caption="Model Architecture", use_container_width=True)

# ─── Main Content ──────────────────────────────────────────────────────
st.title("Pneumonia Detection AI")
st.write("Upload a chest X-ray image to get a live prediction from the trained CNN model.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        with st.spinner("Analyzing X-ray..."):
            try:
                # Read bytes and predict
                image_bytes = uploaded_file.getvalue()
                result = predictor.predict_from_bytes(image_bytes)
                
                label = result["label"]
                confidence = result["confidence"]
                
                # Display results nicely
                if label == "NORMAL":
                    st.success(f"### Result: **{label}** 🟢")
                else:
                    st.error(f"### Result: **{label}** 🔴")
                    
                st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")
                
                st.progress(confidence)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please upload an image to see the prediction.")
