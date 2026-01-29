import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

MODEL_PATH = "shark_model.tflite"

# =========================
# BACKGROUND IMAGE
# =========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# panggil background
set_background("background.png")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "tiger_shark",
    "whale_shark",
    "white_shark",
    "whitetip_shark",
]

# =========================
# UI
# =========================
st.markdown(
    """
    <div style="background-color: rgba(0,0,0,0.6);
                padding: 20px;
                border-radius: 15px;">
    """,
    unsafe_allow_html=True
)

st.title("🦈 Shark Species Identifier")
st.write("Upload foto hiu, sistem akan memprediksi spesiesnya.")
st.write(
    "Sistem ini menggunakan model berbasis CNN "
    "(Convolutional Neural Network) yang telah dilatih."
)

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])

    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {CLASS_NAMES[class_idx]}")
    st.write(f"Confidence: {confidence:.2%}")

st.markdown("</div>", unsafe_allow_html=True)
