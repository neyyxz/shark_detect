import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "shark_model.tflite"
LABEL_PATH = "labels.txt"
BACKGROUND_IMAGE = "1080.png"
IMG_SIZE = 224

st.set_page_config(
    page_title="Shark Species Identifier",
    layout="centered"
)

# =========================
# LOAD LABELS
# =========================
with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

# =========================
# BACKGROUND + CENTER FIX
# =========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        html, body {{
            height: 100%;
        }}

        .stApp {{
            background:
                linear-gradient(
                    rgba(0,0,0,0.65),
                    rgba(0,0,0,0.65)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .block-container {{
            background-color: rgba(0,0,0,0.55);
            padding: 2.5rem;
            border-radius: 18px;
            max-width: 680px;
            width: 100%;
            text-align: center;
        }}

        div[data-testid="stFileUploader"] {{
            background-color: rgba(0,0,0,0.55);
            padding: 16px;
            border-radius: 14px;
        }}

        h1, h2, h3, p, span, label {{
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(BACKGROUND_IMAGE)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# UI (NO SCROLL)
# =========================
st.title("🦈 Shark Species Identifier")
st.write("Upload foto hiu untuk mengetahui jenisnya")
st.caption(
    "Sistem deteksi ikan hiu berbasis Machine Learning menggunakan metode Convolutional Neural Network (CNN) yang telah dilatih dan memiliki akurasi tinggi"
)

uploaded_file = st.file_uploader(
    "📤 Upload gambar hiu (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=420)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.subheader(f"Prediction: **{CLASS_NAMES[class_idx]}**")
    st.progress(confidence)
    st.write(f"Confidence: **{confidence:.2%}**")
