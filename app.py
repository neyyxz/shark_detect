import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# =========================
# CONFIG
# =========================
MODEL_PATH = "shark_model.tflite"
BACKGROUND_IMAGE = "background.png"
IMG_SIZE = 224

CLASS_NAMES = [
    "not_shark",
    "tiger_shark",
    "whale_shark",
    "white_shark",
    "whitetip_shark",
]

st.set_page_config(
    page_title="Shark Species Identifier",
    layout="centered"
)

# =========================
# BACKGROUND + DARK MODE
# =========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Paksa dark mode */
        :root {{
            color-scheme: dark;
        }}

        /* Background */
        .stApp {{
            background:
                linear-gradient(
                    rgba(0,0,0,0.55),
                    rgba(0,0,0,0.55)
                ),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
        }}

        /* Teks global */
        html, body, [class*="css"] {{
            color: #ffffff !important;
        }}

        h1, h2, h3 {{
            color: #ffffff !important;
        }}

        p, span, label {{
            color: #e5e7eb !important;
        }}

        /* Card effect */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.45);
            padding: 2rem;
            border-radius: 16px;
        }}

        /* File uploader */
        div[data-testid="stFileUploader"] {{
            background-color: rgba(0,0,0,0.5);
            padding: 12px;
            border-radius: 12px;
        }}

        /* Mobile fix */
        @media (max-width: 768px) {{
            .stApp {{
                background-position: center top;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(BACKGROUND_IMAGE)

# =========================
# LOAD TFLITE MODEL
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# UI
# =========================
st.title("🦈 Shark Species Identifier")
st.write("Upload foto hiu, sistem akan memprediksi spesiesnya.")
st.caption("Sistem identifikasi Shark Species Identifies menggunakan Model berbasis CNN (Convolutional Neural Network) yang sudah dilatih dan memiliki akurasi tinggi")

uploaded_file = st.file_uploader(
    "Upload gambar hiu (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.subheader(f"Prediction: **{CLASS_NAMES[class_idx]}**")
    st.progress(confidence)
    st.write(f"Confidence: **{confidence:.2%}**")
