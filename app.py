import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# === Load TFLite model ===
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model_pneumonia.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Preprocess image ===
def preprocess_image(img: Image.Image):
    # Sesuaikan ukuran dengan input model
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# === Predict ===
def predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# === Streamlit UI ===
st.title("Pneumonia Detection with TFLite")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = predict(img_array)

    # Jika output 1 neuron (probability)
    prob = prediction[0][0]
    if prob > 0.5:
        st.error(f"⚠️ Pneumonia Detected (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Normal (Confidence: {(1-prob)*100:.2f}%)")
