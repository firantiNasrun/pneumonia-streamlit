import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('model_pneumonia.h5', compile=False)
class_names = ['NORMAL', 'PNEUMONIA']

# Judul aplikasi
st.title("Klasifikasi Pneumonia dari X-Ray Dada")

st.write("Upload gambar X-Ray, lalu klik prediksi untuk melihat hasil.")

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Gambar yang diupload', use_column_width=True)

    if st.button('Prediksi'):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        pred_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Hasil Prediksi: {class_names[pred_index]} ({confidence:.2f}%)")
