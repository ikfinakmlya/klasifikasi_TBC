import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model (.keras)
@st.cache_resource  # biar model cuma diload sekali
def load_model():
    return tf.keras.models.load_model("cnn_tbc_model.keras")

model = load_model()

# Judul Aplikasi
st.title("Klasifikasi Citra TBC vs Normal")
st.write("Upload gambar X-ray untuk deteksi **TBC** menggunakan model CNN.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Preprocessing gambar
    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    prob = float(prediction[0][0])  # prob utk kelas index=1 (tuber)

    confidence = max(prob, 1-prob)  # ambil confidence
    if prob > 0.5:
        label = "TBC"
    else:
        label = "Normal"

    st.image(img, caption=f"Hasil Prediksi: {label}", use_container_width=True)
    st.write(f"Probabilitas TBC: {prob:.3f}")
    st.write(f"Confidence: {confidence:.3f}")

    if confidence < 0.75:
        st.warning("⚠️ Model tidak yakin. Mungkin gambar bukan chest X-ray atau kualitas data rendah.")
    else:
        if label == "TBC":
            st.success("✅ Prediksi Model: **TBC**")
        else:
            st.success("✅ Prediksi Model: **Normal (non-TBC)**")
