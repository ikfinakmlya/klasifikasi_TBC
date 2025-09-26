import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_tbc_model.keras")

model = load_model()

st.title("🩻 Klasifikasi Citra X-Ray TBC vs Normal")
st.write("Upload gambar **X-ray dada** untuk deteksi TBC menggunakan model CNN. \
Jika gambar bukan X-ray dada, model akan memberi peringatan.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Preprocessing gambar
    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prob = float(model.predict(img_array)[0][0])

    # Threshold confidence
    if prob > 0.7:
        label = "TBC"
        st.success(f"✅ Prediksi Model: **{label}** (Probabilitas {prob:.3f})")
    elif prob < 0.3:
        label = "Normal"
        st.success(f"✅ Prediksi Model: **{label}** (Probabilitas {prob:.3f})")
    else:
        label = "Unknown"
        st.warning(f"⚠️ Model tidak yakin (Probabilitas {prob:.3f}). \
Mungkin gambar bukan X-ray dada atau kualitas gambar buruk.")

    # Tampilkan gambar
    st.image(img, caption=f"Hasil Prediksi: {label}", use_container_width=True)
