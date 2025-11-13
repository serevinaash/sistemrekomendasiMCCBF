import streamlit as st
import pandas as pd
import joblib
from utils.recommendation import get_recommendation_for_user  # Fungsi dari notebookmu

# ======================
# Load model & data
# ======================
df_train = pd.read_csv('../data/data_train.csv')
vectorizer = joblib.load('../model/vectorizer.joblib')
tfidf_train_matrix = joblib.load('../model/tfidf_matrix.joblib')

# ======================
# UI Header
# ======================
st.title("Sistem Rekomendasi Menu Diet UMKM")

# ======================
# Input User
# ======================
user_kalori = st.slider("Kalori (kcal)", 200, 800, 400)
user_kategori = st.selectbox("Kategori", df_train['Kategori'].unique())
user_karbo = st.selectbox("Sumber Karbohidrat", df_train['Sumber_Karbohidrat'].unique())
user_teks = st.text_input("Deskripsi Menu (misal 'pedas tanpa santan')")

st.subheader("Bobot Kriteria")
bobot_deskripsi = st.slider("Deskripsi", 0.0, 1.0, 0.25)
bobot_kategori = st.slider("Kategori", 0.0, 1.0, 0.25)
bobot_karbo = st.slider("Karbohidrat", 0.0, 1.0, 0.25)
bobot_kalori = st.slider("Kalori", 0.0, 1.0, 0.25)

weights = {
    'deskripsi': bobot_deskripsi,
    'kategori': bobot_kategori,
    'karbohidrat': bobot_karbo,
    'kalori': bobot_kalori
}

# ======================
# Tampilkan Rekomendasi
# ======================
if st.button("Rekomendasikan"):
    hasil = get_recommendation_for_user(
        pref_kalori=user_kalori,
        pref_kategori=user_kategori,
        pref_karbo=user_karbo,
        pref_teks=user_teks,
        weights=weights,
        df_train=df_train,
        vectorizer=vectorizer,
        tfidf_train_matrix=tfidf_train_matrix,
        top_n=5
    )
    st.dataframe(hasil)
