# generate_model_artifacts.py
"""
Script untuk generate model artifacts yang dibutuhkan Streamlit
Jalankan script ini SEBELUM menjalankan Streamlit
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

print("="*60)
print("🔧 GENERATE MODEL ARTIFACTS UNTUK STREAMLIT")
print("="*60)

# ========================================
# KONFIGURASI PATH
# ========================================
DATA_PATH = 'data/dataset_mentah.csv'  # Sesuaikan dengan lokasi file kamu
OUTPUT_DIR = 'model'

# Buat folder model jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================
# STEP 1: LOAD DATA MENTAH
# ========================================
print("\n📂 STEP 1: Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data berhasil dimuat: {len(df)} baris")
except FileNotFoundError:
    print(f"❌ File tidak ditemukan: {DATA_PATH}")
    print("💡 Pastikan path file benar!")
    exit()

# ========================================
# STEP 2: CLEANING DATA
# ========================================
print("\n🧹 STEP 2: Cleaning data...")

# Hapus duplikat
df.drop_duplicates(inplace=True)
print(f"   • Data setelah hapus duplikat: {len(df)} baris")

# Isi missing value dengan string kosong
df.fillna('', inplace=True)

# Normalisasi nama kolom (hapus spasi, ganti dengan underscore)
df.columns = df.columns.str.strip().str.replace(' ', '_')
print(f"   • Kolom: {list(df.columns)}")

# ========================================
# STEP 3: FUNGSI PEMBERSIHAN TEKS
# ========================================
def clean_text(text):
    """Pembersihan teks standar"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ========================================
# STEP 4: BUAT CORPUS
# ========================================
print("\n📝 STEP 3: Membuat corpus...")

text_cols = ['Nama_Menu', 'Kategori', 'Sumber_Karbohidrat',
             'Bahan_Utama_/_Pendamping', 'Deskripsi_Singkat']

# Bersihkan semua kolom teks
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)
    else:
        print(f"⚠️  Kolom '{col}' tidak ditemukan, dilewati")

# Gabungkan semua kolom jadi satu corpus
df['corpus'] = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
print(f"✅ Corpus berhasil dibuat")
print(f"   Contoh corpus:\n   {df['corpus'].iloc[0][:100]}...")

# ========================================
# STEP 5: NORMALISASI KALORI
# ========================================
print("\n🔢 STEP 4: Normalisasi kalori...")

if 'Kalori_(kcal)' in df.columns:
    scaler = MinMaxScaler()
    df['kalori_normalized'] = scaler.fit_transform(df[['Kalori_(kcal)']])
    print(f"✅ Kalori berhasil dinormalisasi (range: {df['kalori_normalized'].min():.3f} - {df['kalori_normalized'].max():.3f})")
else:
    print("❌ Kolom 'Kalori_(kcal)' tidak ditemukan!")
    exit()

# ========================================
# STEP 6: FIT TF-IDF VECTORIZER
# ========================================
print("\n🧮 STEP 5: Fitting TF-IDF Vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=1,  # Minimum document frequency
    max_df=0.95  # Maximum document frequency
)

# ✅ FIT vectorizer dengan corpus
tfidf_matrix = vectorizer.fit_transform(df['corpus'])

print(f"✅ TF-IDF berhasil di-fit")
print(f"   • Shape: {tfidf_matrix.shape}")
print(f"   • Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   • Feature names (5 pertama): {vectorizer.get_feature_names_out()[:5].tolist()}")

# ========================================
# STEP 7: SIMPAN MODEL ARTIFACTS
# ========================================
print(f"\n💾 STEP 6: Menyimpan model artifacts ke folder '{OUTPUT_DIR}'...")

# 1️⃣ Simpan TF-IDF Vectorizer
vectorizer_path = os.path.join(OUTPUT_DIR, 'vectorizer_tfidf.pkl')
joblib.dump(vectorizer, vectorizer_path)
print(f"   ✅ Saved: {vectorizer_path}")

# 2️⃣ Simpan MinMaxScaler
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"   ✅ Saved: {scaler_path}")

# 3️⃣ Simpan Data Train (untuk digunakan di app)
data_train_path = os.path.join(OUTPUT_DIR, 'data_train.csv')
df.to_csv(data_train_path, index=False)
print(f"   ✅ Saved: {data_train_path}")

# ========================================
# STEP 8: VALIDASI MODEL
# ========================================
print("\n🔍 STEP 7: Validasi model yang disimpan...")

# Load ulang untuk testing
vectorizer_loaded = joblib.load(vectorizer_path)
scaler_loaded = joblib.load(scaler_path)
df_loaded = pd.read_csv(data_train_path)

# Test transformasi
test_corpus = "ayam rendah lemak nasi merah"
try:
    test_tfidf = vectorizer_loaded.transform([test_corpus])
    print(f"✅ Vectorizer BERHASIL: dapat transform text baru")
    print(f"   • Test corpus: '{test_corpus}'")
    print(f"   • Output shape: {test_tfidf.shape}")
except Exception as e:
    print(f"❌ Vectorizer GAGAL: {str(e)}")

# Test scaler
test_kalori = [[400]]
try:
    test_norm = scaler_loaded.transform(test_kalori)
    print(f"✅ Scaler BERHASIL: dapat normalize angka baru")
    print(f"   • Input: {test_kalori[0][0]} kcal")
    print(f"   • Output: {test_norm[0][0]:.3f}")
except Exception as e:
    print(f"❌ Scaler GAGAL: {str(e)}")

# ========================================
# SUMMARY
# ========================================
print("\n" + "="*60)
print("✅ MODEL ARTIFACTS BERHASIL DIBUAT!")
print("="*60)
print(f"📂 Lokasi file:")
print(f"   • {vectorizer_path}")
print(f"   • {scaler_path}")
print(f"   • {data_train_path}")
print("\n🚀 Sekarang kamu bisa menjalankan: streamlit run app.py")
print("="*60)