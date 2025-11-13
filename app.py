# app.py
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Import MCCBF Engine
sys.path.append(str(Path(__file__).parent))
from utils.mccbf_engine import load_mccbf_engine

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Sistem Rekomendasi Menu Diet Sehat",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# LOAD MODEL (dengan caching agar tidak reload terus)
# ========================================
@st.cache_resource
def load_engine():
    """Load MCCBF engine sekali saja (di-cache oleh Streamlit)"""
    try:
        engine = load_mccbf_engine(model_dir='model')
        return engine
    except Exception as e:
        st.error(f"❌ Gagal load model: {str(e)}")
        return None

engine = load_engine()

# ========================================
# HEADER APLIKASI
# ========================================
st.title("🥗 Sistem Rekomendasi Menu Diet Sehat")
st.markdown("### Icel's Room Kitchen - Katering Diet Personal")
st.markdown("---")

# Info singkat
with st.expander("ℹ️ Tentang Sistem Ini"):
    st.write("""
    Sistem ini menggunakan metode **Multi-Criteria Content-Based Filtering (MCCBF)** 
    untuk merekomendasikan menu diet sehat berdasarkan:
    
    - 🔢 **Kebutuhan Kalori** (target harian Anda)
    - 🍗 **Jenis Lauk** (Ayam, Ikan, Daging)
    - 🍚 **Sumber Karbohidrat** (Nasi Merah, Kentang, dll)
    - 📝 **Preferensi Deskripsi** (rendah lemak, tanpa santan, dll)
    
    Sistem akan memberikan **Top-5 menu** yang paling sesuai dengan profil Anda.
    """)

# ========================================
# SIDEBAR - INPUT PREFERENSI USER
# ========================================
st.sidebar.header("🎯 Masukkan Preferensi Anda")

# 1️⃣ Input Kalori
kalori_target = st.sidebar.slider(
    "Target Kalori (kcal)",
    min_value=300,
    max_value=600,
    value=400,
    step=10,
    help="Pilih target kalori harian yang Anda inginkan"
)

# 2️⃣ Pilih Kategori Lauk
kategori_lauk = st.sidebar.selectbox(
    "Pilih Jenis Lauk",
    options=["Ayam", "Ikan", "Daging"],
    help="Pilih satu jenis lauk favorit Anda"
)

# 3️⃣ Pilih Sumber Karbohidrat (Multi-select)
pilihan_karbo = st.sidebar.multiselect(
    "Pilih Sumber Karbohidrat",
    options=["nasi merah", "nasi putih", "kentang", "ubi", "jagung"],
    default=["nasi merah"],
    help="Anda bisa memilih lebih dari satu"
)

# 4️⃣ Deskripsi Preferensi (Text Input)
deskripsi_pref = st.sidebar.text_area(
    "Deskripsi Preferensi Lainnya",
    value="rendah lemak tanpa santan",
    help="Contoh: 'rendah lemak', 'tanpa santan', 'tinggi protein', dll"
)

# 5️⃣ Pilih Skenario Bobot (Advanced)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Lanjutan")

skenario_bobot = st.sidebar.radio(
    "Pilih Skenario Pembobotan",
    options=["Bobot Seimbang", "Fokus Kalori"],
    help="""
    - Bobot Seimbang: Semua kriteria diperhitungkan secara merata
    - Fokus Kalori: Prioritas utama pada kesesuaian kalori
    """
)

# Mapping skenario ke bobot
if skenario_bobot == "Bobot Seimbang":
    weights = {
        'deskripsi': 0.25,
        'kategori': 0.25,
        'karbohidrat': 0.20,
        'kalori': 0.30
    }
else:  # Fokus Kalori
    weights = {
        'deskripsi': 0.25,
        'kategori': 0.15,
        'karbohidrat': 0.10,
        'kalori': 0.50
    }

# Tampilkan bobot yang digunakan
with st.sidebar.expander("Lihat Bobot yang Digunakan"):
    st.json(weights)

# Jumlah rekomendasi
top_n = st.sidebar.slider(
    "Jumlah Rekomendasi",
    min_value=3,
    max_value=10,
    value=5,
    help="Berapa menu yang ingin ditampilkan?"
)

# ========================================
# TOMBOL UNTUK GENERATE REKOMENDASI
# ========================================
st.sidebar.markdown("---")
generate_btn = st.sidebar.button("🚀 Dapatkan Rekomendasi", type="primary")

# ========================================
# MAIN CONTENT - HASIL REKOMENDASI
# ========================================

if generate_btn:
    if engine is None:
        st.error("❌ Model belum ter-load. Pastikan file model ada di folder `model/`")
    else:
        with st.spinner("🔍 Mencari menu terbaik untuk Anda..."):
            try:
                # Panggil fungsi rekomendasi
                recommendations = engine.get_recommendations(
                    kalori_target=kalori_target,
                    kategori_lauk=kategori_lauk,
                    sumber_karbo_list=pilihan_karbo,
                    deskripsi_preferensi=deskripsi_pref,
                    weights=weights,
                    top_n=top_n
                )
                
                # Tampilkan ringkasan input user
                st.success("✅ Rekomendasi berhasil dibuat!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Target Kalori", f"{kalori_target} kcal")
                with col2:
                    st.metric("Jenis Lauk", kategori_lauk)
                with col3:
                    st.metric("Sumber Karbo", len(pilihan_karbo))
                with col4:
                    st.metric("Skenario", skenario_bobot)
                
                st.markdown("---")
                st.subheader(f"🏆 Top-{top_n} Menu yang Direkomendasikan")
                
                # Tampilkan hasil dalam format card
                for idx, row in recommendations.iterrows():
                    rank = row['Rank']
                    skor = row['Skor_Similarity']
                    nama_menu = row.get('Nama_Menu', 'N/A')
                    kategori = row.get('Kategori', 'N/A')
                    kalori = row.get('Kalori_(kcal)', 'N/A')
                    karbo = row.get('Sumber_Karbohidrat', 'N/A')
                    deskripsi = row.get('Deskripsi_Singkat', 'N/A')
                    
                    # Card dengan styling
                    with st.container():
                        col_rank, col_info = st.columns([1, 9])
                        
                        with col_rank:
                            st.markdown(f"### #{rank}")
                        
                        with col_info:
                            st.markdown(f"**{nama_menu}**")
                            st.caption(f"Skor Kemiripan: {skor:.3f} | Kategori: {kategori} | Kalori: {kalori} kcal")
                            st.write(f"🍚 Karbohidrat: {karbo}")
                            st.write(f"📝 {deskripsi}")
                        
                        st.markdown("---")
                
                # Tombol download hasil
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Rekomendasi (CSV)",
                    data=csv,
                    file_name='rekomendasi_menu_diet.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.exception(e)

else:
    # Tampilan default sebelum tombol diklik
    st.info("👈 Masukkan preferensi Anda di sidebar, lalu klik **Dapatkan Rekomendasi**")
    
    # Tampilkan preview dataset
    st.subheader("📊 Preview Dataset Menu")
    if engine:
        st.dataframe(
            engine.df_train[['Nama_Menu', 'Kategori', 'Kalori_(kcal)', 'Sumber_Karbohidrat']].head(10),
            use_container_width=True
        )

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("🔬 Sistem Rekomendasi MCCBF - Icel's Room Kitchen | UIN Sunan Gunung Djati Bandung")