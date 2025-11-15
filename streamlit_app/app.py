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

# Ambil range kalori dari dataset
if engine:
    kalori_min = int(engine.df_train['Kalori_(kcal)'].min())
    kalori_max = int(engine.df_train['Kalori_(kcal)'].max())
    kalori_mean = int(engine.df_train['Kalori_(kcal)'].mean())
    
    # Tampilkan info range kalori dengan visual
    st.sidebar.caption(f"📊 **Range Kalori Menu:**")
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Min", f"{kalori_min}")
    col2.metric("Avg", f"{kalori_mean}")
    col3.metric("Max", f"{kalori_max}")
else:
    # Fallback jika engine belum load
    kalori_min, kalori_max, kalori_mean = 300, 600, 400

# 1️⃣ Input Kalori (Dynamic Range)
kalori_target = st.sidebar.slider(
    "Target Kalori (kcal)",
    min_value=kalori_min,
    max_value=kalori_max,
    value=kalori_mean,  # Default value = rata-rata kalori di dataset
    step=5,
    help=f"Pilih target kalori (tersedia: {kalori_min}-{kalori_max} kcal)"
)

# 2️⃣ Pilih Kategori Lauk (Dynamic dari Dataset)
if engine:
    kategori_options = sorted(engine.df_train['Kategori'].unique().tolist())
else:
    kategori_options = ["Ayam", "Ikan", "Sapi", "Lainnya"]

kategori_lauk = st.sidebar.selectbox(
    "Pilih Jenis Lauk",
    options=kategori_options,
    help="Pilih satu jenis lauk favorit Anda"
)

# 3️⃣ Pilih Sumber Karbohidrat (Multi-select) - Dynamic dari Dataset
if engine and 'karbo_list' in engine.df_train.columns:
    # Ekstrak semua opsi karbo unik dari dataset
    all_karbo = []
    for karbo_list_str in engine.df_train['karbo_list'].dropna():
        try:
            import ast
            karbo_items = ast.literal_eval(karbo_list_str) if isinstance(karbo_list_str, str) else karbo_list_str
            all_karbo.extend(karbo_items)
        except:
            pass
    karbo_options = sorted(list(set([k.strip() for k in all_karbo if k])))
else:
    # Fallback manual
    karbo_options = ["nasi merah", "nasi putih", "kentang", "ubi", "jagung"]

pilihan_karbo = st.sidebar.multiselect(
    "Pilih Sumber Karbohidrat",
    options=karbo_options,
    default=[karbo_options[0]] if karbo_options else [],
    help="Anda bisa memilih lebih dari satu"
)

# Mode filtering (NEW)
karbo_filter_mode = st.sidebar.radio(
    "Mode Filter Karbohidrat",
    options=["Fleksibel", "Strict"],
    help="""
    • Fleksibel: Menu dengan minimal 1 pilihan karbo yang cocok akan muncul
    • Strict: Hanya menu yang punya SEMUA pilihan karbo user
    """,
    horizontal=True
)

st.sidebar.caption(f"Mode: {'🟢 Fleksibel' if karbo_filter_mode == 'Fleksibel' else '🔴 Strict'}")

# 4️⃣ Deskripsi Preferensi (Text Input)

# Quick select tags (suggested preferences)
st.sidebar.markdown("**🏷️ Tag Populer (klik untuk tambahkan):**")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.caption("✅ **Yang Diinginkan:**")
    if st.button("🟢 Rendah Lemak", key="tag1"):
        deskripsi_pref = "rendah lemak"
    if st.button("🟢 Tinggi Protein", key="tag2"):
        deskripsi_pref = "tinggi protein"
    if st.button("🟢 Kukus", key="tag3"):
        deskripsi_pref = "kukus"

with col2:
    st.caption("❌ **Yang Dihindari:**")
    if st.button("🔴 Tanpa Santan", key="tag4"):
        deskripsi_pref = "tanpa santan"
    if st.button("🔴 Tanpa Tempe", key="tag5"):
        deskripsi_pref = "tanpa tempe"
    if st.button("🔴 Tidak Pedas", key="tag6"):
        deskripsi_pref = "tidak pedas"

deskripsi_pref = st.sidebar.text_area(
    "Atau Ketik Manual:",
    value="rendah lemak tanpa santan",
    height=80,
    help="""
    💡 **Tips Pencarian:**
    
    ✅ **Yang Diinginkan (Positif):**
    - "rendah lemak"
    - "tinggi protein"
    - "kukus"
    - "pedas"
    
    ✅ **Yang Dihindari (Negatif):**
    - "tanpa santan"
    - "tanpa tempe"
    - "tidak pedas"
    - "bebas MSG"
    
    ⚠️ **Hindari:**
    - Typo (misal: "renda" ❌, gunakan "rendah" ✅)
    - Bahasa campuran (gunakan 1 bahasa saja)
    """
)

# Tampilkan preview parsing (opsional, untuk debug)
if st.sidebar.checkbox("🔍 Preview Parsing Input", value=False):
    if engine and deskripsi_pref.strip():
        parsed = engine.preprocess_user_input(deskripsi_pref)
        st.sidebar.json({
            "Kata Positif": parsed['positive'],
            "Kata Negatif (Dihindari)": parsed['negative']
        })

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
    st.sidebar.caption("📊 Semua kriteria diperhitungkan secara proporsional")
else:  # Fokus Kalori
    weights = {
        'deskripsi': 0.25,
        'kategori': 0.15,
        'karbohidrat': 0.10,
        'kalori': 0.50
    }
    st.sidebar.caption("📊 Prioritas utama pada kesesuaian kalori (50%)")

# Tampilkan bobot yang digunakan dengan visual yang lebih baik
with st.sidebar.expander("📋 Lihat Detail Bobot Kriteria"):
    st.markdown("**Bobot yang Digunakan:**")
    
    # Tampilkan sebagai progress bar untuk visualisasi
    for kriteria, bobot in weights.items():
        st.write(f"**{kriteria.title()}**: {bobot:.0%}")
        st.progress(bobot)
    
    st.caption("💡 Total bobot = 100%")

# Jumlah rekomendasi
top_n = st.sidebar.slider(
    "Jumlah Rekomendasi",
    min_value=3,
    max_value=10,
    value=5,
    help="Berapa menu yang ingin ditampilkan?"
)

# Debug mode
show_debug = st.sidebar.checkbox(
    "🔍 Tampilkan Detail Skor",
    help="Tampilkan breakdown skor similarity per kriteria"
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
                    top_n=top_n,
                    karbo_strict_mode=(karbo_filter_mode == "Strict")  # NEW parameter
                )
                
                # ✨ VALIDASI: Cek apakah input deskripsi valid
                if deskripsi_pref.strip():
                    parsed = engine.preprocess_user_input(deskripsi_pref)
                    
                    # Warning jika ada kata negatif terdeteksi
                    if parsed['negative']:
                        st.info(f"ℹ️ **Kata yang dihindari terdeteksi:** {', '.join(parsed['negative'])}")
                    
                    # Warning jika input terlalu pendek
                    if len(parsed['positive'].split()) == 0:
                        st.warning("⚠️ Input deskripsi hanya berisi kata negatif. Rekomendasi berdasarkan kriteria lain.")
                
                # Tampilkan ringkasan input user
                st.success("✅ Rekomendasi berhasil dibuat!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Target Kalori", f"{kalori_target} kcal")
                with col2:
                    st.metric("Jenis Lauk", kategori_lauk)
                with col3:
                    # Tampilkan jumlah pilihan karbo yang dipilih user
                    jumlah_karbo = len(pilihan_karbo) if pilihan_karbo else 0
                    st.metric("Pilihan Karbo", f"{jumlah_karbo} jenis")
                with col4:
                    st.metric("Skenario", skenario_bobot.split()[0])  # "Bobot" atau "Fokus"
                
                # ✨ Tampilkan detail pilihan karbohidrat user
                if pilihan_karbo:
                    st.caption(f"🍚 **Karbohidrat yang dipilih:** {', '.join([k.title() for k in pilihan_karbo])}")
                else:
                    st.caption("🍚 **Karbohidrat:** Semua (tidak ada filter)")
                
                st.markdown("---")
                st.subheader(f"🏆 Top-{top_n} Menu yang Direkomendasikan")
                
                # Cek apakah ada skor negatif (indikasi bug)
                if (recommendations['Skor_Similarity'] < 0).any():
                    st.error("⚠️ PERINGATAN: Terdeteksi skor negatif! Mungkin ada bug di perhitungan.")
                
                # Tampilkan hasil dalam format card
                for idx, row in recommendations.iterrows():
                    rank = row['Rank']
                    skor = row['Skor_Similarity']
                    nama_menu = row.get('Nama_Menu', 'N/A')
                    kategori = row.get('Kategori', 'N/A')
                    kalori = row.get('Kalori_(kcal)', 'N/A')
                    
                    # ✨ Format karbohidrat dengan koma
                    karbo_raw = row.get('Sumber_Karbohidrat', 'N/A')
                    if karbo_raw and karbo_raw != 'N/A':
                        # Split by space, capitalize, join with comma
                        karbo_list = []
                        karbo_str = str(karbo_raw).lower()
                        
                        # Handle multi-word items like "nasi merah"
                        multi_word = ['nasi merah', 'nasi putih', 'nasi coklat', 'roti gandum']
                        for item in multi_word:
                            if item in karbo_str:
                                karbo_list.append(item.title())
                                karbo_str = karbo_str.replace(item, '')
                        
                        # Handle single words
                        single_words = [w.strip().title() for w in karbo_str.split() if len(w.strip()) > 2]
                        karbo_list.extend(single_words)
                        
                        # Remove duplicates
                        karbo_list = list(dict.fromkeys(karbo_list))
                        karbo = ', '.join(karbo_list) if karbo_list else 'N/A'
                    else:
                        karbo = 'N/A'
                    
                    deskripsi = row.get('Deskripsi_Singkat', 'N/A')
                    
                    # Card dengan styling
                    with st.container():
                        col_rank, col_info = st.columns([1, 9])
                        
                        with col_rank:
                            st.markdown(f"### #{rank}")
                        
                        with col_info:
                            st.markdown(f"**{nama_menu}**")
                            st.caption(f"Skor Kemiripan: {skor:.3f} | Kategori: {kategori.title()} | Kalori: {kalori} kcal")
                            
                            # Tampilkan karbohidrat sebagai pills/badges
                            if karbo != 'N/A':
                                karbo_items = [k.strip() for k in karbo.split(',')]
                                
                                # Highlight karbo yang sesuai pilihan user
                                karbo_display = []
                                user_karbo_lower = [k.lower() for k in pilihan_karbo]
                                
                                for item in karbo_items:
                                    if item.lower() in user_karbo_lower:
                                        # Tambahkan checkmark untuk yang cocok
                                        karbo_display.append(f"`{item}` ✅")
                                    else:
                                        karbo_display.append(f"`{item}`")
                                
                                st.markdown(f"🍚 **Pilihan Karbohidrat:** {' '.join(karbo_display)}")
                                
                                # Info jumlah opsi
                                st.caption(f"   └─ {len(karbo_items)} pilihan tersedia")
                            else:
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