import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Import MCCBF Engine
sys.path.append(str(Path(__file__).parent))
from utils.mccbf_engine import MCCBFEngine

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
        # ✅ FIX: Gunakan MCCBFEngine langsung
        engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
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
    
    - 🔢 **Kebutuhan Kalori** (target harian Anda) - dengan Gaussian Scoring
    - 🍗 **Jenis Lauk** (Ayam, Ikan, Daging) - dengan Fuzzy Matching
    - 🍚 **Sumber Karbohidrat** (Nasi Merah, Kentang, dll)
    - 📝 **Preferensi Deskripsi** (rendah lemak, tanpa santan, dll) - dengan TF-IDF Bigram
    
    Sistem akan memberikan **Top-N menu** yang paling sesuai dengan profil Anda.
    
    **Mode yang tersedia:**
    - 🟢 Mode Seimbang (F1: **80.01%** - Optimal!) ✅
    - 🟠 Mode Fokus Deskripsi (F1: **80.01%** - Optimal!) ✅
    - 🔵 Mode Fokus Lauk (F1: **77.41%**)
    
    **Fitur Optimisasi:**
    - ✨ Gaussian Calorie Scoring (sigma=30) untuk matching presisi
    - ✨ Fuzzy Matching untuk toleransi variasi nama lauk
    - ✨ TF-IDF Bigram + Keyword Boost untuk analisis deskripsi
    - ✨ Multi-criteria balanced weighting
    """)

# ========================================
# SIDEBAR - INPUT PREFERENSI USER
# ========================================
st.sidebar.header("🎯 Masukkan Preferensi Anda")

# Ambil range kalori dari dataset
if engine and not engine.df.empty:
    # ✅ FIX: Gunakan engine.df (bukan df_train) dan kolom 'Kalori' (bukan 'Kalori_(kcal)')
    kalori_min = int(engine.df['Kalori'].min())
    kalori_max = int(engine.df['Kalori'].max())
    kalori_mean = int(engine.df['Kalori'].mean())
    
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
    value=kalori_mean,
    step=5,
    help=f"Pilih target kalori (tersedia: {kalori_min}-{kalori_max} kcal)\n\n💡 Menggunakan Gaussian Scoring untuk matching kalori yang lebih smooth"
)

# 2️⃣ Pilih Kategori Lauk (Dynamic dari Dataset)
if engine and not engine.df.empty:
    # ✅ FIX: Gunakan kolom 'Kategori_Lauk' (bukan 'Kategori')
    kategori_options = sorted(engine.df['Kategori_Lauk'].dropna().unique().tolist())
else:
    kategori_options = ["ayam", "ikan", "sapi"]

kategori_lauk = st.sidebar.selectbox(
    "Pilih Jenis Lauk",
    options=kategori_options,
    help="Pilih satu jenis lauk favorit Anda\n\n💡 Menggunakan Fuzzy Matching untuk toleransi variasi nama"
)

# 3️⃣ Pilih Sumber Karbohidrat
if engine and not engine.df.empty:
    # ✅ FIX: Gunakan kolom 'Sumber_Karbohidrat'
    karbo_raw = engine.df['Sumber_Karbohidrat'].dropna().str.lower().str.split()
    all_karbo = set([item for sublist in karbo_raw for item in sublist])
    karbo_options = sorted(list(all_karbo))
else:
    karbo_options = ["nasi merah", "nasi putih", "kentang", "ubi", "jagung"]

pilihan_karbo = st.sidebar.multiselect(
    "Pilih Sumber Karbohidrat",
    options=karbo_options,
    default=[karbo_options[0]] if karbo_options else [],
    help="Pilih sumber karbohidrat yang Anda inginkan\n\n💡 Engine akan mencari menu yang mengandung karbohidrat ini"
)

# 4️⃣ Deskripsi Preferensi (Text Input)
st.sidebar.markdown("**🏷️ Tag Populer (klik untuk tambahkan):**")

# Initialize session state for description
if 'deskripsi_pref' not in st.session_state:
    st.session_state.deskripsi_pref = "rendah lemak"

col1, col2 = st.sidebar.columns(2)

with col1:
    st.caption("✅ **Yang Diinginkan:**")
    if st.button("🟢 Rendah Lemak", key="tag1"):
        st.session_state.deskripsi_pref = "rendah lemak"
    if st.button("🟢 Tinggi Protein", key="tag2"):
        st.session_state.deskripsi_pref = "tinggi protein"
    if st.button("🟢 Kukus", key="tag3"):
        st.session_state.deskripsi_pref = "kukus"

with col2:
    st.caption("❌ **Yang Dihindari:**")
    if st.button("🔴 Tanpa Santan", key="tag4"):
        st.session_state.deskripsi_pref = "tanpa santan"
    if st.button("🔴 Tanpa Tempe", key="tag5"):
        st.session_state.deskripsi_pref = "tanpa tempe"
    if st.button("🔴 Tidak Pedas", key="tag6"):
        st.session_state.deskripsi_pref = "tidak pedas"

deskripsi_pref = st.sidebar.text_area(
    "Atau Ketik Manual:",
    value=st.session_state.deskripsi_pref,
    height=80,
    help="""
    💡 **Tips Pencarian dengan TF-IDF Bigram:**
    
    ✅ **Keyword Penting (akan diberi bobot tinggi):**
    - Rasa: pedas, manis, gurih, asam, asin
    - Metode: panggang, bakar, goreng, kukus, rebus, tumis
    - Tekstur: crispy, renyah, lembut, empuk
    - Nutrisi: rendah lemak, tinggi protein, tanpa santan
    
    ⚠️ **Hindari:**
    - Typo (gunakan "rendah" bukan "renda")
    - Bahasa campuran (pilih 1 bahasa saja)
    """
)

# ⚠️ WARNING jika deskripsi kosong
if not deskripsi_pref or deskripsi_pref.strip() == "":
    st.sidebar.warning("⚠️ Deskripsi kosong! Rekomendasi hanya berdasarkan kalori, lauk, dan karbo.")

# 5️⃣ Pilih Skenario Bobot (Advanced)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pengaturan Lanjutan")

skenario_bobot = st.sidebar.radio(
    "Pilih Mode Rekomendasi",
    options=["seimbang", "fokus_deskripsi", "fokus_lauk"],
    format_func=lambda x: {
        "seimbang": "🟢 Mode Seimbang (Recommended - 80.01%)",
        "fokus_deskripsi": "🟠 Mode Fokus Deskripsi (80.01%)",
        "fokus_lauk": "🔵 Mode Fokus Lauk (77.41%)"
    }[x],
    help="""
    🟢 Mode Seimbang: F1-Score 80.01% ✅ (paling akurat & balanced)
    🟠 Mode Fokus Deskripsi: F1-Score 80.01% ✅ (optimal untuk preferensi detail)
    🔵 Mode Fokus Lauk: F1-Score 77.41% (prioritas jenis protein)
    
    💡 Mode Seimbang & Fokus Deskripsi memiliki performa sama karena
    menggunakan balanced weight distribution yang optimal.
    """
)

# Tampilkan performance metrics (UPDATED)
performance_metrics = {
    "seimbang": {"precision": 0.8111, "recall": 0.8000, "f1": 0.8001},
    "fokus_deskripsi": {"precision": 0.8111, "recall": 0.8000, "f1": 0.8001},
    "fokus_lauk": {"precision": 0.7878, "recall": 0.7744, "f1": 0.7741}
}

metrics = performance_metrics[skenario_bobot]
st.sidebar.caption(f"📊 **Performance Mode {skenario_bobot.title()}:**")
st.sidebar.caption(f"Precision: {metrics['precision']:.2%} | Recall: {metrics['recall']:.2%} | F1: {metrics['f1']:.2%}")

# Tampilkan bobot yang digunakan
if engine:
    weights = engine.modes[skenario_bobot]
    with st.sidebar.expander("📋 Lihat Detail Bobot Kriteria"):
        st.markdown("**Bobot yang Digunakan:**")
        
        for key, value in weights.items():
            label = key.replace('w_', '').replace('_', ' ').title()
            st.write(f"**{label}**: {value:.0%}")
            st.progress(value)
        
        st.caption("💡 Total bobot = 100%")

# Jumlah rekomendasi
top_n = st.sidebar.slider(
    "Jumlah Rekomendasi",
    min_value=3,
    max_value=20,
    value=5,
    help="Berapa menu yang ingin ditampilkan?"
)

# Debug mode
show_debug = st.sidebar.checkbox(
    "🔍 Mode Debug (Tampilkan Detail Skor)",
    help="Tampilkan breakdown skor similarity per kriteria untuk troubleshooting"
)

# ✅ TAMBAHAN: Dataset Diagnostic Tool
if st.sidebar.checkbox("🛠️ Dataset Diagnostic", help="Cek kolom dan data karbohidrat"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Info Dataset")
    
    if engine and not engine.df.empty:
        # Cek kolom
        st.sidebar.caption(f"**Total Menu:** {len(engine.df)}")
        st.sidebar.caption(f"**Kolom:** {', '.join(engine.df.columns.tolist())}")
        
        # Cek karbohidrat
        if 'Sumber_Karbohidrat' in engine.df.columns:
            karbo_count = engine.df['Sumber_Karbohidrat'].notna().sum()
            karbo_null = engine.df['Sumber_Karbohidrat'].isna().sum()
            
            st.sidebar.caption(f"**Karbo Terisi:** {karbo_count}/{len(engine.df)}")
            st.sidebar.caption(f"**Karbo Kosong:** {karbo_null}")
            
            # Sample data
            sample = engine.df[engine.df['Sumber_Karbohidrat'].notna()]['Sumber_Karbohidrat'].head(3).tolist()
            st.sidebar.caption(f"**Sample:** {sample}")
        else:
            st.sidebar.error("❌ Kolom 'Sumber_Karbohidrat' TIDAK ADA!")
        
        # Cek deskripsi
        if 'Deskripsi_Menu' in engine.df.columns:
            desc_count = engine.df['Deskripsi_Menu'].notna().sum()
            st.sidebar.caption(f"**Deskripsi Terisi:** {desc_count}/{len(engine.df)}")
        else:
            st.sidebar.error("❌ Kolom 'Deskripsi_Menu' TIDAK ADA!")
    else:
        st.sidebar.error("❌ Engine belum load!")

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
        st.error("❌ Model belum ter-load. Pastikan file data ada di `data/Preprocessing/data_preprocessed.csv`")
    else:
        with st.spinner("🔍 Mencari menu terbaik untuk Anda..."):
            try:
                # ✅ FIX: Panggil dengan parameter yang benar
                recommendations = engine.get_recommendations(
                    kalori_target=kalori_target,
                    kategori_lauk=kategori_lauk,
                    sumber_karbo_list=pilihan_karbo,
                    deskripsi_preferensi=deskripsi_pref,
                    top_n=top_n,
                    mode=skenario_bobot  # ✅ FIX: Gunakan parameter 'mode' (bukan weights)
                )
                
                # Tampilkan ringkasan input user
                st.success("✅ Rekomendasi berhasil dibuat!")
                
                # ✅ VALIDASI: Cek apakah deskripsi diisi
                desc_status = "✅ Diisi" if deskripsi_pref and deskripsi_pref.strip() else "⚠️ Kosong"
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🔥 Target Kalori", f"{kalori_target} kcal")
                with col2:
                    st.metric("🍖 Jenis Lauk", kategori_lauk.title())
                with col3:
                    jumlah_karbo = len(pilihan_karbo) if pilihan_karbo else 0
                    st.metric("🍚 Pilihan Karbo", f"{jumlah_karbo} jenis")
                with col4:
                    st.metric("⚙️ Mode", skenario_bobot.title())
                
                # ✅ Info tambahan
                col_a, col_b = st.columns(2)
                with col_a:
                    if pilihan_karbo:
                        st.caption(f"🍚 **Karbohidrat yang dipilih:** {', '.join([k.title() for k in pilihan_karbo])}")
                    else:
                        st.caption("🍚 **Karbohidrat:** Semua (tidak ada filter)")
                
                with col_b:
                    st.caption(f"📝 **Status Deskripsi:** {desc_status}")
                    if desc_status == "⚠️ Kosong":
                        st.caption("   └─ Skor deskripsi & keyword boost = 0")
                    else:
                        st.caption(f"   └─ Input: '{deskripsi_pref[:30]}...'")
                
                st.markdown("---")
                st.subheader(f"🏆 Top-{top_n} Menu yang Direkomendasikan")
                
                # Validasi hasil
                if recommendations.empty:
                    st.warning("⚠️ Tidak ada menu yang cocok dengan kriteria Anda. Coba ubah preferensi.")
                else:
                    # Info range kalori hasil
                    kalori_results = recommendations['Kalori'].tolist()
                    min_kal_result = min(kalori_results)
                    max_kal_result = max(kalori_results)
                    
                    st.caption(f"📊 **Range kalori hasil:** {min_kal_result} - {max_kal_result} kcal")
                    
                    # Warning jika tidak ada menu dengan kalori exact
                    if kalori_target not in kalori_results:
                        closest_kalori = min(kalori_results, key=lambda x: abs(x - kalori_target))
                        diff = abs(closest_kalori - kalori_target)
                        
                        st.info(
                            f"💡 **Menu dengan kalori terdekat:** {closest_kalori} kcal (selisih {diff} kcal)\n\n"
                            f"Sistem menggunakan Gaussian scoring dengan toleransi ±30 kcal untuk hasil yang lebih presisi."
                        )
                    
                    # ✅ FIX: Reset index untuk ranking yang benar
                    recommendations_display = recommendations.reset_index(drop=True)
                    
                    # Tampilkan hasil
                    for idx, row in recommendations_display.iterrows():
                        rank = idx + 1  # ✅ Sekarang rank akan 1, 2, 3, 4, 5...
                        final_score = row['Final_Score']
                        nama_menu = row.get('Nama_Menu', 'N/A')
                        kategori = row.get('Kategori_Lauk', 'N/A')
                        kalori = row.get('Kalori', 'N/A')
                        karbo = row.get('Sumber_Karbohidrat', 'N/A')
                        deskripsi = row.get('Deskripsi_Menu', 'Tidak ada deskripsi')  # ✅ TAMBAHKAN INI
                        
                        # Card dengan styling
                        with st.container():
                            col_rank, col_info = st.columns([1, 9])
                            
                            with col_rank:
                                st.markdown(f"### #{rank}")
                            
                            with col_info:
                                st.markdown(f"**{nama_menu.title()}**")
                                
                                # ✅ HIDE technical score, show user-friendly info instead
                                st.caption(f"Kategori: {kategori.title()} | Kalori: {kalori} kcal")
                                
                                # ✅ FIX: Tampilkan deskripsi
                                if deskripsi and deskripsi != 'Tidak ada deskripsi':
                                    st.write(f"📝 {deskripsi.capitalize()}")
                                
                                # ✅ FIX: Parsing karbohidrat yang lebih baik (handle multi-word)
                                if karbo and karbo != 'N/A' and str(karbo).strip() and str(karbo).lower() != 'nan':
                                    # Parse dengan handle multi-word items
                                    karbo_str = str(karbo).lower().strip()
                                    
                                    # Definisi multi-word items yang harus dijaga utuh
                                    multi_word_items = [
                                        'nasi merah', 'nasi putih', 'nasi coklat', 
                                        'roti gandum', 'kentang tumbuk', 'ubi jalar',
                                        'jagung manis', 'oat meal'
                                    ]
                                    
                                    # Extract multi-word items dulu
                                    karbo_list = []
                                    for item in multi_word_items:
                                        if item in karbo_str:
                                            karbo_list.append(item.title())
                                            karbo_str = karbo_str.replace(item, '')  # Remove dari string
                                    
                                    # Kemudian ambil single-word items dari sisa string
                                    single_words = karbo_str.split()
                                    for word in single_words:
                                        word = word.strip()
                                        if len(word) > 2 and word not in ['dan', 'atau', 'dengan']:
                                            karbo_list.append(word.title())
                                    
                                    # Remove duplicates sambil maintain order
                                    karbo_list = list(dict.fromkeys(karbo_list))
                                    
                                    # Highlight karbo yang cocok pilihan user
                                    karbo_display = []
                                    user_karbo_lower = [k.lower() for k in pilihan_karbo]
                                    
                                    for item in karbo_list:
                                        if item.lower() in user_karbo_lower:
                                            karbo_display.append(f"`{item}` ✅")
                                        else:
                                            karbo_display.append(f"`{item}`")
                                    
                                    if karbo_display:
                                        st.markdown(f"🍚 **Karbohidrat:** {' '.join(karbo_display)}")
                                    else:
                                        st.markdown(f"🍚 **Karbohidrat:** _Data tidak valid_ (raw: `{karbo}`)")
                                else:
                                    # ✅ Tampilkan pesan debug yang lebih informatif
                                    raw_val = str(karbo) if karbo else 'None'
                                    st.markdown(f"🍚 **Karbohidrat:** _Tidak ada data_ (raw: `{raw_val}`)")
                                    st.caption("   └─ ⚠️ Data karbohidrat kosong di dataset untuk menu ini")
                                
                                # Debug scores (only show in debug mode)
                                if show_debug:
                                    st.caption(f"📊 **Technical Scores:**")
                                    st.caption(f"   • Total Score: {final_score:.3f}")
                                    st.caption(f"   • Kalori Match: {row['Score_Kalori']:.3f}")
                                    st.caption(f"   • Lauk Match: {row['Score_Lauk']:.3f}")
                                    st.caption(f"   • Karbo Match: {row['Score_Karbo']:.3f}")
                                    st.caption(f"   • Deskripsi Match: {row['Score_Deskripsi']:.3f}")
                                    st.caption(f"   • Keyword Boost: {row['Keyword_Boost']:.3f}")
                            
                            st.markdown("---")
                    
                    # Tombol download hasil
                    csv = recommendations.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil Rekomendasi (CSV)",
                        data=csv,
                        file_name=f'rekomendasi_menu_{skenario_bobot}.csv',
                        mime='text/csv',
                    )
                    
                    # Debug table
                    if show_debug:
                        st.markdown("---")
                        st.subheader("🔍 Debug Mode: Tabel Detail Skor")
                        
                        debug_df = recommendations[[
                            'Nama_Menu', 'Final_Score', 'Score_Kalori', 
                            'Score_Lauk', 'Score_Karbo', 'Score_Deskripsi', 'Keyword_Boost'
                        ]].copy()
                        
                        st.dataframe(debug_df, use_container_width=True)
                        
                        st.caption("💡 **Interpretasi:**")
                        st.caption("- Final Score = weighted sum dari semua kriteria + keyword boost")
                        st.caption("- Gaussian scoring (sigma=30) untuk Score_Kalori")
                        st.caption("- Fuzzy matching untuk Score_Lauk dan Score_Karbo")
                        st.caption("- TF-IDF + Bigram untuk Score_Deskripsi")
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.exception(e)

else:
    # Tampilan default sebelum tombol diklik
    st.info("👈 Masukkan preferensi Anda di sidebar, lalu klik **Dapatkan Rekomendasi**")
    
    # Tampilkan preview dataset
    st.subheader("📊 Preview Dataset Menu")
    if engine and not engine.df.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Total Menu Tersedia:** {len(engine.df)} menu")
        with col2:
            show_all = st.checkbox("Tampilkan Semua", value=False)
        
        # ✅ FIX: Gunakan nama kolom yang benar
        display_df = engine.df[['Nama_Menu', 'Kategori_Lauk', 'Kalori', 'Sumber_Karbohidrat']]
        
        if show_all:
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.dataframe(display_df.head(10), use_container_width=True)
        
        # Statistik dataset
        with st.expander("📊 Lihat Statistik Dataset"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Menu", len(engine.df))
                st.metric("Kategori Lauk", engine.df['Kategori_Lauk'].nunique())
            
            with col2:
                st.metric("Kalori Min", f"{engine.df['Kalori'].min():.0f} kcal")
                st.metric("Kalori Max", f"{engine.df['Kalori'].max():.0f} kcal")
            
            with col3:
                st.metric("Kalori Rata-rata", f"{engine.df['Kalori'].mean():.0f} kcal")
                st.metric("Total Karbo Unik", engine.df['Sumber_Karbohidrat'].str.split().explode().nunique())

# ========================================
# FOOTER (UPDATED)
# ========================================
st.markdown("---")
st.caption("🔬 Sistem Rekomendasi MCCBF v2.0 - Optimized with Gaussian Scoring + Fuzzy Matching + TF-IDF Bigram")
st.caption("📊 **System Performance:** Mode Seimbang & Fokus Deskripsi (Precision: 81.11%, Recall: 80.00%, **F1: 80.01%** ✅)")
st.caption("⚙️ **Optimization:** sigma=30 (calorie matching), balanced weights (0.35|0.30|0.25|0.10)")
st.caption("🏫 Icel's Room Kitchen | UIN Sunan Gunung Djati Bandung")