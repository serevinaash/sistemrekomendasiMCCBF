# utils/mccbf_engine.py
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

class MCCBFEngine:
    """
    Engine untuk sistem rekomendasi Multi-Criteria Content-Based Filtering
    """
    
    def __init__(self, vectorizer_path, scaler_path, data_train_path):
        """
        Load model artifacts yang sudah di-training sebelumnya
        
        Args:
            vectorizer_path: path ke vectorizer_tfidf.pkl
            scaler_path: path ke scaler.pkl
            data_train_path: path ke data_train.csv
        """
        self.vectorizer = joblib.load(vectorizer_path)
        self.scaler = joblib.load(scaler_path)
        self.df_train = pd.read_csv(data_train_path)
        
        # Normalisasi nama kolom (ganti spasi dengan underscore)
        self.df_train.columns = self.df_train.columns.str.strip().str.replace(' ', '_')
        
        # Pre-compute TF-IDF untuk semua menu di training data
        if 'corpus' not in self.df_train.columns:
            self.df_train['corpus'] = self._create_corpus(self.df_train)
        
        self.tfidf_train = self.vectorizer.transform(self.df_train['corpus'])
        
        print("✅ MCCBF Engine berhasil diinisialisasi")
        print(f"   📊 Jumlah menu: {len(self.df_train)}")
    
    
    def _create_corpus(self, df):
        """Buat corpus text dari kolom-kolom penting"""
        text_cols = ['Nama_Menu', 'Kategori', 'Sumber_Karbohidrat', 
                     'Bahan_Utama_/_Pendamping', 'Deskripsi_Singkat']
        
        corpus = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_cols:
                if col in df.columns:
                    text_parts.append(str(row[col]))
            corpus.append(' '.join(text_parts))
        
        return corpus
    
    
    def clean_text(self, text):
        """Pembersihan teks standar"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    
    def categorical_similarity(self, user_value, menu_value):
        """
        Rule-based similarity untuk kategori (exact/partial match)
        
        Returns:
            1.0 jika exact match
            0.5 jika partial match
            0.0 jika tidak match
        """
        user_value = str(user_value).lower()
        menu_value = str(menu_value).lower()
        
        if user_value == menu_value:
            return 1.0
        if any(word in menu_value for word in user_value.split()):
            return 0.5
        return 0.0
    
    
    def karbo_partial_match(self, user_karbo_list, menu_karbo_str):
        """
        Partial matching untuk sumber karbohidrat (multi-select)
        
        Args:
            user_karbo_list: list preferensi user, e.g. ['nasi merah', 'kentang']
            menu_karbo_str: string dari menu, e.g. 'nasi merah nasi putih'
        
        Returns:
            ratio jumlah match / jumlah preferensi user
        """
        if not user_karbo_list:
            return 1.0  # tidak ada preferensi = semua cocok
        
        menu_karbo_str = str(menu_karbo_str).lower()
        match_count = sum(1 for k in user_karbo_list if k.lower() in menu_karbo_str)
        return match_count / len(user_karbo_list)
    
    
    def get_recommendations(self, 
                           kalori_target, 
                           kategori_lauk, 
                           sumber_karbo_list, 
                           deskripsi_preferensi,
                           weights=None,
                           top_n=5):
        """
        Fungsi utama untuk mendapatkan rekomendasi menu
        
        Args:
            kalori_target: int, target kalori user (e.g. 400)
            kategori_lauk: str, pilihan user ('Ayam', 'Ikan', 'Daging')
            sumber_karbo_list: list of str, pilihan karbo (['nasi merah', 'kentang'])
            deskripsi_preferensi: str, teks bebas preferensi user
            weights: dict bobot kriteria, jika None pakai default
            top_n: jumlah rekomendasi yang dikembalikan
        
        Returns:
            DataFrame berisi top-N menu dengan skor similarity
        """
        
        # Default weights (Skenario 1: Bobot Seimbang)
        if weights is None:
            weights = {
                'deskripsi': 0.25,
                'kategori': 0.25,
                'karbohidrat': 0.20,
                'kalori': 0.30
            }
        
        # ========================================
        # 1️⃣ SIMILARITY DESKRIPSI (TF-IDF + Cosine)
        # ========================================
        user_corpus = self.clean_text(deskripsi_preferensi)
        user_tfidf = self.vectorizer.transform([user_corpus])
        sim_deskripsi = cosine_similarity(user_tfidf, self.tfidf_train)[0]
        
        
        # ========================================
        # 2️⃣ SIMILARITY KALORI (Normalized Distance)
        # ========================================
        # Normalisasi kalori user menggunakan scaler yang sama
        kalori_normalized = self.scaler.transform([[kalori_target]])[0][0]
        
        menu_kalori_norm = self.df_train['kalori_normalized'].values
        sim_kalori = 1 - np.abs(kalori_normalized - menu_kalori_norm)
        
        
        # ========================================
        # 3️⃣ SIMILARITY KATEGORI LAUK (Rule-based)
        # ========================================
        sim_kategori = np.array([
            self.categorical_similarity(kategori_lauk, row['Kategori'])
            for _, row in self.df_train.iterrows()
        ])
        
        
        # ========================================
        # 4️⃣ SIMILARITY SUMBER KARBOHIDRAT (Partial Match)
        # ========================================
        sim_karbo = np.array([
            self.karbo_partial_match(sumber_karbo_list, row['Sumber_Karbohidrat'])
            for _, row in self.df_train.iterrows()
        ])
        
        
        # ========================================
        # 5️⃣ GABUNGKAN SEMUA SIMILARITY (WEIGHTED AVERAGE)
        # ========================================
        sim_total = (
            weights['deskripsi'] * sim_deskripsi +
            weights['kategori'] * sim_kategori +
            weights['karbohidrat'] * sim_karbo +
            weights['kalori'] * sim_kalori
        )
        
        
        # ========================================
        # 6️⃣ AMBIL TOP-N MENU DENGAN SKOR TERTINGGI
        # ========================================
        top_indices = sim_total.argsort()[::-1][:top_n]
        
        # Kolom yang ingin ditampilkan
        display_cols = ['Nama_Menu', 'Kategori', 'Kalori_(kcal)', 
                       'Sumber_Karbohidrat', 'Deskripsi_Singkat']
        
        # Pastikan kolom ada di dataframe
        available_cols = [col for col in display_cols if col in self.df_train.columns]
        
        result = self.df_train.iloc[top_indices][available_cols].copy()
        result['Skor_Similarity'] = sim_total[top_indices]
        result['Rank'] = range(1, top_n + 1)
        
        # Urutkan kolom
        result = result[['Rank', 'Skor_Similarity'] + available_cols]
        
        return result


# ========================================
# FUNGSI HELPER UNTUK STREAMLIT
# ========================================

def load_mccbf_engine(model_dir='model'):
    """Load MCCBF Engine dengan path default"""
    import os
    
    vectorizer_path = os.path.join(model_dir, 'vectorizer_tfidf.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    data_train_path = os.path.join(model_dir, 'data_train.csv')
    
    engine = MCCBFEngine(vectorizer_path, scaler_path, data_train_path)
    return engine