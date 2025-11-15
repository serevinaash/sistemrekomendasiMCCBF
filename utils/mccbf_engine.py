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
    
    
    def preprocess_user_input(self, text):
        """
        Preprocessing khusus untuk input user dengan handling negatif
        
        Transformasi:
        - "tanpa tempe" → hapus kata "tempe" dari pencarian
        - "tidak pedas" → cari "tidak pedas" atau favoritkan menu tanpa kata "pedas"
        - "rendah lemak" → tambah boost untuk "rendah lemak"
        """
        text = text.lower().strip()
        
        # Deteksi kata negatif
        negative_keywords = ['tanpa', 'tidak', 'no', 'bebas', 'tanpai']
        
        # Split jadi kata-kata
        words = text.split()
        
        # Identifikasi pola negatif
        negative_terms = []
        positive_terms = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Jika ketemu kata negatif, ambil kata berikutnya
            if word in negative_keywords and i + 1 < len(words):
                negative_terms.append(words[i + 1])
                i += 2  # Skip 2 kata
            else:
                positive_terms.append(word)
                i += 1
        
        return {
            'positive': ' '.join(positive_terms),
            'negative': negative_terms,
            'original': text
        }
    
    
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
    
    
    def _format_karbohidrat(self, karbo_str):
        """
        Format string karbohidrat jadi lebih rapi dengan koma
        Input: "kentang nasi merah nasi putih"
        Output: "Kentang, Nasi Merah, Nasi Putih"
        """
        if not karbo_str or pd.isna(karbo_str):
            return "-"
        
        karbo_str = str(karbo_str).strip().lower()
        
        # Definisi pola karbohidrat yang valid (2 kata atau 1 kata)
        valid_patterns = [
            'nasi merah', 'nasi putih', 'nasi coklat', 'nasi jagung',
            'roti gandum', 'roti tawar'
        ]
        
        found = []
        remaining = karbo_str
        
        # Cari pola 2 kata dulu
        for pattern in valid_patterns:
            if pattern in remaining:
                found.append(pattern.title())  # Capitalize each word
                remaining = remaining.replace(pattern, '')
        
        # Ambil sisa kata tunggal
        tokens = [t.strip().title() for t in remaining.split() if len(t.strip()) > 2]
        found.extend(tokens)
        
        # Remove duplicates sambil pertahankan urutan
        seen = set()
        unique = []
        for item in found:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique.append(item)
        
        return ', '.join(unique) if unique else "-"
    
    
    def _calculate_karbo_similarity(self, user_karbo_list, menu_row):
        """
        Helper untuk menghitung similarity karbohidrat
        Mendukung dua format:
        1. Jika ada kolom 'karbo_list' (hasil preprocessing)
        2. Fallback ke parsing manual dari 'Sumber_Karbohidrat'
        """
        # Format 1: Sudah ada kolom karbo_list (optimal)
        if 'karbo_list' in menu_row.index:
            menu_karbo_list = menu_row['karbo_list']
            if isinstance(menu_karbo_list, str):
                # Jika masih string (dari CSV), parse dulu
                import ast
                try:
                    menu_karbo_list = ast.literal_eval(menu_karbo_list)
                except:
                    menu_karbo_list = []
            
            if not user_karbo_list:
                return 1.0
            
            user_set = set([k.lower().strip() for k in user_karbo_list])
            menu_set = set([k.lower().strip() for k in menu_karbo_list])
            
            # Hitung intersection
            intersection = user_set.intersection(menu_set)
            if intersection:
                return len(intersection) / len(user_set)
            else:
                return 0.0
        
        # Format 2: Fallback ke parsing manual
        else:
            return self.karbo_partial_match(user_karbo_list, menu_row['Sumber_Karbohidrat'])
    
    
    def karbo_partial_match(self, user_karbo_list, menu_karbo_str):
        """
        Partial matching untuk sumber karbohidrat (multi-select)
        Menggunakan Set Intersection untuk akurasi lebih tinggi
        
        Args:
            user_karbo_list: list preferensi user, e.g. ['nasi merah', 'kentang']
            menu_karbo_str: string dari menu, e.g. 'kentang nasi merah nasi putih'
        
        Returns:
            Skor similarity berdasarkan:
            - 1.0 jika ada exact match
            - 0.5 jika ada partial match
            - 0.0 jika tidak ada match sama sekali
        """
        if not user_karbo_list:
            return 1.0  # tidak ada preferensi = semua cocok
        
        # Parse menu karbo string jadi list
        menu_karbo_str = str(menu_karbo_str).lower().strip()
        
        # Split berdasarkan spasi/koma dan normalisasi
        menu_karbo_tokens = set([
            token.strip() 
            for token in menu_karbo_str.replace(',', ' ').split() 
            if token.strip()
        ])
        
        # Normalisasi user input
        user_karbo_set = set([k.lower().strip() for k in user_karbo_list])
        
        # --- Strategi Matching ---
        
        # 1️⃣ Exact Match (prioritas tertinggi)
        exact_matches = user_karbo_set.intersection(menu_karbo_tokens)
        if exact_matches:
            # Semakin banyak match, semakin tinggi skornya
            return min(len(exact_matches) / len(user_karbo_set), 1.0)
        
        # 2️⃣ Partial Match (untuk frasa multi-kata seperti "nasi merah")
        partial_score = 0
        for user_pref in user_karbo_set:
            # Cek apakah preferensi user ada di string menu
            if user_pref in menu_karbo_str:
                partial_score += 0.7
            # Cek apakah ada kata yang sama (misal: "nasi" cocok dengan "nasi merah")
            elif any(word in menu_karbo_str for word in user_pref.split()):
                partial_score += 0.3
        
        # Normalisasi skor partial
        if partial_score > 0:
            return min(partial_score / len(user_karbo_set), 1.0)
        
        # 3️⃣ Tidak ada match sama sekali
        return 0.0
    
    
    def get_recommendations(self, 
                           kalori_target, 
                           kategori_lauk, 
                           sumber_karbo_list, 
                           deskripsi_preferensi,
                           weights=None,
                           top_n=5,
                           karbo_strict_mode=False):  # NEW parameter
        """
        Fungsi utama untuk mendapatkan rekomendasi menu
        
        Args:
            kalori_target: int, target kalori user (e.g. 400)
            kategori_lauk: str, pilihan user ('Ayam', 'Ikan', 'Daging')
            sumber_karbo_list: list of str, pilihan karbo (['nasi merah', 'kentang'])
            deskripsi_preferensi: str, teks bebas preferensi user
            weights: dict bobot kriteria, jika None pakai default
            top_n: jumlah rekomendasi yang dikembalikan
            karbo_strict_mode: bool, jika True filter strict (harus cocok semua)
        
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
        
        # Safety check: jika corpus kosong
        if not user_corpus.strip():
            user_corpus = "menu sehat"  # Default fallback
        
        user_tfidf = self.vectorizer.transform([user_corpus])
        sim_deskripsi = cosine_similarity(user_tfidf, self.tfidf_train)[0]
        
        # Pastikan tidak ada NaN
        sim_deskripsi = np.nan_to_num(sim_deskripsi, nan=0.0)
        
        
        # ========================================
        # 2️⃣ SIMILARITY KALORI (Normalized Distance)
        # ========================================
        # Cek apakah kolom kalori_normalized ada
        if 'kalori_normalized' not in self.df_train.columns:
            # Jika tidak ada, normalisasi manual
            kalori_min = self.df_train['Kalori_(kcal)'].min()
            kalori_max = self.df_train['Kalori_(kcal)'].max()
            
            if kalori_max > kalori_min:
                kalori_normalized = (kalori_target - kalori_min) / (kalori_max - kalori_min)
                menu_kalori_norm = (self.df_train['Kalori_(kcal)'] - kalori_min) / (kalori_max - kalori_min)
            else:
                # Semua kalori sama, return skor sempurna
                kalori_normalized = 1.0
                menu_kalori_norm = np.ones(len(self.df_train))
        else:
            # Gunakan kolom yang sudah ada
            kalori_normalized = self.scaler.transform([[kalori_target]])[0][0]
            menu_kalori_norm = self.df_train['kalori_normalized'].values
        
        # Hitung similarity (1 - jarak absolut)
        sim_kalori = 1 - np.abs(kalori_normalized - menu_kalori_norm)
        
        # Pastikan tidak ada nilai negatif
        sim_kalori = np.clip(sim_kalori, 0, 1)
        
        
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
            self._calculate_karbo_similarity(sumber_karbo_list, row)
            for _, row in self.df_train.iterrows()
        ])
        
        
        # ========================================
        # 5️⃣ GABUNGKAN SEMUA SIMILARITY (WEIGHTED AVERAGE)
        # ========================================
        
        # Clip semua similarity ke range [0, 1] untuk keamanan
        sim_deskripsi = np.clip(sim_deskripsi, 0, 1)
        sim_kategori = np.clip(sim_kategori, 0, 1)
        sim_karbo = np.clip(sim_karbo, 0, 1)
        sim_kalori = np.clip(sim_kalori, 0, 1)
        
        # Weighted average
        sim_total = (
            weights['deskripsi'] * sim_deskripsi +
            weights['kategori'] * sim_kategori +
            weights['karbohidrat'] * sim_karbo +
            weights['kalori'] * sim_kalori
        )
        
        # Final clipping (double check)
        sim_total = np.clip(sim_total, 0, 1)
        
        
        # ========================================
        # 6️⃣ AMBIL TOP-N MENU DENGAN SKOR TERTINGGI
        # ========================================
        
        # Filter strict (opsional)
        if karbo_strict_mode and sumber_karbo_list:
            # Hanya ambil menu yang punya skor karbo > threshold
            valid_indices = np.where(sim_karbo >= 0.7)[0]
            
            if len(valid_indices) == 0:
                # Tidak ada menu yang cocok, kembalikan top-N biasa tapi beri warning
                print("⚠️  Mode Strict: Tidak ada menu yang sesuai kriteria. Menampilkan rekomendasi terbaik.")
                valid_indices = np.arange(len(sim_total))
            
            # Filter similarity matrix
            sim_total_filtered = sim_total[valid_indices]
            top_indices_filtered = sim_total_filtered.argsort()[::-1][:top_n]
            top_indices = valid_indices[top_indices_filtered]
        else:
            # Mode normal
            top_indices = sim_total.argsort()[::-1][:top_n]
        
        # Kolom yang ingin ditampilkan
        display_cols = ['Nama_Menu', 'Kategori', 'Kalori_(kcal)', 
                       'Sumber_Karbohidrat', 'Deskripsi_Singkat']
        
        # Pastikan kolom ada di dataframe
        available_cols = [col for col in display_cols if col in self.df_train.columns]
        
        result = self.df_train.iloc[top_indices][available_cols].copy()
        result['Skor_Similarity'] = sim_total[top_indices]
        result['Rank'] = range(1, len(top_indices) + 1)
        
        # ✨ FORMAT KARBOHIDRAT DENGAN KOMA
        if 'Sumber_Karbohidrat' in result.columns:
            result['Sumber_Karbohidrat'] = result['Sumber_Karbohidrat'].apply(
                lambda x: self._format_karbohidrat(x)
            )
        
        # Urutkan kolom
        result = result[['Rank', 'Skor_Similarity'] + available_cols]
        
        # DEBUG: Print statistik similarity (opsional, bisa dinonaktifkan)
        print(f"\n📊 Debug Info:")
        print(f"   • Skor Deskripsi: min={sim_deskripsi.min():.3f}, max={sim_deskripsi.max():.3f}")
        print(f"   • Skor Kategori:  min={sim_kategori.min():.3f}, max={sim_kategori.max():.3f}")
        print(f"   • Skor Karbo:     min={sim_karbo.min():.3f}, max={sim_karbo.max():.3f}")
        print(f"   • Skor Kalori:    min={sim_kalori.min():.3f}, max={sim_kalori.max():.3f}")
        print(f"   • Skor Total:     min={sim_total.min():.3f}, max={sim_total.max():.3f}")
        
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