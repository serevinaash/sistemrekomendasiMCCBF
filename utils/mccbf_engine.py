import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MCCBFEngine:
    def __init__(self, data_path=None, dataframe=None):
        """
        Inisialisasi Engine MCCBF.
        Bisa load dari path CSV atau langsung dari DataFrame.
        
        Args:
            data_path: Path ke file CSV data menu
            dataframe: DataFrame yang sudah di-load
        """
        # Load data
        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_path and os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            print(f"Warning: Path '{data_path}' tidak ditemukan. Menggunakan DataFrame kosong.")
            self.df = pd.DataFrame(columns=['Menu_ID', 'Nama_Menu', 'Kalori', 'Kategori_Lauk', 
                                           'Sumber_Karbohidrat', 'Deskripsi_Menu'])
        
        # Preprocessing awal
        self._preprocess_data()
        
        # Inisialisasi TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words=None)
        
        # Fit vectorizer (cegah error jika data kosong)
        if not self.df.empty:
            descriptions = self.df['Deskripsi_Menu'].fillna('').astype(str)
            self.tfidf_matrix = self.vectorizer.fit_transform(descriptions)
            self.min_calories = self.df['Kalori'].min()
            self.max_calories = self.df['Kalori'].max()
        else:
            self.tfidf_matrix = None
            self.min_calories = 0
            self.max_calories = 0

    def _preprocess_data(self):
        """Preprocessing kolom data"""
        if self.df.empty: 
            return
        
        # Print kolom untuk debugging
        print(f"✅ Kolom CSV: {list(self.df.columns)}")
        
        # Rename kolom agar konsisten
        column_mapping = {
            'Kalori (kcal)': 'Kalori',
            'Nama Menu': 'Nama_Menu',
            'Kategori': 'Kategori_Lauk',
            'Sumber Karbohidrat': 'Sumber_Karbohidrat',
            'Deskripsi Singkat': 'Deskripsi_Menu',
            'No': 'Menu_ID'
        }
        
        self.df.rename(columns=column_mapping, inplace=True)
        
        # Pastikan kolom numerik aman
        self.df['Kalori'] = pd.to_numeric(self.df['Kalori'], errors='coerce').fillna(0)
        
        # Pastikan kolom teks aman & lowercase
        text_cols = ['Kategori_Lauk', 'Sumber_Karbohidrat', 'Deskripsi_Menu', 'Nama_Menu']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().str.strip().replace('nan', '')

    def _calculate_calorie_score(self, item_cal, user_cal, tolerance=30):
        """
        Hitung skor kalori dengan toleransi
        
        Args:
            item_cal: Kalori item menu
            user_cal: Target kalori user
            tolerance: Toleransi kalori (default 40)
        
        Returns:
            Score 0-1
        """
        diff = abs(user_cal - item_cal)
        
        # Jika dalam toleransi, score = 1.0
        if diff <= tolerance:
            return 1.0
        
        # Normalisasi berdasarkan range
        cal_range = self.max_calories - self.min_calories
        if cal_range == 0: 
            return 1.0
        
        normalized_diff = diff / cal_range
        return max(0.0, 1.0 - normalized_diff)

    def _calculate_category_score(self, item_val, user_val):
        """
        Hitung skor kategori dengan fuzzy matching
        
        Args:
            item_val: Nilai kategori item
            user_val: Nilai kategori yang dicari user
        
        Returns:
            Score 0-1
        """
        # Jika user tidak specify, berikan score netral
        if not user_val or user_val == 'nan' or user_val == '':
            return 0.5 
        
        item_str = str(item_val).lower()
        user_str = str(user_val).lower()
        
        # Exact match
        if user_str == item_str:
            return 1.0
        
        # Fuzzy match (contains)
        if user_str in item_str or item_str in user_str:
            return 1.0
        
        return 0.0
    
    def _calculate_keyword_boost(self, item_desc, user_desc):
        """Boost lebih agresif"""
        if not user_desc or user_desc == 'nan':
            return 0.0
        
        # Keyword penting + tambahkan lebih banyak
        important_keywords = {
            # Rasa
            'pedas', 'manis', 'gurih', 'asam', 'asin', 'pahit',
            # Metode masak
            'panggang', 'bakar', 'goreng', 'kukus', 'rebus', 'tumis', 'crispy', 'grill',
            # Karakteristik
            'rendah', 'tinggi', 'tanpa', 'kuah', 'kering', 'bening',
            'renyah', 'lembut', 'empuk', 'segar', 'kaya', 'protein',
            'santan', 'lemak', 'minyak',
            # Bumbu/style
            'teriyaki', 'balado', 'sambal', 'woku', 'korea', 'yakiniku', 
            'bulgogi', 'curry', 'soto', 'rawon'
        }
        
        user_keywords = set(str(user_desc).lower().split())
        item_keywords = set(str(item_desc).lower().split())
        
        matched = user_keywords & item_keywords & important_keywords
        
        # BOOST LEBIH BESAR: 0.08 per keyword (max 0.3)
        return min(0.3, len(matched) * 0.08)

    def get_recommendations(self, kalori_target=None, kategori_lauk=None, 
                          sumber_karbo_list=None, deskripsi_preferensi=None,
                          user_preferences=None, weights=None, top_n=10):
        """
        Generate rekomendasi menu
        
        Args:
            kalori_target: Target kalori
            kategori_lauk: Kategori lauk (ayam/ikan/sapi)
            sumber_karbo_list: List sumber karbohidrat
            deskripsi_preferensi: Deskripsi preferensi user
            user_preferences: Dict alternatif {'kalori', 'lauk', 'karbo', 'deskripsi'}
            weights: Dict bobot kriteria
            top_n: Jumlah rekomendasi
        
        Returns:
            DataFrame berisi rekomendasi
        """
        if self.df.empty: 
            return pd.DataFrame()

        # Parse input: support both formats
        if user_preferences is not None:
            kalori_target = user_preferences.get('kalori', kalori_target)
            kategori_lauk = user_preferences.get('lauk', kategori_lauk)
            karbo = user_preferences.get('karbo', '')
            deskripsi_preferensi = user_preferences.get('deskripsi', deskripsi_preferensi)
        else:
            karbo = sumber_karbo_list[0] if sumber_karbo_list else ''

        # Default weights
        if weights is None:
            # Turunkan w_kalori, naikkan w_lauk & w_deskripsi
            weights = {
                'w_kalori': 0.30,      # Turun dari 0.35
                'w_lauk': 0.25,        # Naik dari 0.25
                'w_karbo': 0.20,       # Naik dari 0.20
                'w_deskripsi': 0.25    # Turun dari 0.20 (karena boost sudah kuat)
            }
        scores = []
        
        # Vectorize user query untuk TF-IDF
        user_desc_vec = None
        if deskripsi_preferensi and str(deskripsi_preferensi) != 'nan':
            try:
                user_desc_vec = self.vectorizer.transform([str(deskripsi_preferensi).lower()])
            except:
                user_desc_vec = None

        # Scoring untuk setiap menu
        for idx, row in self.df.iterrows():
            # Score kalori
            s_kalori = self._calculate_calorie_score(row['Kalori'], kalori_target)
            
            # Score lauk
            s_lauk = self._calculate_category_score(row['Kategori_Lauk'], kategori_lauk)
            
            # Score karbo
            s_karbo = self._calculate_category_score(row['Sumber_Karbohidrat'], karbo)
            
            # Score deskripsi (TF-IDF)
            s_deskripsi = 0.0
            if user_desc_vec is not None:
                try:
                    cosine_sim = cosine_similarity(user_desc_vec, self.tfidf_matrix[idx])[0][0]
                    s_deskripsi = cosine_sim
                except:
                    s_deskripsi = 0.0
            
            # Keyword boost
            keyword_boost = self._calculate_keyword_boost(row['Deskripsi_Menu'], deskripsi_preferensi)
            
            # Final score
            final_score = (s_kalori * weights['w_kalori']) + \
                          (s_lauk * weights['w_lauk']) + \
                          (s_karbo * weights['w_karbo']) + \
                          (s_deskripsi * weights['w_deskripsi']) + \
                          keyword_boost
            
            scores.append({
                'Menu_ID': row.get('Menu_ID', idx),
                'Nama_Menu': row.get('Nama_Menu', 'Unknown'),
                'Kalori': row.get('Kalori', 0),
                'Kategori_Lauk': row.get('Kategori_Lauk', ''),
                'Final_Score': final_score,
                'Score_Kalori': s_kalori,
                'Score_Lauk': s_lauk,
                'Score_Karbo': s_karbo,
                'Score_Deskripsi': s_deskripsi,
                'Keyword_Boost': keyword_boost
            })
        
        # Convert ke DataFrame dan sort
        results_df = pd.DataFrame(scores)
        if not results_df.empty:
            results_df = results_df.sort_values(by='Final_Score', ascending=False).head(top_n)
        
        return results_df