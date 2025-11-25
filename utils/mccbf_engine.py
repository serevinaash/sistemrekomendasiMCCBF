import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
STOPWORDS_ID = [
    "dan", "yang", "di", "ke", "dengan", "tanpa", "pakai",
    "serta", "untuk", "agar", "supaya", "adalah"
]


class MCCBFEngine:
    def __init__(self, data_path=None, dataframe=None):
        """
        Engine MCCBF dengan support 3 MODE:
        - seimbang
        - fokus_deskripsi
        - fokus_lauk
        """
        
        # =======================
        # LOAD DATA
        # =======================
        if dataframe is not None:
            self.df = dataframe.copy()
        elif data_path and os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            print(f"Warning: Path '{data_path}' tidak ditemukan.")
            self.df = pd.DataFrame(columns=[
                'Menu_ID', 'Nama_Menu', 'Kalori',
                'Kategori_Lauk', 'Sumber_Karbohidrat',
                'Deskripsi_Menu'
            ])

        # Preprocess data
        self._preprocess_data()

        # =======================
        # MODE SETTINGS
        # =======================
        self.modes = {
            'seimbang': {
                'w_deskripsi': 0.45,
                'w_lauk': 0.25,
                'w_karbo': 0.20,
                'w_kalori': 0.10
            },
            'fokus_deskripsi': {
                'w_deskripsi': 0.50,
                'w_lauk': 0.20,
                'w_karbo': 0.20,
                'w_kalori': 0.10
            },
            'fokus_lauk': {
                'w_deskripsi': 0.20,
                'w_lauk': 0.50,
                'w_karbo': 0.20,
                'w_kalori': 0.10
            }
        }

        # =======================
        # TF-IDF VECTOR
        # =======================
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            stop_words=STOPWORDS_ID,
            min_df=1,
            sublinear_tf=True
        )

        if not self.df.empty:
            desc = self.df['Deskripsi_Menu'].fillna('').astype(str)
            self.tfidf_matrix = self.vectorizer.fit_transform(desc)
            self.min_calories = self.df['Kalori'].min()
            self.max_calories = self.df['Kalori'].max()
        else:
            self.tfidf_matrix = None
            self.min_calories = 0
            self.max_calories = 0

    # =============================================================
    # PREPROCESSING
    # =============================================================
    def _preprocess_data(self):
        """Membersihkan dan menormalisasi dataset."""
        if self.df.empty: 
            return
        
        print(f"✅ Kolom CSV: {list(self.df.columns)}")

        # Rename kolom agar seragam
        mapping = {
            'Kalori (kcal)': 'Kalori',
            'Nama Menu': 'Nama_Menu',
            'Kategori': 'Kategori_Lauk',
            'Sumber Karbohidrat': 'Sumber_Karbohidrat',
            'Deskripsi Singkat': 'Deskripsi_Menu',
            'No': 'Menu_ID'
        }
        self.df.rename(columns=mapping, inplace=True)

        # Ensure numeric
        self.df['Kalori'] = pd.to_numeric(self.df['Kalori'], errors='coerce').fillna(0)

        # Normalisasi text
        text_cols = ['Kategori_Lauk', 'Sumber_Karbohidrat', 'Deskripsi_Menu', 'Nama_Menu']
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str)
                    .str.lower()
                    .str.strip()
                    .replace("nan", "")
                )

    # =============================================================
    # SCORING COMPONENTS
    # =============================================================
    def _calculate_calorie_score(self, item_cal, user_cal, tolerance=40):
        diff = abs(user_cal - item_cal)

        if diff <= tolerance:
            return 1.0

        cal_range = self.max_calories - self.min_calories
        if cal_range == 0: 
            return 1.0
        
        normalized_diff = diff / cal_range
        return max(0.0, 1.0 - normalized_diff)

    def _calculate_category_score(self, item_val, user_val):
        if not user_val or user_val == 'nan' or item_val == 'nan':
            return 0.5

        item = str(item_val).lower()
        user = str(user_val).lower()

        # Exact match
        if item == user:
            return 1.0

        # Partial match: kata pengguna ada di item
        if user in item or item in user:
            return 0.8

        # Soft similarity: huruf awal sama (ayam – ayam fillet, sapi – sapi lada)
        if item[0] == user[0]:
            return 0.5

        # Fallback
        return 0.0


    def _calculate_keyword_boost(self, item_desc, user_desc):
        """
        Boost sedang (0.08 per keyword, max 0.3)
        """
        if not user_desc or user_desc == 'nan':
            return 0.0

        important_keywords = {
            'pedas', 'manis', 'gurih', 'asam', 'asin',
            'panggang', 'bakar', 'goreng', 'kukus', 'rebus', 'tumis',
            'crispy', 'grill',
            'rendah', 'tinggi', 'tanpa', 'kuah', 'kering', 
            'bening', 'renyah', 'lembut', 'empuk', 'segar', 
            'protein', 'santan', 'lemak', 'minyak',
            'teriyaki', 'balado', 'sambal', 'woku', 'korea',
            'yakiniku', 'bulgogi', 'curry', 'soto', 'rawon'
        }

        user_kw = set(str(user_desc).lower().split())
        item_kw = set(str(item_desc).lower().split())

        matched = user_kw & item_kw & important_keywords

        return min(0.3, len(matched) * 0.08)

    # =============================================================
    # RECOMMENDATIONS CORE
    # =============================================================
    def get_recommendations(
        self,
        kalori_target=None,
        kategori_lauk=None,
        sumber_karbo_list=None,
        deskripsi_preferensi=None,
        user_preferences=None,
        weights=None,
        top_n=10,
        mode="seimbang"   # <<===== NEW PARAMETER
    ):

        # ============================
        # PARSE INPUT
        # ============================
        if user_preferences:
            kalori_target = user_preferences.get("kalori", kalori_target)
            kategori_lauk = user_preferences.get("lauk", kategori_lauk)
            karbo = user_preferences.get("karbo", "")
            deskripsi_preferensi = user_preferences.get("deskripsi", deskripsi_preferensi)
        else:
            karbo = sumber_karbo_list[0] if sumber_karbo_list else ""

        # ============================
        # MODE WEIGHT SELECTOR
        # ============================
        if weights is None:  
            mode = str(mode).lower().strip()
            if mode not in self.modes:
                print(f"⚠️ Mode '{mode}' tidak ditemukan, menggunakan mode seimbang.")
                mode = 'seimbang'

            weights = self.modes[mode]

        # ============================
        # TF-IDF VECTOR USER
        # ============================
        user_desc_vec = None
        if deskripsi_preferensi and str(deskripsi_preferensi) != 'nan':
            try:
                user_desc_vec = self.vectorizer.transform([str(deskripsi_preferensi).lower()])
            except:
                user_desc_vec = None

        # ============================
        # START SCORING
        # ============================
        scores = []

        for idx, row in self.df.iterrows():

            s_kalori = self._calculate_calorie_score(row['Kalori'], kalori_target)
            s_lauk = self._calculate_category_score(row['Kategori_Lauk'], kategori_lauk)
            s_karbo = self._calculate_category_score(row['Sumber_Karbohidrat'], karbo)

            s_deskripsi = 0.0
            if user_desc_vec is not None:
                try:
                    s_deskripsi = cosine_similarity(user_desc_vec, self.tfidf_matrix[idx])[0][0]
                except:
                    s_deskripsi = 0.0

            keyword_boost = self._calculate_keyword_boost(row['Deskripsi_Menu'], deskripsi_preferensi)

            final_score = (
                (s_kalori * weights['w_kalori']) +
                (s_lauk * weights['w_lauk']) +
                (s_karbo * weights['w_karbo']) +
                (s_deskripsi * weights['w_deskripsi']) +
                keyword_boost
            )

            scores.append({
                "Menu_ID": row.get("Menu_ID", idx),
                "Nama_Menu": row.get("Nama_Menu", "Unknown"),
                "Kalori": row.get("Kalori", 0),
                "Kategori_Lauk": row.get("Kategori_Lauk", ""),
                "Final_Score": final_score,
                "Score_Kalori": s_kalori,
                "Score_Lauk": s_lauk,
                "Score_Karbo": s_karbo,
                "Score_Deskripsi": s_deskripsi,
                "Keyword_Boost": keyword_boost
            })

        results_df = pd.DataFrame(scores)
        if not results_df.empty:
            results_df = results_df.sort_values(by="Final_Score", ascending=False).head(top_n)

        return results_df
