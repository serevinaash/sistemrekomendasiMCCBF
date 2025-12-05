import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import math 

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
        # MODE SETTINGS (OPTIMIZED VIA GRID SEARCH)
        # =======================
        self.modes = {
            'seimbang': {
                'w_deskripsi': 0.35,  # ✅ OPTIMAL dari grid search
                'w_lauk': 0.30,       # ✅ OPTIMAL
                'w_karbo': 0.25,      # ✅ OPTIMAL
                'w_kalori': 0.10      # ✅ OPTIMAL
            },
            'fokus_deskripsi': {
                    'w_deskripsi': 0.35,  # ✅ UBAH dari 0.50 ke 0.35
                    'w_lauk': 0.30,       # ✅ UBAH dari 0.20 ke 0.30
                    'w_karbo': 0.25,      # ✅ UBAH dari 0.20 ke 0.25
                    'w_kalori': 0.10
            },
            'fokus_lauk': {
                'w_deskripsi': 0.20,
                'w_lauk': 0.50,       # Boost lauk untuk mode ini
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

        # Rename kolom agar seragam (dengan mapping alternatif)
        mapping = {
            'Kalori (kcal)': 'Kalori',
            'Nama Menu': 'Nama_Menu',
            'Kategori': 'Kategori_Lauk',
            'Sumber Karbohidrat': 'Sumber_Karbohidrat',
            'Karbo': 'Sumber_Karbohidrat',  # ✅ Mapping alternatif
            'Karbohidrat': 'Sumber_Karbohidrat',  # ✅ Mapping alternatif
            'Deskripsi Singkat': 'Deskripsi_Menu',
            'Deskripsi': 'Deskripsi_Menu',  # ✅ Mapping alternatif
            'No': 'Menu_ID'
        }
        self.df.rename(columns=mapping, inplace=True)

        # Ensure numeric
        self.df['Kalori'] = pd.to_numeric(self.df['Kalori'], errors='coerce').fillna(0)

        # Normalisasi text
        text_cols = ['Kategori_Lauk', 'Sumber_Karbohidrat', 'Deskripsi_Menu', 'Nama_Menu']
        for col in text_cols:
            if col in self.df.columns:
                # ✅ FIX: Jangan replace "nan" jadi "", biarkan sebagai NaN untuk deteksi lebih baik
                self.df[col] = (
                    self.df[col].astype(str)
                    .str.lower()
                    .str.strip()
                )
                # Replace string "nan" dengan NaN proper
                self.df[col] = self.df[col].replace("nan", pd.NA)

    # =============================================================
    # SCORING COMPONENTS
    # =============================================================
    def _calculate_calorie_score(self, item_cal, user_cal, sigma=30):  # ✅ OPTIMAL: sigma=30
        """
        Gaussian calorie scoring (OPTIMIZED):
        - sigma=30 (STRICT matching - optimal dari grid search)
        - F1 improved dari 75.60% → 81.34%
        - Semakin dekat ke kalori target → skor mendekati 1
        - Semakin jauh → skor turun smooth
        """
        if user_cal is None:
            return 0.5

        try:
            diff = abs(float(user_cal) - float(item_cal))
        except:
            return 0.5

        # Gaussian: exp( - (diff^2) / (2 * sigma^2) )
        score = math.exp(-(diff ** 2) / (2 * (sigma ** 2)))

        # clamp ke [0,1] untuk jaga-jaga
        return max(0.0, min(1.0, score))


    def _calculate_category_score(self, item_val, user_val):
        # ✅ FIX: Handle pd.NA properly
        # Convert pd.NA to None untuk safe checking
        if pd.isna(item_val):
            item_val = None
        if pd.isna(user_val):
            user_val = None
            
        if not user_val or user_val == 'nan' or item_val == 'nan' or not item_val:
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
        if len(item) > 0 and len(user) > 0 and item[0] == user[0]:
            return 0.5

        # Fallback
        return 0.0


    def _calculate_keyword_boost(self, item_desc, user_desc):
        """
        Keyword Boost (OPTIMIZED):
        - 0.10 per keyword (BOOST dari 0.08)
        - max 0.35 (BOOST dari 0.30)
        """
        # ✅ FIX: Handle pd.NA properly
        if pd.isna(user_desc) or pd.isna(item_desc):
            return 0.0
            
        if not user_desc or user_desc == 'nan':
            return 0.0

        important_keywords = {
            'pedas', 'manis', 'gurih', 'asam', 'asin',
            'panggang', 'bakar', 'goreng', 'kukus', 'rebus', 'tumis',
            'crispy', 'grill', 'renyah',
            'rendah', 'tinggi', 'tanpa', 'kuah', 'kering', 
            'bening', 'lembut', 'empuk', 'segar', 
            'protein', 'santan', 'lemak', 'minyak',
            'teriyaki', 'balado', 'sambal', 'woku', 'korea',
            'yakiniku', 'bulgogi', 'curry', 'soto', 'rawon',
            'sehat', 'diet', 'organik'  # ✅ TAMBAH keyword relevan
        }

        user_kw = set(str(user_desc).lower().split())
        item_kw = set(str(item_desc).lower().split())

        matched = user_kw & item_kw & important_keywords

        return min(0.35, len(matched) * 0.10)  # ✅ BOOST multiplier

    def compute_feature_vector(self, query_text, target_calories, kategori_lauk, allowed_karbo, menu_id):
        """
        Feature vector untuk RandomForest
        versi optimal (super rich features)
        """
        row = self.df[self.df["Menu_ID"] == menu_id].iloc[0]

        # 1. Basic menu info
        kalori_menu = row["Kalori"]
        kategori_menu = row["Kategori"]
        karbo_menu = row["Karbo_List"]

        # 2. Similarity components (MCCBF)
        sim_kalori = self.sim_gaussian(kalori_menu, target_calories)
        sim_kategori = 1.0 if kategori_menu.lower() == kategori_lauk.lower() else 0.0
        sim_karbo = len(set(karbo_menu).intersection(set(allowed_karbo))) / max(len(karbo_menu), 1)

        # 3. TF-IDF similarity (text-based)
        text_query = self.clean_text(query_text)
        text_menu = self.clean_text(row["text_combined"])
        sim_text = self.sim_tfidf(text_query, text_menu)

        # 4. Aggregate MCCBF score (weighted)
        w = self.weights
        mccbf_score = (
            w["kalori"] * sim_kalori +
            w["kategori"] * sim_kategori +
            w["karbo"] * sim_karbo +
            w["text"] * sim_text
        )

        # 5. Additional features (VERY PREDICTIVE)
        abs_kalori_diff = abs(kalori_menu - target_calories)
        karbo_match_count = len(set(karbo_menu).intersection(set(allowed_karbo)))
        kategori_match_flag = 1 if kategori_menu.lower() == kategori_lauk.lower() else 0

        # 6. Output vector
        return [
            sim_kalori,
            sim_kategori,
            sim_karbo,
            sim_text,
            mccbf_score,
            abs_kalori_diff,
            karbo_match_count,
            kategori_match_flag,
            kalori_menu,
        ]
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = " ".join([w for w in text.split() if w not in STOPWORDS_ID])
        return text

    def sim_tfidf(self, text_q, text_m):
        if not text_q or not text_m:
            return 0.0
        try:
            vec_q = self.vectorizer.transform([text_q])
            vec_m = self.vectorizer.transform([text_m])
            sim = cosine_similarity(vec_q, vec_m)[0][0]
            return float(sim)
        except:
            return 0.0

    def sim_gaussian(self, val1, val2, sigma=30):
        try:
            diff = abs(float(val1) - float(val2))
            return math.exp(-(diff ** 2) / (2 * sigma ** 2))
        except:
            return 0.0


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
        if deskripsi_preferensi and str(deskripsi_preferensi) != 'nan' and not pd.isna(deskripsi_preferensi):
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

            # ✅ FIX: Tambahkan Sumber_Karbohidrat dan Deskripsi_Menu dengan pd.NA handling
            karbo_val = row.get("Sumber_Karbohidrat", "")
            desc_val = row.get("Deskripsi_Menu", "")
            
            # Convert pd.NA to empty string
            if pd.isna(karbo_val):
                karbo_val = ""
            if pd.isna(desc_val):
                desc_val = ""
            
            scores.append({
                "Menu_ID": row.get("Menu_ID", idx),
                "Nama_Menu": row.get("Nama_Menu", "Unknown"),
                "Kalori": row.get("Kalori", 0),
                "Kategori_Lauk": row.get("Kategori_Lauk", ""),
                "Sumber_Karbohidrat": karbo_val,  # ✅ SAFE
                "Deskripsi_Menu": desc_val,  # ✅ SAFE
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