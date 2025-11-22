# utils/mccbf_engine_we.py

import os
import re
import numpy as np
import pandas as pd
import fasttext  # pastikan sudah install: pip install fasttext
from sklearn.metrics.pairwise import cosine_similarity


class MCCBFEngineEmbedding:
    """
    Engine MCCBF dengan similarity deskripsi berbasis
    FastText word embedding + cosine similarity.
    """

    def __init__(self, embedding_path, data_train_path):
        """
        Args:
            embedding_path: path ke model FastText .bin (Bahasa Indonesia)
            data_train_path: path ke data_train.csv
        """
        # Load dataset menu
        self.df_train = pd.read_csv(data_train_path)
        self.df_train.columns = (
            self.df_train.columns.str.strip().str.replace(' ', '_')
        )

        # Load model FastText
        print("📥 Loading FastText model...")
        self.ft = fasttext.load_model(embedding_path)
        self.embedding_dim = self.ft.get_dimension()
        print(f"✅ FastText loaded (dim={self.embedding_dim})")

        # Buat corpus text menu
        if 'corpus' not in self.df_train.columns:
            self.df_train['corpus'] = self._create_corpus(self.df_train)

        # Precompute embedding untuk semua menu
        print("⚙️  Precomputing menu embeddings...")
        self.menu_embeddings = self._compute_menu_embeddings(self.df_train['corpus'].tolist())
        print("✅ MCCBFEngineEmbedding siap dipakai")
        print(f"   📊 Jumlah menu: {len(self.df_train)}")

    # =========================================================
    # UTIL
    # =========================================================
    def _create_corpus(self, df):
        text_cols = [
            'Nama_Menu',
            'Kategori',
            'Sumber_Karbohidrat',
            'Bahan_Utama_/_Pendamping',
            'Deskripsi_Singkat'
        ]
        corpus = []
        for _, row in df.iterrows():
            parts = [str(row[col]) for col in text_cols if col in df.columns]
            corpus.append(" ".join(parts))
        return corpus

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def text_to_embedding(self, text):
        """
        Ubah teks menjadi vektor embedding dengan cara
        average semua vektor kata yang dikenali FastText.
        """
        text = self.clean_text(text)
        tokens = text.split()
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        vecs = []
        for t in tokens:
            vecs.append(self.ft.get_word_vector(t))
        if not vecs:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        return np.mean(vecs, axis=0)

    def _compute_menu_embeddings(self, corpus_list):
        """
        Precompute embedding untuk seluruh menu di dataset.
        """
        embeddings = []
        for text in corpus_list:
            emb = self.text_to_embedding(text)
            embeddings.append(emb)
        return np.vstack(embeddings)  # shape: (n_menu, dim)

    # =========================================================
    # SIMILARITY KATEGORI & KARBO & KALORI
    # =========================================================
    def categorical_similarity(self, user_value, menu_value):
        user_value = str(user_value).lower()
        menu_value = str(menu_value).lower()
        if user_value == menu_value:
            return 1.0
        if user_value in menu_value:
            return 0.5
        return 0.0

    def karbo_partial_match(self, user_list, menu_str):
        if not user_list:
            return 1.0
        menu_str = str(menu_str).lower()
        menu_tokens = set(menu_str.replace(',', ' ').split())
        user_tokens = set([k.lower().strip() for k in user_list])

        # Exact match
        exact = user_tokens.intersection(menu_tokens)
        if exact:
            return min(len(exact) / len(user_tokens), 1.0)

        # Partial match
        partial_score = 0
        for pref in user_tokens:
            if pref in menu_str:
                partial_score += 0.7
            elif any(w in menu_str for w in pref.split()):
                partial_score += 0.3

        if partial_score:
            return min(partial_score / len(user_tokens), 1.0)

        return 0.0

    # =========================================================
    # FUNGSI UTAMA: GET RECOMMENDATIONS
    # =========================================================
    def get_recommendations(
        self,
        kalori_target,
        kategori_lauk,
        sumber_karbo_list,
        deskripsi_preferensi,
        weights=None,
        top_n=5,
        karbo_strict_mode=False
    ):
        """
        MCCBF dengan similarity deskripsi berbasis FastText embedding.
        """
        # Default weights (boleh kamu sesuaikan lagi)
        if weights is None:
            weights = {
                'deskripsi': 0.45,
                'kategori': 0.25,
                'karbohidrat': 0.20,
                'kalori': 0.10
            }

        # 1️⃣ DESKRIPSI → Embedding + Cosine
        user_emb = self.text_to_embedding(deskripsi_preferensi)
        user_emb_2d = user_emb.reshape(1, -1)
        sim_deskripsi = cosine_similarity(user_emb_2d, self.menu_embeddings)[0]  # shape: (n_menu,)

        # 2️⃣ KALORI → normalized distance
        kal_min = self.df_train['Kalori_(kcal)'].min()
        kal_max = self.df_train['Kalori_(kcal)'].max()
        if kal_max > kal_min:
            kal_norm = (kalori_target - kal_min) / (kal_max - kal_min)
            menu_norm = (self.df_train['Kalori_(kcal)'] - kal_min) / (kal_max - kal_min)
        else:
            kal_norm = 1.0
            menu_norm = np.ones(len(self.df_train))

        sim_kalori = 1 - np.abs(kal_norm - menu_norm)
        sim_kalori = np.clip(sim_kalori, 0, 1)

        # 3️⃣ KATEGORI
        sim_kategori = np.array([
            self.categorical_similarity(kategori_lauk, row['Kategori'])
            for _, row in self.df_train.iterrows()
        ])

        # 4️⃣ KARBO
        sim_karbo = np.array([
            self.karbo_partial_match(sumber_karbo_list, row['Sumber_Karbohidrat'])
            for _, row in self.df_train.iterrows()
        ])

        # 5️⃣ GABUNGKAN (WEIGHTED AVERAGE)
        sim_deskripsi = np.clip(sim_deskripsi, 0, 1)
        sim_total = (
            weights['deskripsi'] * sim_deskripsi +
            weights['kategori'] * sim_kategori +
            weights['karbohidrat'] * sim_karbo +
            weights['kalori'] * sim_kalori
        )
        sim_total = np.clip(sim_total, 0, 1)

        # 6️⃣ STRICT KARBO (opsional)
        if karbo_strict_mode and sumber_karbo_list:
            valid_idx = np.where(sim_karbo >= 0.7)[0]
            if len(valid_idx) == 0:
                valid_idx = np.arange(len(sim_total))
            sim_valid = sim_total[valid_idx]
            top_idx = valid_idx[np.argsort(sim_valid)[::-1][:top_n]]
        else:
            top_idx = np.argsort(sim_total)[::-1][:top_n]

        # 7️⃣ FORMAT HASIL
        cols = [
            'Nama_Menu', 'Kategori', 'Kalori_(kcal)',
            'Sumber_Karbohidrat', 'Deskripsi_Singkat'
        ]
        available_cols = [c for c in cols if c in self.df_train.columns]
        result = self.df_train.iloc[top_idx][available_cols].copy()
        result['Skor_Similarity'] = sim_total[top_idx]
        result['Rank'] = range(1, len(top_idx) + 1)
        result = result[['Rank', 'Skor_Similarity'] + available_cols]

        return result


def load_mccbf_engine_embedding(
    model_dir='model',
    embedding_filename='cc.id.300.bin'
):
    """
    Loader helper untuk Streamlit.
    Asumsi:
        - data_train.csv ada di model_dir
        - model FastText (.bin) juga ada di model_dir
    """
    data_train_path = os.path.join(model_dir, 'data_train.csv')
    embedding_path = os.path.join(model_dir, embedding_filename)
    engine = MCCBFEngineEmbedding(embedding_path, data_train_path)
    return engine
