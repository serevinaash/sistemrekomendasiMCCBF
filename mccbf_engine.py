# mccbf_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier


@dataclass
class UserProfile:
    """
    Representasi preferensi user untuk MCCBF + RF.
    """
    query_text: str                      # deskripsi kebutuhan user (mis: "menu ayam 400 kalori pedas")
    target_calories: Optional[int] = None
    prefer_kategori: Optional[str] = None
    allowed_karbo: Optional[List[str]] = None
    banned_karbo: Optional[List[str]] = None


class MCCBFEngine:
    """
    MCCBF LAMA + hook RandomForest sebagai booster ranking.
    """

    def __init__(
        self,
        menu_df: pd.DataFrame,
        id_col: str = "No",
        corpus_col: str = "corpus",
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.menu_df = menu_df.copy().reset_index(drop=True)
        self.id_col = id_col

        if self.id_col not in self.menu_df.columns:
            # fallback: buat ID sendiri
            self.menu_df[self.id_col] = np.arange(1, len(self.menu_df) + 1)

        if corpus_col not in self.menu_df.columns:
            raise ValueError(f"Tidak ditemukan kolom corpus: {corpus_col}")

        self.corpus_col = corpus_col

        # pastikan Karbo_List adalah list
        if "Karbo_List" in self.menu_df.columns:
            self.menu_df["Karbo_List"] = self.menu_df["Karbo_List"].apply(
                lambda v: v if isinstance(v, list) else []
            )
        else:
            self.menu_df["Karbo_List"] = [[] for _ in range(len(self.menu_df))]

        # TF-IDF
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.menu_df[self.corpus_col].astype(str)
        )

    # ---------- UTILITAS SKOR DASAR ----------

    def _content_similarity(self, query_text: str) -> np.ndarray:
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        return sims

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if np.allclose(x, x[0]):
            return np.ones_like(x)
        min_x = x.min()
        max_x = x.max()
        if max_x == min_x:
            return np.ones_like(x)
        return (x - min_x) / (max_x - min_x + 1e-9)

    def _build_base_scores(
        self,
        profile: UserProfile,
        w_sim: float = 0.5,
        w_cal: float = 0.3,
        w_match: float = 0.2,
    ) -> pd.DataFrame:
        """
        Hitung skor MCCBF (tanpa RandomForest).
        Return DataFrame temp dengan kolom skor-skor.
        """
        df = self.menu_df.copy()

        # 1. similarity konten
        sims = self._content_similarity(profile.query_text)
        df["sim_content"] = sims
        sim_norm = self._normalize(sims)

        # 2. skor kalori (semakin dekat ke target semakin tinggi)
        if profile.target_calories is not None:
            cal_diff = (df["Kalori"] - profile.target_calories).abs()
            cal_diff_norm = cal_diff / (cal_diff.max() + 1e-9)
            df["cal_diff"] = cal_diff
            df["cal_score"] = 1.0 - cal_diff_norm
        else:
            df["cal_diff"] = np.nan
            df["cal_score"] = 1.0

        # 3. kategori match
        if profile.prefer_kategori:
            pref = profile.prefer_kategori.strip().lower()
            df["kategori_match"] = (
                df["Kategori"].astype(str).str.lower() == pref
            ).astype(int)
        else:
            df["kategori_match"] = 0

        # 4. karbo match
        allowed = (
            [k.lower() for k in profile.allowed_karbo]
            if profile.allowed_karbo
            else []
        )
        banned = (
            [k.lower() for k in profile.banned_karbo]
            if profile.banned_karbo
            else []
        )

        karbo_overlap = []
        karbo_bad_overlap = []

        for karbo_list in df["Karbo_List"]:
            karbos = [k.lower() for k in karbo_list]
            if allowed:
                overlap = len(set(karbos) & set(allowed))
            else:
                overlap = 0

            if banned:
                bad = len(set(karbos) & set(banned))
            else:
                bad = 0

            karbo_overlap.append(overlap)
            karbo_bad_overlap.append(bad)

        df["karbo_overlap"] = karbo_overlap
        df["karbo_bad_overlap"] = karbo_bad_overlap

        if df["karbo_overlap"].max() > 0:
            df["karbo_score"] = df["karbo_overlap"] / df["karbo_overlap"].max()
        else:
            df["karbo_score"] = 0.0

        if df["karbo_bad_overlap"].max() > 0:
            penalty = df["karbo_bad_overlap"] / df["karbo_bad_overlap"].max()
        else:
            penalty = 0.0
        df["karbo_penalty"] = penalty
        df["karbo_final"] = df["karbo_score"] * (1.0 - df["karbo_penalty"])

        # 5. gabungkan
        match_score = (df["kategori_match"] + df["karbo_final"]) / 2.0

        df["mccbf_score"] = (
            w_sim * sim_norm
            + w_cal * df["cal_score"]
            + w_match * match_score
        )

        return df

    # ---------- FITUR UNTUK RANDOMFOREST ----------

    @staticmethod
    def build_feature_matrix_from_df(
        df_scored: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Bangun fitur numerik untuk RandomForest dari DataFrame yang sudah berisi skor-skor MCCBF.
        Fitur bisa kamu tambahkan sendiri kalau mau lebih kaya.
        """
        features = pd.DataFrame(
            {
                "sim_content": df_scored["sim_content"].astype(float),
                "cal_diff": df_scored["cal_diff"].fillna(
                    df_scored["cal_diff"].max()
                ),
                "cal_score": df_scored["cal_score"].astype(float),
                "kategori_match": df_scored["kategori_match"].astype(int),
                "karbo_overlap": df_scored["karbo_overlap"].astype(int),
                "karbo_bad_overlap": df_scored["karbo_bad_overlap"].astype(int),
                "karbo_final": df_scored["karbo_final"].astype(float),
                "mccbf_score": df_scored["mccbf_score"].astype(float),
                "kalori": df_scored["Kalori"].astype(int),
            }
        )
        return features

    # ---------- API RECOMMEND ----------

    def recommend(
        self,
        profile: UserProfile,
        top_k: int = 10,
        w_sim: float = 0.5,
        w_cal: float = 0.3,
        w_match: float = 0.2,
        rf_model: Optional[RandomForestClassifier] = None,
        alpha_mccbf_vs_rf: float = 0.5,
    ) -> pd.DataFrame:
        """
        Kembalikan top_k rekomendasi:
        - Kalau rf_model=None -> murni MCCBF
        - Kalau rf_model ada -> kombinasi MCCBF + prediksi RF
        """
        scored = self._build_base_scores(
            profile, w_sim=w_sim, w_cal=w_cal, w_match=w_match
        )

        if rf_model is not None:
            feats = self.build_feature_matrix_from_df(scored)
            proba = rf_model.predict_proba(feats)[:, 1]  # probabilitas relevan
            scored["rf_score"] = proba
            # gabung dua skor
            scored["final_score"] = (
                alpha_mccbf_vs_rf * scored["mccbf_score"]
                + (1.0 - alpha_mccbf_vs_rf) * scored["rf_score"]
            )
        else:
            scored["rf_score"] = np.nan
            scored["final_score"] = scored["mccbf_score"]

        scored = scored.sort_values("final_score", ascending=False).reset_index(
            drop=True
        )
        return scored.head(top_k)
