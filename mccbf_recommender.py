# mccbf_recommender.py

from __future__ import annotations

import pandas as pd
from joblib import load

from mccbf_engine import MCCBFEngine, UserProfile


MENU_CSV_PATH = "menu_450_clean.csv"
MODEL_PATH = "rf_mccbf_model.joblib"  # boleh None kalau mau murni MCCBF


def main():
    # 1. load menu
    menu_df = pd.read_csv(MENU_CSV_PATH)

    # 2. init engine
    engine = MCCBFEngine(menu_df)

    # 3. load RF (opsional)
    try:
        rf_model = load(MODEL_PATH)
        print("✅ RandomForest model loaded.")
    except FileNotFoundError:
        rf_model = None
        print("⚠️ RF model tidak ditemukan. Pakai MCCBF saja.")

    # 4. contoh profile user
    profile = UserProfile(
        query_text="menu ayam pedas sehat untuk diet",
        target_calories=400,
        prefer_kategori="ayam",
        allowed_karbo=["nasi merah", "kentang"],
        banned_karbo=["mie", "pasta"],
    )

    # 5. rekomendasi
    result = engine.recommend(
        profile,
        top_k=10,
        w_sim=0.5,
        w_cal=0.3,
        w_match=0.2,
        rf_model=rf_model,
        alpha_mccbf_vs_rf=0.5,
    )

    # 6. tampilkan kolom utama
    cols_show = [
        "No",
        "Nama_Menu",
        "Kategori",
        "Kalori",
        "Sumber_Karbohidrat",
        "mccbf_score",
        "rf_score",
        "final_score",
    ]
    cols_show = [c for c in cols_show if c in result.columns]

    print(result[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
