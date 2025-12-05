# train_random_forest.py

from __future__ import annotations

import ast
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from joblib import dump

from mccbf_engine import MCCBFEngine, UserProfile


MENU_CSV_PATH = "dataset450_clean_mccbf.csv"
INTERACTION_CSV_PATH = "training_interactions_fixed.csv"
MODEL_OUTPUT_PATH = "rf_mccbf_model.joblib"


def parse_allowed_karbo(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    # contoh format: "nasi merah;kentang;ubi"
    parts = [p.strip().lower() for p in s.split(";") if p.strip()]
    return parts


def build_training_data(engine, interactions):
    rows = []
    labels = []

    for _, row in interactions.iterrows():
        menu_id = int(row["menu_id"])

        # Ambil MENU dari MCCBF dataset
        try:
            menu_row = engine.df[engine.df["No"] == menu_id].iloc[0]
        except:
            continue  # skip jika tidak ditemukan (harusnya tidak terjadi)

        # ============================================
        # BUILD QUERY TEXT BARU (format updated GT)
        # ============================================

        kalori_target = row["kalori_target"]
        kategori_lauk = row["kategori_lauk"]
        sumber_karbo = row["sumber_karbo"]
        deskripsi_preferensi = row["deskripsi_preferensi"]

        # query_text lama diganti dengan kombinasi fitur user
        query_text = f"{kalori_target} {kategori_lauk} {sumber_karbo} {deskripsi_preferensi}"

        # ============================================
        # HITUNG FITUR MCCBF
        # ============================================

        scores = engine.compute_scores_for_menu(
            menu_row,
            kalori_target=kalori_target,
            kategori_lauk=kategori_lauk,
            sumber_karbo_list=str(sumber_karbo).split(),
            deskripsi_preferensi=deskripsi_preferensi
        )

        rows.append(scores)
        labels.append(int(row["label"]))

    X = pd.DataFrame(rows)
    y = pd.Series(labels)

    return X, y


def main():
    # 1. load data
    menu_df = pd.read_csv(MENU_CSV_PATH)
    interactions = pd.read_csv(INTERACTION_CSV_PATH)

    # 2. init engine
    engine = MCCBFEngine(menu_df)

    # 3. bangun fitur training
    X, y = build_training_data(engine, interactions)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. train RF
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    # 5. eval singkat
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC tidak bisa dihitung (label cuma 1 kelas).")

    # 6. simpan model
    dump(rf, MODEL_OUTPUT_PATH)
    print(f"✅ Model RandomForest disimpan ke: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
