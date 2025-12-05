# evaluate_system.py

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score

from mccbf_engine import MCCBFEngine, UserProfile


MENU_CSV_PATH = "menu_450_clean.csv"
TEST_INTERACTIONS_PATH = "test_interactions.csv"
MODEL_PATH = "rf_mccbf_model.joblib"


def parse_allowed_karbo(s: str):
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip().lower() for p in s.split(";") if p.strip()]


def main(top_k: int = 10):
    menu_df = pd.read_csv(MENU_CSV_PATH)
    test_df = pd.read_csv(TEST_INTERACTIONS_PATH)
    engine = MCCBFEngine(menu_df)

    try:
        rf_model = load(MODEL_PATH)
        print("✅ RF loaded for evaluation.")
    except FileNotFoundError:
        rf_model = None
        print("⚠️ RF model not found, evaluate MCCBF only.")

    # grup per user (boleh juga per konteks lain)
    grouped = test_df.groupby("user_id")

    y_true_all = []
    y_pred_all = []

    for user_id, group in grouped:
        # ambil satu profile user dari grup
        first = group.iloc[0]
        profile = UserProfile(
            query_text=str(first["query_text"]),
            target_calories=int(first["target_calories"])
            if not pd.isna(first["target_calories"])
            else None,
            prefer_kategori=str(first["prefer_kategori"])
            if isinstance(first["prefer_kategori"], str)
            else None,
            allowed_karbo=parse_allowed_karbo(first.get("allowed_karbo", "")),
            banned_karbo=None,
        )

        recs = engine.recommend(
            profile,
            top_k=top_k,
            rf_model=rf_model,
        )

        recommended_ids = set(recs[engine.id_col].tolist())

        for _, row in group.iterrows():
            menu_id = row["menu_id"]
            label = int(row["label"])
            y_true_all.append(label)
            y_pred_all.append(1 if menu_id in recommended_ids else 0)

    # hitung metrics
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    print(f"Top-{top_k} Precision: {precision:.4f}")
    print(f"Top-{top_k} Recall   : {recall:.4f}")
    print(f"Top-{top_k} F1       : {f1:.4f}")


if __name__ == "__main__":
    main(top_k=10)
