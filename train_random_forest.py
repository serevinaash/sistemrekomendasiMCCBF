import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mccbf_engine import MCCBFEngine


MENU_DATASET = "dataset450_clean_mccbf.csv"
TRAIN_CSV = "training_interactions_full.csv"
MODEL_OUT = "random_forest_model.pkl"


# ============================
# Build single feature vector
# ============================
def build_feature_row(engine, user_row):
    menu_name = user_row["menu_name"]
    menu_id = engine.find_menu_id(menu_name)

    if menu_id is None:
        return None

    fv = engine.compute_feature_vector(
        query_text=user_row["deskripsi_preferensi"],
        target_calories=user_row["kalori_target"],
        kategori_lauk=user_row["kategori_lauk"],
        allowed_karbo=user_row["sumber_karbo"].split(";"),
        menu_id=menu_id
    )
    return fv


# ============================
# MAIN PIPELINE
# ============================
def main():
    print("🔧 Load menu & engine...")
    menu_df = pd.read_csv(MENU_DATASET)
    engine = MCCBFEngine(menu_df)

    print("🔧 Load training interactions...")
    df = pd.read_csv(TRAIN_CSV)

    X, y, groups = [], [], []

    # Build data
    for _, row in df.iterrows():
        fv = build_feature_row(engine, row)
        if fv is not None:
            X.append(fv)
            y.append(row["relevance"])
            groups.append(row["user_id"])  # Group by user!

    print(f"📊 Total usable samples: {len(X)}")

    # ====================================================
    # 🚫 NO DATA LEAKAGE — Grouped Train/Test by USER
    # ====================================================
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test  = [y[i] for i in test_idx]

    print(f"👥 Train users: {len(set([groups[i] for i in train_idx]))}")
    print(f"👥 Test users : {len(set([groups[i] for i in test_idx]))}\n")

    # ====================================================
    # 💪 ANTI-OVERFITTING RandomForest
    # ====================================================
    print("🔧 Hyperparameter Search (Anti-Overfitting RF)...")

    param_grid = {
        "n_estimators": [200],
        "max_depth": [6, 10],
        "min_samples_split": [10, 15],
        "min_samples_leaf": [4, 6],
        "max_features": ["sqrt"],
        "bootstrap": [True],
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ====================================================
    # 📊 TRUE TEST PERFORMANCE (Realistic)
    # ====================================================
    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print("\n🎯 FINAL TEST PERFORMANCE (NO LEAK, REALISTIC):")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1 Score  = {f1:.4f}\n")

    # Save the model
    print("💾 Saving model...")
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(best_model, f)

    print(f"🎉 RandomForest saved to {MODEL_OUT}")


# ============================
# RUN
# ============================
if __name__ == "__main__":
    main()
