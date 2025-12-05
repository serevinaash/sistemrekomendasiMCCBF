import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from mccbf_engine import MCCBFEngine

MENU_DATASET = "dataset450_clean_mccbf.csv"
TRAIN_CSV = "training_interactions_full.csv"
MODEL_OUT = "random_forest_model.pkl"


def build_feature_row(engine, user_row):
    menu_name = user_row["menu_name"]
    menu_id = engine.find_menu_id(menu_name)

    if menu_id is None:
        return None

    feature = engine.compute_feature_vector(
        query_text=user_row["deskripsi_preferensi"],
        target_calories=user_row["kalori_target"],
        kategori_lauk=user_row["kategori_lauk"],
        allowed_karbo=user_row["sumber_karbo"].split(";"),
        menu_id=menu_id
    )

    return feature


def main():
    print("🔧 Load dataset & engine...")
    menu_df = pd.read_csv(MENU_DATASET)
    engine = MCCBFEngine(menu_df)

    print("🔧 Load training interactions...")
    df = pd.read_csv(TRAIN_CSV)

    X, y = [], []

    for _, row in df.iterrows():
        fv = build_feature_row(engine, row)
        if fv is not None:
            X.append(fv)
            y.append(row["relevance"])

    print(f"📊 Total training samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔧 Hyperparameter search RandomForest...")
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = best_model.score(X_test, y_test)
    prec = sum((p == 1) & (y_test[i] == 1) for i, p in enumerate(y_pred)) / sum(p == 1 for p in y_pred)
    rec = sum((p == 1) & (y_test[i] == 1) for i, p in enumerate(y_pred)) / sum(y_test)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"✅ Accuracy  = {acc:.4f}")
    print(f"📊 Precision = {prec:.4f}")
    print(f"📊 Recall    = {rec:.4f}")
    print(f"📊 F1-score  = {f1:.4f}")

    print("💾 Save model...")
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(best_model, f)

    print(f"🎉 RandomForest saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
