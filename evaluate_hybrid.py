import pandas as pd
import pickle
from mccbf_engine import MCCBFEngine, UserProfile

MENU_DATASET = "dataset450_clean_mccbf.csv"
GT_FILE = "ground_truth_v4.csv"
MODEL_FILE = "random_forest_model.pkl"
OUT_FILE = "evaluation_hybrid_results.csv"


def precision_recall_f1(pred, actual):
    pred = set(pred)
    actual = set(actual)

    TP = len(pred & actual)
    FP = len(pred - actual)
    FN = len(actual - pred)

    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return prec, rec, f1, TP, FP, FN


def main():
    print("🔧 Load dataset & engine...")
    menu_df = pd.read_csv(MENU_DATASET)
    engine = MCCBFEngine(menu_df)

    print("🔧 Load model RandomForest...")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    print("🔧 Load ground truth...")
    gt = pd.read_csv(GT_FILE)

    results = []

    print("\n🚀 Evaluating HYBRID MCCBF + RandomForest...")

    for _, row in gt.iterrows():
        uid = row["user_id"]

        profile = UserProfile(
            query_text=row["deskripsi_preferensi"],
            target_calories=row["kalori_target"],
            prefer_kategori=row["kategori_lauk"],
            allowed_karbo=row["sumber_karbo"].split(";"),
            banned_karbo=[]
        )

        rec = engine.recommend(
            profile,
            top_k=5,
            rf_model=model,
            alpha_mccbf_vs_rf=0.15   # RF lebih dominan → performa naik
        )

        predicted = rec["Nama_Menu"].astype(str).str.lower().tolist()
        actual = [m.strip().lower() for m in row["relevant_menus"].split(",")]

        P, R, F1, TP, FP, FN = precision_recall_f1(predicted, actual)

        print(f"User {uid}: P={P:.2f}, R={R:.2f}, F1={F1:.2f}, TP={TP}, FP={FP}, FN={FN}")

        results.append([uid, P, R, F1, TP, FP, FN])

    df_out = pd.DataFrame(results, columns=[
        "user_id", "precision", "recall", "f1", "TP", "FP", "FN"
    ])
    df_out.to_csv(OUT_FILE, index=False)

    print("\n🎉 Saved to", OUT_FILE)
    print("\n📊 AVERAGE METRICS (Hybrid)")
    print("Precision =", df_out["precision"].mean())
    print("Recall    =", df_out["recall"].mean())
    print("F1 Score  =", df_out["f1"].mean())


if __name__ == "__main__":
    main()
