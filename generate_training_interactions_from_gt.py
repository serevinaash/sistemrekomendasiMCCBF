import pandas as pd
import random

GROUND_TRUTH_CSV = "data/ground_truth_v4.csv"
MENU_DATASET_CSV = "dataset450_clean_mccbf.csv"
OUTPUT_CSV = "training_interactions.csv"

NEGATIVE_SAMPLES_PER_USER = 20  # bisa disesuaikan

def main():
    print("🔧 Load ground truth...")
    gt = pd.read_csv(GROUND_TRUTH_CSV)
    menu_df = pd.read_csv(MENU_DATASET_CSV)

    all_menus = menu_df["Nama_Menu"].str.lower().unique().tolist()

    records = []

    for idx, row in gt.iterrows():
        user = row["user_id"]
        relevant_list = [
            m.strip().lower() for m in str(row["relevant_menus"]).split(",") if m.strip()
        ]
        relevant_list = list(set(relevant_list))  # remove duplicates

        # POSITIVE samples
        for menu in relevant_list:
            records.append({
                "user_id": user,
                "menu_name": menu,
                "relevance": 1
            })

        # NEGATIVE samples
        negative_candidates = [m for m in all_menus if m not in relevant_list]

        negative_selected = random.sample(
            negative_candidates,
            min(NEGATIVE_SAMPLES_PER_USER, len(negative_candidates))
        )

        for menu in negative_selected:
            records.append({
                "user_id": user,
                "menu_name": menu,
                "relevance": 0
            })

        print(f"User {user} → {len(relevant_list)} positif, {len(negative_selected)} negatif")

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Selesai! Disimpan ke {OUTPUT_CSV}")
    print(f"📊 Total sampel: {len(df_out)}")


if __name__ == "__main__":
    main()
