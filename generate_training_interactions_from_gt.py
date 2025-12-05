import pandas as pd

MENU_DATASET = "dataset450_clean_mccbf.csv"
GT_FILE = "ground_truth_v4.csv"        # file GT kamu
OUT_FILE = "training_interactions_full.csv"


def main():
    print("🔧 Load dataset menu...")
    menu_df = pd.read_csv(MENU_DATASET)

    print("🔧 Load ground truth...")
    gt = pd.read_csv(GT_FILE)

    rows = []

    for _, row in gt.iterrows():
        user_id = row["user_id"]

        kalori_target = row["kalori_target"]
        kategori_lauk = row["kategori_lauk"]
        sumber_karbo = str(row["sumber_karbo"]).split(";")
        deskripsi_pref = str(row["deskripsi_preferensi"])

        relevant = [m.strip().lower() for m in str(row["relevant_menus"]).split(",")]

        # ========== POSITIF SAMPLE ==========
        for menu in relevant:
            rows.append({
                "user_id": user_id,
                "menu_name": menu,
                "relevance": 1,
                "kalori_target": kalori_target,
                "kategori_lauk": kategori_lauk,
                "sumber_karbo": ";".join(sumber_karbo),
                "deskripsi_preferensi": deskripsi_pref
            })

        # ========== NEGATIF SAMPLE ==========
        all_menus = menu_df["Nama_Menu"].astype(str).str.lower().tolist()

        # ambil 20 random negatif
        neg_candidates = [m for m in all_menus if m not in relevant]
        neg_pick = pd.Series(neg_candidates).sample(20, random_state=user_id).tolist()

        for menu in neg_pick:
            rows.append({
                "user_id": user_id,
                "menu_name": menu,
                "relevance": 0,
                "kalori_target": kalori_target,
                "kategori_lauk": kategori_lauk,
                "sumber_karbo": ";".join(sumber_karbo),
                "deskripsi_preferensi": deskripsi_pref
            })

        print(f"User {user_id} → {len(relevant)} positif, 20 negatif")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_FILE, index=False)

    print(f"\n🎉 Selesai! Disimpan ke {OUT_FILE}")
    print(f"📊 Total sampel: {len(df_out)}")


if __name__ == "__main__":
    main()
