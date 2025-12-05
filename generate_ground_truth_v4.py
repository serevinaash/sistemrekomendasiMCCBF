import pandas as pd
from difflib import get_close_matches

GT_OLD = "data/ground_truth_v4.csv"
MENU_DATASET = "dataset450_clean_mccbf.csv"
GT_OUT = "ground_truth_v4.csv"


def find_best_match(name, menu_list):
    if not isinstance(name, str):
        return None
    name_clean = name.lower().strip()
    match = get_close_matches(name_clean, menu_list, n=1, cutoff=0.55)
    return match[0] if match else None


def main():
    print("🔧 Load menu dataset...")
    df_menu = pd.read_csv(MENU_DATASET)
    df_menu["clean_name"] = df_menu["Nama_Menu"].astype(str).str.lower().str.strip()
    menu_list = df_menu["clean_name"].tolist()

    print("🔧 Load ground truth lama...")
    df_old = pd.read_csv(GT_OLD)

    results = []

    for _, row in df_old.iterrows():
        user_id = row["user_id"]

        # pecah daftar relevan lama
        relevant_raw = [
            m.strip().lower()
            for m in str(row["relevant_menus"]).split(",")
            if m.strip()
        ]

        relevant_new = []
        for m in relevant_raw:
            match = find_best_match(m, menu_list)
            if match:
                relevant_new.append(match)
            else:
                print(f"⚠️ Menu relevan '{m}' tidak ditemukan di dataset baru!")

        results.append({
            "user_id": user_id,
            "kalori_target": row["kalori_target"],
            "kategori_lauk": row["kategori_lauk"],
            "sumber_karbo": row["sumber_karbo"],
            "deskripsi_preferensi": row["deskripsi_preferensi"],
            "relevant_menus": ",".join(relevant_new)
        })

        print(f"User {user_id} → {len(relevant_new)} relevan cocok")

    df_out = pd.DataFrame(results)
    df_out.to_csv(GT_OUT, index=False)

    print("\n🎉 ground_truth_v4.csv berhasil dibuat!")
    print(f"📊 Total users: {len(df_out)}")
    print("➡ Gunakan file ini untuk evaluasi Hybrid/MCCBF.")


if __name__ == "__main__":
    main()
