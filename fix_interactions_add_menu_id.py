import pandas as pd

MENU_PATH = "dataset450_clean_mccbf.csv"
INTER_PATH = "training_interactions.csv"
OUTPUT_PATH = "training_interactions_fixed.csv"

def main():
    print("🔧 Load menu data...")
    menu_df = pd.read_csv(MENU_PATH)

    # Normalisasi nama menu agar cocok
    menu_df["Nama_Menu_clean"] = menu_df["Nama_Menu"].str.lower().str.strip()

    print("🔧 Load interaction data...")
    inter = pd.read_csv(INTER_PATH)

    # Normalisasi menu name di interactions
    inter["menu_name_clean"] = inter["menu_name"].str.lower().str.strip()

    # Merge untuk mendapatkan menu_id = No
    merged = inter.merge(
        menu_df[["No", "Nama_Menu_clean"]],
        left_on="menu_name_clean",
        right_on="Nama_Menu_clean",
        how="left"
    )

    # Rename No → menu_id
    merged = merged.rename(columns={"No": "menu_id"})

    # Drop kolom bantu
    merged = merged.drop(columns=["Nama_Menu_clean", "menu_name_clean"])

    # Cek apakah ada menu yang gagal dipetakan
    missing = merged[merged["menu_id"].isna()]
    if len(missing) > 0:
        print("⚠️ WARNING: Ada menu yang tidak ditemukan di dataset!")
        print(missing[["menu_name"]])
        print("\nPeriksa ejaan atau preprocess dataset.")
    else:
        print("✅ Semua menu berhasil dipetakan ke ID.")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ File fixed saved → {OUTPUT_PATH}")
    print(f"📊 Total rows: {len(merged)}")


if __name__ == "__main__":
    main()
