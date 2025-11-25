import pandas as pd
import numpy as np
from utils.mccbf_engine import MCCBFEngine
import os
import re

def normalize_menu_name(name):
    """
    Normalisasi nama menu agar konsisten antara ground truth dan rekomendasi
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name


class MCCBFEvaluator:
    """
    Evaluator untuk MCCBF dengan support multi-mode:
    - seimbang
    - fokus_deskripsi
    - fokus_lauk
    """

    def __init__(self, ground_truth_path, data_path, engine=None):
        self.ground_truth = pd.read_csv(ground_truth_path)

        # Engine multimode
        self.engine = engine if engine else MCCBFEngine(data_path=data_path)

        print("✅ Evaluator berhasil diinisialisasi")
        print(f"📊 Jumlah test cases: {len(self.ground_truth)}")
        print(f"🍽️ Jumlah menu: {len(self.engine.df)}")

        self.available_modes = ["seimbang", "fokus_deskripsi", "fokus_lauk"]


    # =============================================================
    # HELPERS
    # =============================================================
    def parse_relevant_menus(self, menu_str):
        if pd.isna(menu_str) or not menu_str:
            return set()
        menus = str(menu_str).split(',')
        return set([normalize_menu_name(m) for m in menus])


    # =============================================================
    # METRIK UNTUK SATU USER
    # =============================================================
    def calculate_metrics_for_user(self, user_row, top_n=5, mode="seimbang", verbose=False):

        # input user
        kalori = int(user_row['kalori_target'])
        kategori = user_row['kategori_lauk']
        karbo = user_row['sumber_karbo']
        deskripsi = user_row['deskripsi_preferensi']

        relevant_menus = self.parse_relevant_menus(user_row['relevant_menus'])

        # REKOMENDASI ENGINE PER MODE
        try:
            rec = self.engine.get_recommendations(
                kalori_target=kalori,
                kategori_lauk=kategori,
                sumber_karbo_list=[karbo],
                deskripsi_preferensi=deskripsi,
                top_n=top_n,
                mode=mode
            )

            recommended = set([normalize_menu_name(n) for n in rec['Nama_Menu']])
        except:
            recommended = set()

        tp = len(relevant_menus & recommended)
        fp = len(recommended - relevant_menus)
        fn = len(relevant_menus - recommended)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn
        }


    # =============================================================
    # EVALUASI SATU MODE
    # =============================================================
    def evaluate_mode(self, mode="seimbang", top_n=5, verbose=False):
        results = []

        for _, row in self.ground_truth.iterrows():
            m = self.calculate_metrics_for_user(row, top_n=top_n, mode=mode, verbose=verbose)
            results.append(m)

        df = pd.DataFrame(results)

        return df, {
            "avg_precision": df['precision'].mean(),
            "avg_recall": df['recall'].mean(),
            "avg_f1": df['f1'].mean(),
            "tp": df['tp'].sum(),
            "fp": df['fp'].sum(),
            "fn": df['fn'].sum()
        }


    # =============================================================
    # EVALUASI SEMUA MODE
    # =============================================================
    def evaluate_all_modes(self, top_n=5, verbose=False):
        summary = {}

        for mode in self.available_modes:
            df, metrics = self.evaluate_mode(mode=mode, top_n=top_n, verbose=verbose)
            summary[mode] = {"df": df, "metrics": metrics}

            print(f"\n==============================")
            print(f"  📌 MODE: {mode.upper()}")
            print("==============================")
            print(f"Precision: {metrics['avg_precision']:.4f}")
            print(f"Recall:    {metrics['avg_recall']:.4f}")
            print(f"F1-Score:  {metrics['avg_f1']:.4f}")
            print(f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

        return summary


    # =============================================================
    # EVALUASI MULTI-K untuk TIAP MODE
    # =============================================================
    def evaluate_multik_modes(self, k_values=[5, 10, 20], verbose=False):
        result = {}

        for k in k_values:
            result[k] = self.evaluate_all_modes(top_n=k, verbose=verbose)

        return result


    # =============================================================
    # SAVE MODE RESULTS
    # =============================================================
    def save_results_per_mode(self, output_dir="model", top_n=5):
        os.makedirs(output_dir, exist_ok=True)
        summary = self.evaluate_all_modes(top_n=top_n)

        for mode in self.available_modes:
            df = summary[mode]['df']
            metrics = summary[mode]['metrics']

            df.to_csv(f"{output_dir}/evaluasi_detail_{mode}.csv", index=False)

            m = pd.DataFrame([{
                "metric": "precision", "value": metrics['avg_precision']
            }, {
                "metric": "recall", "value": metrics['avg_recall']
            }, {
                "metric": "f1_score", "value": metrics['avg_f1']
            }])

            m.to_csv(f"{output_dir}/evaluasi_summary_{mode}.csv", index=False)

        print("📁 Semua mode berhasil disimpan.")


# ========================================
# MAIN SCRIPT
# ========================================
if __name__ == "__main__":
    # Inisialisasi evaluator
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth_v4.csv',
        data_path='data/Preprocessing/data_preprocessed.csv'
    )
    
    # Evaluasi Multi-K
    results_multi_k = evaluator.evaluate_multik_modes(
        k_values=[5, 10, 20],
        verbose=False
    )
    
    # Simpan hasil untuk K=5
    evaluator.save_results_per_mode(output_dir='model', top_n=5)
