import pandas as pd
import numpy as np
from mccbf_engine import MCCBFEngine
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
        except Exception as e:
            if verbose:
                print(f"⚠️ Error untuk user case: {e}")
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

        for idx, row in self.ground_truth.iterrows():
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

        print("\n" + "="*70)
        print("🔍 EVALUATING ALL MODES")
        print("="*70)

        for mode in self.available_modes:
            df, metrics = self.evaluate_mode(mode=mode, top_n=top_n, verbose=verbose)
            summary[mode] = {"df": df, "metrics": metrics}

            print(f"\n{'─'*70}")
            print(f"  📌 MODE: {mode.upper()}")
            print(f"{'─'*70}")
            print(f"  Precision: {metrics['avg_precision']:.4f} ({metrics['avg_precision']*100:.2f}%)")
            print(f"  Recall:    {metrics['avg_recall']:.4f} ({metrics['avg_recall']*100:.2f}%)")
            print(f"  F1-Score:  {metrics['avg_f1']:.4f} ({metrics['avg_f1']*100:.2f}%)")
            print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")

        # Summary comparison
        print(f"\n{'='*70}")
        print("📊 SUMMARY COMPARISON")
        print(f"{'='*70}")
        
        comparison = []
        for mode in self.available_modes:
            comparison.append({
                'Mode': mode.title(),
                'Precision': f"{summary[mode]['metrics']['avg_precision']:.4f}",
                'Recall': f"{summary[mode]['metrics']['avg_recall']:.4f}",
                'F1-Score': f"{summary[mode]['metrics']['avg_f1']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print(df_comparison.to_string(index=False))

        # Find best
        best_mode = max(self.available_modes, key=lambda m: summary[m]['metrics']['avg_f1'])
        best_f1 = summary[best_mode]['metrics']['avg_f1']
        
        print(f"\n🏆 BEST MODE: {best_mode.upper()} (F1={best_f1:.4f})")

        return summary


    # =============================================================
    # EVALUASI MULTI-K untuk TIAP MODE
    # =============================================================
    def evaluate_multik_modes(self, k_values=[5, 10, 20], verbose=False):
        result = {}

        print("\n" + "="*70)
        print("🔍 MULTI-K EVALUATION")
        print("="*70)

        for k in k_values:
            print(f"\n📍 Evaluating with Top-{k}...")
            result[k] = self.evaluate_all_modes(top_n=k, verbose=verbose)

        # Multi-K comparison
        print(f"\n{'='*70}")
        print("📊 MULTI-K COMPARISON (Mode: Seimbang)")
        print(f"{'='*70}")
        
        multik_data = []
        for k in k_values:
            metrics = result[k]['seimbang']['metrics']
            multik_data.append({
                'Top-K': k,
                'Precision': f"{metrics['avg_precision']:.4f}",
                'Recall': f"{metrics['avg_recall']:.4f}",
                'F1-Score': f"{metrics['avg_f1']:.4f}"
            })
        
        df_multik = pd.DataFrame(multik_data)
        print(df_multik.to_string(index=False))

        return result


    # =============================================================
    # SAVE MODE RESULTS
    # =============================================================
    def save_results_per_mode(self, output_dir="model", top_n=5):
        os.makedirs(output_dir, exist_ok=True)
        summary = self.evaluate_all_modes(top_n=top_n)

        print(f"\n{'='*70}")
        print("💾 SAVING RESULTS")
        print(f"{'='*70}")

        for mode in self.available_modes:
            df = summary[mode]['df']
            metrics = summary[mode]['metrics']

            # Save detailed results
            detail_path = f"{output_dir}/evaluasi_detail_{mode}.csv"
            df.to_csv(detail_path, index=False)
            print(f"✅ Saved: {detail_path}")

            # Save summary metrics
            m = pd.DataFrame([{
                "metric": "precision", "value": metrics['avg_precision']
            }, {
                "metric": "recall", "value": metrics['avg_recall']
            }, {
                "metric": "f1_score", "value": metrics['avg_f1']
            }])

            summary_path = f"{output_dir}/evaluasi_summary_{mode}.csv"
            m.to_csv(summary_path, index=False)
            print(f"✅ Saved: {summary_path}")

        # Save comparison table
        comparison = []
        for mode in self.available_modes:
            comparison.append({
                'mode': mode,
                'precision': summary[mode]['metrics']['avg_precision'],
                'recall': summary[mode]['metrics']['avg_recall'],
                'f1_score': summary[mode]['metrics']['avg_f1'],
                'tp': summary[mode]['metrics']['tp'],
                'fp': summary[mode]['metrics']['fp'],
                'fn': summary[mode]['metrics']['fn']
            })
        
        df_comparison = pd.DataFrame(comparison)
        comparison_path = f"{output_dir}/evaluasi_comparison_all_modes.csv"
        df_comparison.to_csv(comparison_path, index=False)
        print(f"✅ Saved: {comparison_path}")

        print(f"\n📁 All results saved to: {output_dir}/")


# ========================================
# MAIN SCRIPT
# ========================================
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("="*70)
    print("🚀 MCCBF EVALUATION SYSTEM")
    print("="*70)
    print(f"Mode: {mode}")
    print(f"{'='*70}\n")
    
    # Inisialisasi evaluator
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth_v4.csv',
        data_path='data/Preprocessing/data_preprocessed.csv'
    )
    
    if mode == "all":
        # Evaluasi semua mode dengan Top-5
        print("\n📍 Running evaluation for all modes (Top-5)...")
        evaluator.save_results_per_mode(output_dir='model', top_n=5)
    
    elif mode == "multik":
        # Evaluasi Multi-K
        print("\n📍 Running multi-K evaluation...")
        results_multi_k = evaluator.evaluate_multik_modes(
            k_values=[5, 10, 20],
            verbose=False
        )
        
        # Save multi-K results
        for k in [5, 10, 20]:
            for mode_name in evaluator.available_modes:
                metrics = results_multi_k[k][mode_name]['metrics']
                df = results_multi_k[k][mode_name]['df']
                
                output_path = f"model/evaluasi_detail_{mode_name}_top{k}.csv"
                df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")
    
    elif mode == "single":
        # Evaluasi mode seimbang saja
        print("\n📍 Running evaluation for mode: seimbang (Top-5)...")
        _, metrics = evaluator.evaluate_mode(mode='seimbang', top_n=5, verbose=False)
        
        print(f"\n{'='*70}")
        print("📊 FINAL RESULTS (Mode: Seimbang)")
        print(f"{'='*70}")
        print(f"  Precision: {metrics['avg_precision']:.4f} ({metrics['avg_precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['avg_recall']:.4f} ({metrics['avg_recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['avg_f1']:.4f} ({metrics['avg_f1']*100:.2f}%)")
        print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    else:
        print("Usage: python evaluate_system.py [all|multik|single]")
        print("  all:    Evaluate all modes (seimbang, fokus_deskripsi, fokus_lauk)")
        print("  multik: Evaluate with multiple K values (5, 10, 20)")
        print("  single: Evaluate only mode 'seimbang'")