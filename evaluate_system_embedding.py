# evaluate_system_embedding.py

import pandas as pd
import numpy as np
import os
import re

# ⬇️ Ganti import engine: pakai versi EMBEDDING
# kalau file kamu bukan di utils/, sesuaikan: from mccbf_engine_we import MCCBFEngineEmbedding
from utils.mccbf_engine_we import MCCBFEngineEmbedding


def normalize_menu_name(name):
    """
    Normalisasi nama menu agar konsisten antara ground truth dan rekomendasi
    """
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)  # hapus tanda baca
    name = re.sub(r'\s+', ' ', name)          # rapikan spasi
    return name


class MCCBFEvaluatorEmbedding:
    """
    Evaluator untuk sistem rekomendasi MCCBF + Word Embedding
    Menghitung Precision, Recall, F1-Score
    """
    
    def __init__(
        self, 
        ground_truth_path, 
        model_dir='model',
        embedding_filename='cc.id.300.bin'   # nama model FastText kamu
    ):
        """
        Args:
            ground_truth_path: path ke ground_truth_v3.csv
            model_dir: direktori model (data_train.csv, embedding)
            embedding_filename: file FastText .bin (misal: cc.id.300.bin)
        """
        # Load ground truth
        self.ground_truth = pd.read_csv(ground_truth_path)
        
        # Path model
        data_train_path = os.path.join(model_dir, 'data_train.csv')
        embedding_path = os.path.join(model_dir, embedding_filename)
        
        # Load MCCBF Engine berbasis embedding
        self.engine = MCCBFEngineEmbedding(
            embedding_path=embedding_path,
            data_train_path=data_train_path
        )
        
        print("✅ Evaluator EMBEDDING berhasil diinisialisasi")
        print(f"   📊 Jumlah test case: {len(self.ground_truth)}")
    
    
    def parse_karbo_list(self, karbo_str):
        """Parse string karbohidrat jadi list"""
        if pd.isna(karbo_str) or not karbo_str:
            return []
        return [k.strip() for k in str(karbo_str).split(',')]
    
    
    def parse_relevant_menus(self, menu_str):
        if pd.isna(menu_str) or not menu_str:
            return set()
        menus = str(menu_str).split(',')
        normalized = [normalize_menu_name(m) for m in menus]
        return set(normalized)
    
    
    def calculate_metrics_for_user(self, user_row, top_n=5):
        """
        Hitung Precision, Recall, F1 untuk 1 user (pakai engine EMBEDDING)
        """
        # 1️⃣ Parse input user
        kalori = int(user_row['kalori_target'])
        kategori = user_row['kategori_lauk']
        karbo_list = self.parse_karbo_list(user_row['sumber_karbo'])
        deskripsi = user_row['deskripsi_preferensi']
        
        # 2️⃣ Ground truth (menu yang benar-benar relevan)
        relevant_menus = self.parse_relevant_menus(user_row['relevant_menus'])
        
        # 3️⃣ Dapatkan rekomendasi dari sistem
        try:
            recommendations = self.engine.get_recommendations(
                kalori_target=kalori,
                kategori_lauk=kategori,
                sumber_karbo_list=karbo_list,
                deskripsi_preferensi=deskripsi,
                top_n=top_n
            )
            
            # Ekstrak nama menu dari hasil rekomendasi
            recommended_menus = set([
                normalize_menu_name(name)
                for name in recommendations['Nama_Menu'].values
            ])
            
        except Exception as e:
            print(f"⚠️  Error untuk user {user_row['user_id']}: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'tp': 0,
                'fp': top_n,
                'fn': len(relevant_menus)
            }
        
        # 4️⃣ Hitung TP, FP, FN
        true_positive = len(relevant_menus & recommended_menus)
        false_positive = len(recommended_menus - relevant_menus)
        false_negative = len(relevant_menus - recommended_menus)
        
        # 5️⃣ Hitung Precision, Recall, F1-Score
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': true_positive,
            'fp': false_positive,
            'fn': false_negative,
            'relevant': list(relevant_menus),
            'recommended': list(recommended_menus)
        }
    
    
    def evaluate_all(self, top_n=5):
        """
        Evaluasi semua user di ground truth
        """
        results = []
        
        for idx, row in self.ground_truth.iterrows():
            user_id = row['user_id']
            metrics = self.calculate_metrics_for_user(row, top_n=top_n)
            
            results.append({
                'user_id': user_id,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'relevant_count': len(metrics['relevant']),
                'recommended_count': len(metrics['recommended'])
            })
        
        df_results = pd.DataFrame(results)
        
        # Hitung agregat metrics
        avg_metrics = {
            'avg_precision': df_results['precision'].mean(),
            'avg_recall': df_results['recall'].mean(),
            'avg_f1_score': df_results['f1_score'].mean(),
            'std_precision': df_results['precision'].std(),
            'std_recall': df_results['recall'].std(),
            'std_f1_score': df_results['f1_score'].std(),
            'total_tp': df_results['tp'].sum(),
            'total_fp': df_results['fp'].sum(),
            'total_fn': df_results['fn'].sum()
        }
        
        return df_results, avg_metrics
    
    
    def print_evaluation_report(self, top_n=5):
        """
        Cetak laporan evaluasi lengkap ke console
        """
        print("\n" + "="*70)
        print(f"📊 EVALUASI SISTEM REKOMENDASI MCCBF + EMBEDDING (Top-{top_n})")
        print("="*70)
        
        df_results, avg_metrics = self.evaluate_all(top_n=top_n)
        
        # Print agregat metrics
        print(f"\n📈 METRIK RATA-RATA:")
        print(f"   • Precision: {avg_metrics['avg_precision']:.4f} (±{avg_metrics['std_precision']:.4f})")
        print(f"   • Recall:    {avg_metrics['avg_recall']:.4f} (±{avg_metrics['std_recall']:.4f})")
        print(f"   • F1-Score:  {avg_metrics['avg_f1_score']:.4f} (±{avg_metrics['std_f1_score']:.4f})")
        
        print(f"\n🎯 CONFUSION MATRIX (TOTAL):")
        print(f"   • True Positive:  {avg_metrics['total_tp']}")
        print(f"   • False Positive: {avg_metrics['total_fp']}")
        print(f"   • False Negative: {avg_metrics['total_fn']}")
        
        # Print detail per user
        print(f"\n📋 DETAIL PER USER:")
        for _, row in df_results.iterrows():
            print(f"\n   User {int(row['user_id'])}:")
            print(f"      Precision: {row['precision']:.4f}")
            print(f"      Recall:    {row['recall']:.4f}")
            print(f"      F1-Score:  {row['f1_score']:.4f}")
            print(f"      TP/FP/FN:  {int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}")
        
        print("\n" + "="*70 + "\n")
        
        return df_results, avg_metrics
    
    
    def save_results(self, output_dir='model', top_n=5):
        """
        Simpan hasil evaluasi ke CSV
        """
        df_results, avg_metrics = self.evaluate_all(top_n=top_n)
        
        # Simpan detail per user
        detail_path = os.path.join(output_dir, 'evaluasi_detail_embedding_detail.csv')
        df_results.to_csv(detail_path, index=False)
        
        # Simpan summary metrics
        summary = pd.DataFrame([{
            'metric': 'Precision',
            'mean': avg_metrics['avg_precision'],
            'std': avg_metrics['std_precision']
        }, {
            'metric': 'Recall',
            'mean': avg_metrics['avg_recall'],
            'std': avg_metrics['std_recall']
        }, {
            'metric': 'F1-Score',
            'mean': avg_metrics['avg_f1_score'],
            'std': avg_metrics['std_f1_score']
        }])
        
        summary_path = os.path.join(output_dir, 'evaluasi_summary_embedding.csv')
        summary.to_csv(summary_path, index=False)
        
        print(f"✅ Hasil evaluasi (EMBEDDING) berhasil disimpan:")
        print(f"   • {detail_path}")
        print(f"   • {summary_path}")
        
        return df_results, avg_metrics


# ========================================
# MAIN SCRIPT
# ========================================
if __name__ == "__main__":
    evaluator = MCCBFEvaluatorEmbedding(
        ground_truth_path='data/ground_truth_v3.csv',
        model_dir='model',
        embedding_filename='cc.id.300.bin'  # ganti kalau nama file embedding beda
    )
    
    df_results, avg_metrics = evaluator.print_evaluation_report(top_n=5)
    evaluator.save_results(output_dir='model', top_n=5)
