# evaluate_3_modes.py
"""
Script untuk evaluasi 3 mode pembobotan MCCBF:
1. Mode Seimbang (Balanced)
2. Mode Fokus Deskripsi 
3. Mode Fokus Lauk
"""

import pandas as pd
import numpy as np
from utils.mccbf_engine import load_mccbf_engine
import re

def normalize_menu_name(name):
    """Normalisasi nama menu untuk matching"""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name

def parse_relevant_menus(menu_str):
    """Parse string menu jadi set"""
    if pd.isna(menu_str) or not menu_str:
        return set()
    menus = str(menu_str).split(',')
    normalized = [normalize_menu_name(m) for m in menus]
    return set(normalized)

def evaluate_mode(engine, ground_truth, mode_name, weights, top_n=5):
    """
    Evaluasi satu mode pembobotan
    
    Returns:
        dict dengan precision, recall, f1_score
    """
    precision_list = []
    recall_list = []
    f1_list = []
    
    print(f"\n{'='*70}")
    print(f"🔬 EVALUASI: {mode_name}")
    print(f"{'='*70}")
    print(f"Bobot: {weights}")
    print()
    
    for idx, row in ground_truth.iterrows():
        # Parse input
        user_id = row['user_id']
        kalori = int(row['kalori_target'])
        kategori = row['kategori_lauk']
        karbo_list = [k.strip() for k in str(row['sumber_karbo']).split(',')]
        deskripsi = row['deskripsi_preferensi']
        
        # Ground truth
        relevant = parse_relevant_menus(row['relevant_menus'])
        
        # Dapatkan rekomendasi
        try:
            recommendations = engine.get_recommendations(
                kalori_target=kalori,
                kategori_lauk=kategori,
                sumber_karbo_list=karbo_list,
                deskripsi_preferensi=deskripsi,
                weights=weights,
                top_n=top_n
            )
            
            # Ekstrak menu yang direkomendasikan
            recommended = set([
                normalize_menu_name(name)
                for name in recommendations['Nama_Menu'].values
            ])
            
        except Exception as e:
            print(f"⚠️  Error user {user_id}: {e}")
            recommended = set()
        
        # Hitung TP, FP, FN
        tp = len(relevant & recommended)
        fp = len(recommended - relevant)
        fn = len(relevant - recommended)
        
        # Hitung metrik
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        
        # Print detail per user
        print(f"👤 User {user_id}:")
        print(f"   Ground Truth: {list(relevant)[:3]}{'...' if len(relevant) > 3 else ''}")
        print(f"   Rekomendasi:  {list(recommended)[:3]}{'...' if len(recommended) > 3 else ''}")
        print(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print()
    
    # Hitung rata-rata
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    
    return {
        'mode': mode_name,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'std_precision': np.std(precision_list),
        'std_recall': np.std(recall_list),
        'std_f1': np.std(f1_list),
        'precision_list': precision_list,
        'recall_list': recall_list,
        'f1_list': f1_list
    }


def main():
    print("="*70)
    print("🎯 EVALUASI KOMPARATIF 3 MODE PEMBOBOTAN MCCBF")
    print("="*70)
    
    # Load engine
    engine = load_mccbf_engine('model')
    
    # Load ground truth
    ground_truth = pd.read_csv('data/ground_truth_v4.csv')
    print(f"\n✅ Dataset loaded: {len(ground_truth)} test cases\n")
    
    # Definisi 3 mode
    modes = {
        '🟢 Mode Seimbang': {
            'deskripsi': 0.45,
            'kategori': 0.25,
            'karbohidrat': 0.20,
            'kalori': 0.10
        },
        '🟠 Mode Fokus Deskripsi': {
            'deskripsi': 0.50,
            'kategori': 0.20,
            'karbohidrat': 0.20,
            'kalori': 0.10
        },
        '🔵 Mode Fokus Lauk': {
            'deskripsi': 0.20,
            'kategori': 0.50,
            'karbohidrat': 0.20,
            'kalori': 0.10
        }
    }
    
    # Evaluasi semua mode
    results = []
    for mode_name, weights in modes.items():
        result = evaluate_mode(engine, ground_truth, mode_name, weights, top_n=5)
        results.append(result)
    
    # ========================================
    # TABEL PERBANDINGAN
    # ========================================
    print("\n" + "="*70)
    print("📊 HASIL PERBANDINGAN 3 MODE PEMBOBOTAN")
    print("="*70)
    
    df_comparison = pd.DataFrame([{
        'Mode': r['mode'],
        'Precision': f"{r['avg_precision']:.4f} (±{r['std_precision']:.4f})",
        'Recall': f"{r['avg_recall']:.4f} (±{r['std_recall']:.4f})",
        'F1-Score': f"{r['avg_f1']:.4f} (±{r['std_f1']:.4f})"
    } for r in results])
    
    print(df_comparison.to_string(index=False))
    
    # ========================================
    # RANKING MODE
    # ========================================
    print("\n" + "="*70)
    print("🏆 RANKING MODE BERDASARKAN F1-SCORE")
    print("="*70)
    
    sorted_results = sorted(results, key=lambda x: x['avg_f1'], reverse=True)
    
    for rank, r in enumerate(sorted_results, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        status = "✅ BEST" if rank == 1 else "⚠️ GOOD" if rank == 2 else "❌ LOW"
        
        print(f"\n{emoji} Rank {rank}: {r['mode']}")
        print(f"   • Precision: {r['avg_precision']:.2%}")
        print(f"   • Recall:    {r['avg_recall']:.2%}")
        print(f"   • F1-Score:  {r['avg_f1']:.2%} {status}")
    
    # ========================================
    # KESIMPULAN & REKOMENDASI
    # ========================================
    print("\n" + "="*70)
    print("💡 KESIMPULAN & REKOMENDASI")
    print("="*70)
    
    best_mode = sorted_results[0]
    
    print(f"\n✅ **Mode Terbaik:** {best_mode['mode']}")
    print(f"   → F1-Score: {best_mode['avg_f1']:.2%}")
    print(f"\n📌 **Rekomendasi untuk Deployment:**")
    print(f"   • Gunakan '{best_mode['mode']}' sebagai mode default di aplikasi")
    print(f"   • Mode lain tetap tersedia sebagai opsi user")
    
    if best_mode['avg_f1'] >= 0.75:
        print(f"\n🎉 **Status:** PASS (Target ≥75% tercapai!)")
    else:
        print(f"\n⚠️  **Status:** PERLU IMPROVEMENT (Target: ≥75%, Saat ini: {best_mode['avg_f1']:.2%})")
    
    # ========================================
    # SIMPAN HASIL
    # ========================================
    df_detailed = pd.DataFrame([{
        'Mode': r['mode'],
        'Avg_Precision': r['avg_precision'],
        'Avg_Recall': r['avg_recall'],
        'Avg_F1': r['avg_f1'],
        'Std_Precision': r['std_precision'],
        'Std_Recall': r['std_recall'],
        'Std_F1': r['std_f1']
    } for r in results])
    
    df_detailed.to_csv('model/evaluasi_3_modes_comparison.csv', index=False)
    print(f"\n✅ Hasil evaluasi disimpan ke: model/evaluasi_3_modes_comparison.csv")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()