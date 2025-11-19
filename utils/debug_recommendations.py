# utils/debug_recommendations.py
import pandas as pd
from mccbf_engine import MCCBFEngine
import os

def debug_single_user(user_id, ground_truth_path='data/ground_truth.csv', model_dir='model'):
    """
    Debug rekomendasi untuk 1 user tertentu
    """
    # Load data
    gt = pd.read_csv(ground_truth_path)
    user = gt[gt['user_id'] == user_id].iloc[0]
    
    # Load engine
    vectorizer_path = os.path.join(model_dir, 'vectorizer_tfidf.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    data_train_path = os.path.join(model_dir, 'data_train.csv')
    engine = MCCBFEngine(vectorizer_path, scaler_path, data_train_path)
    
    # Parse input
    karbo_list = [k.strip() for k in str(user['sumber_karbo']).split(',')]
    
    # Get recommendations
    recs = engine.get_recommendations(
        kalori_target=int(user['kalori_target']),
        kategori_lauk=user['kategori_lauk'],
        sumber_karbo_list=karbo_list,
        deskripsi_preferensi=user['deskripsi_preferensi'],
        top_n=10  # Ambil 10 untuk lihat lebih banyak
    )
    
    # Parse ground truth
    gt_menus = set([m.strip().lower() for m in str(user['relevant_menus']).split(',')])
    
    print(f"\n{'='*70}")
    print(f"🔍 DEBUG USER {user_id}")
    print(f"{'='*70}")
    print(f"\n📝 INPUT:")
    print(f"   Kalori:     {user['kalori_target']} kcal")
    print(f"   Kategori:   {user['kategori_lauk']}")
    print(f"   Karbohidrat: {karbo_list}")
    print(f"   Deskripsi:  {user['deskripsi_preferensi']}")
    
    print(f"\n✅ GROUND TRUTH ({len(gt_menus)} menu):")
    for m in gt_menus:
        print(f"   • {m}")
    
    print(f"\n🤖 REKOMENDASI SISTEM (Top-10):")
    for idx, row in recs.iterrows():
        menu_name = row['Nama_Menu'].lower()
        is_relevant = "✓" if menu_name in gt_menus else "✗"
        print(f"   {is_relevant} [{row['Rank']}] {row['Nama_Menu']} (skor: {row['Skor_Similarity']:.3f})")
        print(f"       {row['Kategori']} | {row['Kalori_(kcal)']} kcal | {row['Sumber_Karbohidrat']}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    # Debug user yang hasilnya jelek
    for user_id in [1, 5, 13, 19]:  # User dengan precision 0%
        debug_single_user(user_id)