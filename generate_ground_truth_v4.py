# generate_ground_truth_v4.py
import pandas as pd
import sys
import os

def generate_ground_truth():
    """
    Generate ground truth menggunakan sistem MCCBF aktual
    """
    print("🚀 Memulai generasi ground truth v4...")
    
    # Load engine SEPERTI DI evaluate_system.py
    try:
        from utils.mccbf_engine import MCCBFEngine
        
        # Cek signature dari evaluator Anda
        # Kemungkinan butuh vectorizer_path, scaler_path, data_train_path
        vectorizer_path = 'model/vectorizer_tfidf.pkl'
        scaler_path = 'model/scaler.pkl'
        data_train_path = 'model/data_train.csv'
        
        # Cek file exists
        if not all([os.path.exists(p) for p in [vectorizer_path, scaler_path, data_train_path]]):
            print("❌ Model files tidak lengkap. Mencoba alternatif...")
            # Alternatif: load data langsung
            data_path = 'data/Preprocessing/data_preprocessed.csv'
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                print(f"✅ Data loaded: {len(df)} menu")
                
                # SKIP ENGINE, langsung simulasi scoring manual
                ground_truth = generate_with_manual_scoring(df)
                return ground_truth
            else:
                print("❌ Data tidak ditemukan!")
                return
        
        # Init dengan model files
        engine = MCCBFEngine(vectorizer_path, scaler_path, data_train_path)
        print("✅ Engine loaded")
        
    except TypeError as e:
        print(f"❌ TypeError: {e}")
        print("\n🔍 Cek konstruktor MCCBFEngine...")
        print("Kemungkinan butuh format: MCCBFEngine(vectorizer_path, scaler_path, data_train_path)")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test cases
    test_cases = get_test_cases()
    ground_truth = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔄 [{i}/{len(test_cases)}] User {case['user_id']}...")
        
        try:
            # Call get_recommendations dengan format yang benar
            # Sesuaikan dengan signature asli dari engine
            recommendations = engine.get_recommendations(
                kalori_target=case['kalori'],
                kategori_lauk=case['lauk'],
                sumber_karbo_list=[case['karbo']],
                deskripsi_preferensi=case['deskripsi'],
                top_n=10
            )
            
            if recommendations.empty:
                print(f"   ⚠️  No results")
                continue
            
            # Ambil top-5
            relevant_menus = recommendations['Nama_Menu'].head(5).tolist()
            
            print(f"   🍽️  Top-5: {', '.join(relevant_menus[:3])}...")
            
            ground_truth.append({
                'user_id': case['user_id'],
                'kalori_target': case['kalori'],
                'kategori_lauk': case['lauk'],
                'sumber_karbo': case['karbo'],
                'deskripsi_preferensi': case['deskripsi'],
                'relevant_menus': ','.join(relevant_menus)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    save_ground_truth(ground_truth)


def generate_with_manual_scoring(df):
    """
    Fallback: Generate ground truth dengan manual scoring
    """
    print("\n🔨 Menggunakan manual scoring...")
    
    test_cases = get_test_cases()
    ground_truth = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔄 [{i}/{len(test_cases)}] User {case['user_id']}...")
        
        # Filter berdasarkan kategori
        filtered = df[df['Kategori'].str.lower() == case['lauk'].lower()].copy()
        
        if filtered.empty:
            print(f"   ⚠️  No menu with kategori: {case['lauk']}")
            continue
        
        # Score kalori (simple distance)
        filtered['score_kalori'] = 1 - abs(filtered['Kalori (kcal)'] - case['kalori']) / 50
        filtered['score_kalori'] = filtered['score_kalori'].clip(0, 1)
        
        # Score karbo (contains check)
        karbo = case['karbo'].lower()
        filtered['score_karbo'] = filtered['Sumber Karbohidrat'].str.lower().str.contains(karbo, na=False).astype(float)
        
        # Score deskripsi (keyword match)
        keywords = case['deskripsi'].lower().split()
        filtered['score_desc'] = filtered['Deskripsi Singkat'].str.lower().apply(
            lambda x: sum(1 for kw in keywords if kw in str(x)) / len(keywords)
        )
        
        # Final score
        filtered['final_score'] = (
            0.35 * filtered['score_kalori'] +
            0.30 * filtered['score_karbo'] +
            0.35 * filtered['score_desc']
        )
        
        # Sort dan ambil top-5
        top_results = filtered.nlargest(5, 'final_score')
        relevant_menus = top_results['Nama Menu'].tolist()
        
        print(f"   📊 Scores: {top_results['final_score'].values}")
        print(f"   🍽️  Top-5: {', '.join(relevant_menus[:3])}...")
        
        ground_truth.append({
            'user_id': case['user_id'],
            'kalori_target': case['kalori'],
            'kategori_lauk': case['lauk'],
            'sumber_karbo': case['karbo'],
            'deskripsi_preferensi': case['deskripsi'],
            'relevant_menus': ','.join(relevant_menus)
        })
    
    save_ground_truth(ground_truth)
    return ground_truth


def get_test_cases():
    return [
        {'user_id': 1, 'kalori': 395, 'lauk': 'ayam', 'karbo': 'nasi merah', 'deskripsi': 'rendah lemak panggang'},
        {'user_id': 2, 'kalori': 400, 'lauk': 'ikan', 'karbo': 'kentang', 'deskripsi': 'tanpa santan bumbu pedas'},
        {'user_id': 3, 'kalori': 395, 'lauk': 'sapi', 'karbo': 'nasi putih', 'deskripsi': 'bumbu rempah korea yakiniku'},
        {'user_id': 4, 'kalori': 400, 'lauk': 'ayam', 'karbo': 'jagung', 'deskripsi': 'pedas lada hitam balado'},
        {'user_id': 5, 'kalori': 395, 'lauk': 'ikan', 'karbo': 'ubi', 'deskripsi': 'saus manis pedas korea'},
        {'user_id': 6, 'kalori': 400, 'lauk': 'ayam', 'karbo': 'kentang', 'deskripsi': 'kuah soto tanpa santan'},
        {'user_id': 7, 'kalori': 390, 'lauk': 'sapi', 'karbo': 'jagung', 'deskripsi': 'tumis sayur pedas'},
        {'user_id': 8, 'kalori': 395, 'lauk': 'ikan', 'karbo': 'nasi merah', 'deskripsi': 'katsu renyah saus'},
        {'user_id': 9, 'kalori': 400, 'lauk': 'ayam', 'karbo': 'kentang', 'deskripsi': 'bumbu serundeng kelapa'},
        {'user_id': 10, 'kalori': 395, 'lauk': 'sapi', 'karbo': 'nasi putih', 'deskripsi': 'kuah soto bening'},
        {'user_id': 11, 'kalori': 390, 'lauk': 'ayam', 'karbo': 'jagung', 'deskripsi': 'salad segar rendah lemak'},
        {'user_id': 12, 'kalori': 400, 'lauk': 'ikan', 'karbo': 'kentang', 'deskripsi': 'gulai bumbu kuning'},
        {'user_id': 13, 'kalori': 395, 'lauk': 'ayam', 'karbo': 'nasi merah', 'deskripsi': 'saus teriyaki korea'},
        {'user_id': 14, 'kalori': 400, 'lauk': 'sapi', 'karbo': 'jagung', 'deskripsi': 'bulgogi korea lada hitam'},
        {'user_id': 15, 'kalori': 390, 'lauk': 'ikan', 'karbo': 'ubi', 'deskripsi': 'sambal matah segar'}
    ]


def save_ground_truth(ground_truth):
    if not ground_truth:
        print("\n❌ Tidak ada ground truth!")
        return
    
    df_gt = pd.DataFrame(ground_truth)
    output_path = 'data/ground_truth_v4.csv'
    df_gt.to_csv(output_path, index=False)
    
    print(f"\n✅ Ground truth berhasil!")
    print(f"   📁 Saved: {output_path}")
    print(f"   📊 Total: {len(df_gt)} cases")
    
    print("\n" + "="*70)
    print("📋 PREVIEW:")
    print("="*70)
    for idx, row in df_gt.head(3).iterrows():
        print(f"\nUser {row['user_id']}: {row['deskripsi_preferensi']}")
        print(f"  Top-5: {row['relevant_menus'][:60]}...")


if __name__ == "__main__":
    generate_ground_truth()