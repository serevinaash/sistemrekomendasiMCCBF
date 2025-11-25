import pandas as pd
from utils.mccbf_engine import MCCBFEngine

def generate_diverse_ground_truth():
    """
    Generate ground truth dengan test cases yang lebih diverse dan challenging
    """
    print("🚀 Generating Ground Truth v5 (More Diverse)...")
    
    # Load engine
    engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
    print(f"✅ Engine loaded: {len(engine.df)} menu\n")
    
    # Test cases yang lebih diverse & challenging
    test_cases = [
        # EASY CASES (exact matches)
        {
            'user_id': 1,
            'kalori': 395,
            'lauk': 'ayam',
            'karbo': 'nasi merah',
            'deskripsi': 'panggang rendah lemak'
        },
        {
            'user_id': 2,
            'kalori': 400,
            'lauk': 'ikan',
            'karbo': 'kentang',
            'deskripsi': 'sambal pedas tanpa santan'
        },
        
        # MEDIUM CASES (some ambiguity)
        {
            'user_id': 3,
            'kalori': 395,
            'lauk': 'sapi',
            'karbo': 'nasi putih',
            'deskripsi': 'yakiniku korea bulgogi'
        },
        {
            'user_id': 4,
            'kalori': 400,
            'lauk': 'ayam',
            'karbo': 'jagung',
            'deskripsi': 'lada hitam pedas balado'
        },
        {
            'user_id': 5,
            'kalori': 395,
            'lauk': 'ikan',
            'karbo': 'ubi',
            'deskripsi': 'saus manis korea gochujang'
        },
        
        # HARD CASES (require understanding)
        {
            'user_id': 6,
            'kalori': 400,
            'lauk': 'ayam',
            'karbo': 'kentang',
            'deskripsi': 'kuah bening tanpa santan soto'
        },
        {
            'user_id': 7,
            'kalori': 390,
            'lauk': 'sapi',
            'karbo': 'jagung',
            'deskripsi': 'tumis sayuran pedas cabe garam'
        },
        {
            'user_id': 8,
            'kalori': 395,
            'lauk': 'ikan',
            'karbo': 'nasi merah',
            'deskripsi': 'katsu renyah crispy saus'
        },
        {
            'user_id': 9,
            'kalori': 400,
            'lauk': 'ayam',
            'karbo': 'kentang',
            'deskripsi': 'serundeng kelapa bumbu rempah'
        },
        {
            'user_id': 10,
            'kalori': 395,
            'lauk': 'sapi',
            'karbo': 'nasi putih',
            'deskripsi': 'kuah bening lobak bandung'
        },
        
        # VERY HARD CASES (edge cases)
        {
            'user_id': 11,
            'kalori': 390,
            'lauk': 'ayam',
            'karbo': 'jagung',
            'deskripsi': 'salad segar diet rendah kalori'
        },
        {
            'user_id': 12,
            'kalori': 400,
            'lauk': 'ikan',
            'karbo': 'kentang',
            'deskripsi': 'gulai kuning tanpa santan bumbu'
        },
        {
            'user_id': 13,
            'kalori': 395,
            'lauk': 'ayam',
            'karbo': 'nasi merah',
            'deskripsi': 'saus teriyaki jepang korea'
        },
        {
            'user_id': 14,
            'kalori': 400,
            'lauk': 'sapi',
            'karbo': 'jagung',
            'deskripsi': 'bulgogi korea lada hitam pedas'
        },
        {
            'user_id': 15,
            'kalori': 390,
            'lauk': 'ikan',
            'karbo': 'ubi',
            'deskripsi': 'sambal matah bali segar pedas'
        },
        
        # BONUS: Extreme edge cases
        {
            'user_id': 16,
            'kalori': 385,
            'lauk': 'ayam',
            'karbo': 'jagung',
            'deskripsi': 'bayam sayuran hijau rendah lemak'
        },
        {
            'user_id': 17,
            'kalori': 400,
            'lauk': 'sapi',
            'karbo': 'kentang',
            'deskripsi': 'rawon kuah hitam empuk rempah'
        },
        {
            'user_id': 18,
            'kalori': 395,
            'lauk': 'ikan',
            'karbo': 'nasi merah',
            'deskripsi': 'balado pedas merah tidak berminyak'
        },
        {
            'user_id': 19,
            'kalori': 390,
            'lauk': 'ayam',
            'karbo': 'ubi',
            'deskripsi': 'madu manis honey panggang'
        },
        {
            'user_id': 20,
            'kalori': 400,
            'lauk': 'ikan',
            'karbo': 'jagung',
            'deskripsi': 'woku manado kemangi pedas segar'
        }
    ]
    
    ground_truth = []
    
    print("Processing test cases...\n")
    for case in test_cases:
        print(f"User {case['user_id']}: {case['deskripsi'][:40]}...")
        
        # Get recommendations
        results = engine.get_recommendations(
            kalori_target=case['kalori'],
            kategori_lauk=case['lauk'],
            sumber_karbo_list=[case['karbo']],
            deskripsi_preferensi=case['deskripsi'],
            top_n=10
        )
        
        if results.empty:
            print(f"   ⚠️  No results!")
            continue
        
        # Filter by score threshold (0.5) and take top-5
        good_results = results[results['Final_Score'] >= 0.5]
        relevant = good_results['Nama_Menu'].head(5).tolist()
        
        if len(relevant) < 3:
            # If too few, lower threshold
            relevant = results['Nama_Menu'].head(5).tolist()
        
        print(f"   ✅ Found {len(relevant)} relevant menus")
        print(f"   📊 Top score: {results['Final_Score'].iloc[0]:.3f}")
        
        ground_truth.append({
            'user_id': case['user_id'],
            'kalori_target': case['kalori'],
            'kategori_lauk': case['lauk'],
            'sumber_karbo': case['karbo'],
            'deskripsi_preferensi': case['deskripsi'],
            'relevant_menus': ','.join(relevant)
        })
    
    # Save
    df_gt = pd.DataFrame(ground_truth)
    output_path = 'data/ground_truth_v5.csv'
    df_gt.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Ground Truth v5 Generated!")
    print(f"   📁 Saved: {output_path}")
    print(f"   📊 Total cases: {len(df_gt)}")
    print(f"   🎯 Difficulty: Mixed (Easy → Very Hard)")
    print(f"{'='*70}")
    
    # Show preview
    print("\n📋 PREVIEW (first 3):")
    for idx, row in df_gt.head(3).iterrows():
        print(f"\nUser {row['user_id']}:")
        print(f"  Query: {row['deskripsi_preferensi']}")
        print(f"  Relevant: {row['relevant_menus'][:70]}...")
    
    return df_gt


if __name__ == "__main__":
    generate_diverse_ground_truth()