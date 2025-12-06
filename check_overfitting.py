"""
Check for overfitting/data leakage in evaluation
"""
import pandas as pd
from utils.mccbf_engine import MCCBFEngine

def check_for_overfitting():
    """
    Test if ground truth is leaking into evaluation
    """
    print("🔍 CHECKING FOR OVERFITTING / DATA LEAKAGE")
    print("="*70)
    
    # Load ground truth
    gt = pd.read_csv('data/ground_truth_v4.csv')
    print(f"✅ Loaded {len(gt)} test cases\n")
    
    # Load engine
    engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
    
    # Test each case
    exact_matches = 0
    partial_matches = 0
    
    print("Testing each case...\n")
    
    for idx, row in gt.iterrows():
        user_id = row['user_id']
        
        # Get recommendations from current model
        recs = engine.get_recommendations(
            kalori_target=int(row['kalori_target']),
            kategori_lauk=row['kategori_lauk'],
            sumber_karbo_list=[row['sumber_karbo']],
            deskripsi_preferensi=row['deskripsi_preferensi'],
            top_n=5
        )
        
        # Get ground truth
        gt_menus = set([m.strip().lower() for m in row['relevant_menus'].split(',')])
        model_menus = set([m.strip().lower() for m in recs['Nama_Menu'].values])
        
        # Check overlap
        overlap = len(gt_menus & model_menus)
        
        if overlap == 5:
            exact_matches += 1
            status = "🔴 EXACT MATCH (Suspicious!)"
        elif overlap >= 3:
            partial_matches += 1
            status = f"🟡 PARTIAL ({overlap}/5)"
        else:
            status = f"🟢 DIFFERENT ({overlap}/5)"
        
        print(f"User {user_id}: {status}")
        
        if overlap == 5:
            print(f"   GT:    {list(gt_menus)[:3]}...")
            print(f"   Model: {list(model_menus)[:3]}...")
    
    # Summary
    print("\n" + "="*70)
    print("📊 OVERFITTING ANALYSIS:")
    print("="*70)
    
    exact_pct = (exact_matches / len(gt)) * 100
    partial_pct = (partial_matches / len(gt)) * 100
    
    print(f"\n🔴 Exact matches (5/5):  {exact_matches}/{len(gt)} ({exact_pct:.1f}%)")
    print(f"🟡 Partial matches (3+): {partial_matches}/{len(gt)} ({partial_pct:.1f}%)")
    print(f"🟢 Different:            {len(gt) - exact_matches - partial_matches}/{len(gt)}")
    
    # Verdict
    print("\n" + "="*70)
    print("🎯 VERDICT:")
    print("="*70)
    
    if exact_pct > 80:
        print("🚨 SEVERE OVERFITTING DETECTED!")
        print("   → Ground truth is generated from the same model")
        print("   → Precision=1.0 is NOT realistic")
        print("   → Need INDEPENDENT ground truth for valid evaluation")
    elif exact_pct > 50:
        print("⚠️  MODERATE OVERFITTING")
        print("   → Some data leakage present")
        print("   → Results are optimistic")
    else:
        print("✅ NO MAJOR OVERFITTING")
        print("   → Ground truth is sufficiently independent")
        print("   → Results are trustworthy")
    
    return exact_matches, partial_matches


def test_with_perturbed_weights():
    """
    Test with different weights to see if performance drops
    (should drop if overfitting)
    """
    print("\n\n🧪 ROBUSTNESS TEST: Different Weights")
    print("="*70)
    
    from utils.evaluate_system import MCCBFEvaluator
    
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth_v5.csv',
        data_path='data/Preprocessing/data_preprocessed.csv'
    )
    
    # Test with different weights
    weight_configs = [
        {'name': 'Current (Balanced)', 'weights': {'w_kalori': 0.25, 'w_lauk': 0.25, 'w_karbo': 0.25, 'w_deskripsi': 0.25}},
        {'name': 'Random 1', 'weights': {'w_kalori': 0.40, 'w_lauk': 0.20, 'w_karbo': 0.20, 'w_deskripsi': 0.20}},
        {'name': 'Random 2', 'weights': {'w_kalori': 0.20, 'w_lauk': 0.30, 'w_karbo': 0.30, 'w_deskripsi': 0.20}},
        {'name': 'Random 3', 'weights': {'w_kalori': 0.15, 'w_lauk': 0.25, 'w_karbo': 0.25, 'w_deskripsi': 0.35}},
    ]
    
    results = []
    
    for config in weight_configs:
        print(f"\nTesting: {config['name']}")
        
        # Override weights
        original_method = evaluator.engine.get_recommendations
        
        def get_rec_with_weights(*args, **kwargs):
            kwargs['weights'] = config['weights']
            return original_method(*args, **kwargs)
        
        evaluator.engine.get_recommendations = get_rec_with_weights
        
        # Evaluate
        df_res, metrics = evaluator.evaluate_all(top_n=5, verbose=False)
        
        # Restore
        evaluator.engine.get_recommendations = original_method
        
        results.append({
            'Config': config['name'],
            'Precision': metrics['avg_precision'],
            'Recall': metrics['avg_recall'],
            'F1': metrics['avg_f1_score']
        })
        
        print(f"   F1 = {metrics['avg_f1_score']:.4f}")
    
    # Analysis
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("📊 RESULTS:")
    print("="*70)
    print(df.to_string(index=False))
    
    # Check variance
    f1_std = df['F1'].std()
    
    print(f"\n📈 F1-Score Std Dev: {f1_std:.4f}")
    
    if f1_std < 0.05:
        print("🚨 OVERFITTING: Performance too stable across different configs!")
        print("   → Model memorized ground truth")
    elif f1_std < 0.15:
        print("⚠️  POSSIBLE OVERFITTING: Low variance in performance")
    else:
        print("✅ GOOD: Performance varies with different weights")


if __name__ == "__main__":
    # Test 1: Check exact matches
    exact, partial = check_for_overfitting()
    
    # Test 2: Robustness test
    test_with_perturbed_weights()
    
    print("\n\n" + "="*70)
    print("💡 RECOMMENDATIONS:")
    print("="*70)
    print("""
    If OVERFITTING detected:
    
    1. CREATE MANUAL GROUND TRUTH
       - Get 20-30 test queries from real users or experts
       - Manually label relevant menus (don't use model!)
       
    2. USE DIFFERENT SCORING for GT
       - Generate GT with simplified scoring
       - Or use random sampling from top-20
       
    3. ADD NOISE to test robustness
       - Test with typos, synonyms, edge cases
       
    4. CROSS-VALIDATION
       - Split data properly
       - Ensure train/test separation
    """)