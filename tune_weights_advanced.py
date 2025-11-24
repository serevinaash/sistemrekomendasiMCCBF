import pandas as pd
import numpy as np
from utils.mccbf_engine import MCCBFEngine
from utils.evaluate_system import MCCBFEvaluator
import itertools

def grid_search_weights(evaluator, top_n=5):
    """
    Grid search untuk mencari kombinasi bobot terbaik
    """
    print("🔍 Memulai Grid Search untuk Weight Tuning...")
    print("="*70)
    
    # Define weight ranges
    # Total harus = 1.0
    weight_options = {
        'w_kalori': [0.25, 0.30, 0.35, 0.40, 0.45],
        'w_lauk': [0.20, 0.25, 0.30, 0.35],
        'w_karbo': [0.15, 0.20, 0.25],
        'w_deskripsi': [0.15, 0.20, 0.25, 0.30]
    }
    
    best_f1 = 0
    best_weights = None
    results_log = []
    
    # Generate valid weight combinations (yang total = 1.0)
    tested = 0
    for w_kal in weight_options['w_kalori']:
        for w_lauk in weight_options['w_lauk']:
            for w_karbo in weight_options['w_karbo']:
                w_desc = 1.0 - (w_kal + w_lauk + w_karbo)
                
                # Skip jika w_desc tidak valid
                if w_desc < 0.10 or w_desc > 0.35:
                    continue
                
                weights = {
                    'w_kalori': w_kal,
                    'w_lauk': w_lauk,
                    'w_karbo': w_karbo,
                    'w_deskripsi': w_desc
                }
                
                tested += 1
                
                # Test weights
                try:
                    # Temporarily override engine weights
                    original_get_rec = evaluator.engine.get_recommendations
                    
                    def get_rec_with_weights(*args, **kwargs):
                        kwargs['weights'] = weights
                        return original_get_rec(*args, **kwargs)
                    
                    evaluator.engine.get_recommendations = get_rec_with_weights
                    
                    # Evaluate
                    df_results, avg_metrics = evaluator.evaluate_all(top_n=top_n, verbose=False)
                    
                    # Restore original method
                    evaluator.engine.get_recommendations = original_get_rec
                    
                    f1 = avg_metrics['avg_f1_score']
                    precision = avg_metrics['avg_precision']
                    recall = avg_metrics['avg_recall']
                    
                    results_log.append({
                        'w_kalori': w_kal,
                        'w_lauk': w_lauk,
                        'w_karbo': w_karbo,
                        'w_deskripsi': w_desc,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = weights.copy()
                        print(f"\n✨ NEW BEST! F1={f1:.4f}")
                        print(f"   Weights: Kal={w_kal}, Lauk={w_lauk}, Karbo={w_karbo}, Desc={w_desc:.2f}")
                        print(f"   P={precision:.4f}, R={recall:.4f}")
                    
                    if tested % 10 == 0:
                        print(f"   Tested {tested} combinations... Current best F1={best_f1:.4f}")
                
                except Exception as e:
                    print(f"   ⚠️  Error testing weights: {e}")
                    continue
    
    print(f"\n" + "="*70)
    print(f"🏆 BEST WEIGHTS FOUND (from {tested} combinations):")
    print(f"   • w_kalori:    {best_weights['w_kalori']}")
    print(f"   • w_lauk:      {best_weights['w_lauk']}")
    print(f"   • w_karbo:     {best_weights['w_karbo']}")
    print(f"   • w_deskripsi: {best_weights['w_deskripsi']:.2f}")
    print(f"\n   📈 Best F1-Score: {best_f1:.4f}")
    print("="*70)
    
    # Save results
    df_log = pd.DataFrame(results_log)
    df_log = df_log.sort_values('f1_score', ascending=False)
    df_log.to_csv('model/weight_tuning_results.csv', index=False)
    print(f"\n✅ Results saved to: model/weight_tuning_results.csv")
    
    # Show top 5
    print(f"\n📊 TOP 5 WEIGHT COMBINATIONS:")
    print(df_log.head(5).to_string(index=False))
    
    return best_weights, best_f1


def test_specific_weights(evaluator, weights_list, top_n=5):
    """
    Test specific weight combinations
    """
    print("\n🧪 Testing Specific Weight Combinations...")
    print("="*70)
    
    results = []
    
    for idx, weights in enumerate(weights_list, 1):
        print(f"\n[{idx}/{len(weights_list)}] Testing:")
        print(f"   Kal={weights['w_kalori']}, Lauk={weights['w_lauk']}, "
              f"Karbo={weights['w_karbo']}, Desc={weights['w_deskripsi']}")
        
        try:
            # Override weights
            original_get_rec = evaluator.engine.get_recommendations
            
            def get_rec_with_weights(*args, **kwargs):
                kwargs['weights'] = weights
                return original_get_rec(*args, **kwargs)
            
            evaluator.engine.get_recommendations = get_rec_with_weights
            
            # Evaluate
            df_results, avg_metrics = evaluator.evaluate_all(top_n=top_n, verbose=False)
            
            # Restore
            evaluator.engine.get_recommendations = original_get_rec
            
            results.append({
                'config': f"Kal{weights['w_kalori']}_Lauk{weights['w_lauk']}_"
                         f"Karbo{weights['w_karbo']}_Desc{weights['w_deskripsi']}",
                'precision': avg_metrics['avg_precision'],
                'recall': avg_metrics['avg_recall'],
                'f1_score': avg_metrics['avg_f1_score'],
                **weights
            })
            
            print(f"   📊 P={avg_metrics['avg_precision']:.4f}, "
                  f"R={avg_metrics['avg_recall']:.4f}, "
                  f"F1={avg_metrics['avg_f1_score']:.4f}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS:")
    print(df_results[['config', 'precision', 'recall', 'f1_score']].to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    # Initialize evaluator
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth_v3.csv',
        data_path='data/Preprocessing/data_preprocessed.csv'
    )
    
    print("\n" + "="*70)
    print("🎯 ADVANCED WEIGHT TUNING")
    print("="*70)
    
    # Strategy 1: Test some promising combinations first
    print("\n📍 PHASE 1: Testing Promising Combinations")
    
    promising_weights = [
        # Current (baseline)
        {'w_kalori': 0.35, 'w_lauk': 0.25, 'w_karbo': 0.20, 'w_deskripsi': 0.20},
        
        # Focus on exact matches (kategori lauk & karbo)
        {'w_kalori': 0.30, 'w_lauk': 0.35, 'w_karbo': 0.25, 'w_deskripsi': 0.10},
        {'w_kalori': 0.30, 'w_lauk': 0.30, 'w_karbo': 0.30, 'w_deskripsi': 0.10},
        
        # Focus on kalori precision
        {'w_kalori': 0.45, 'w_lauk': 0.25, 'w_karbo': 0.20, 'w_deskripsi': 0.10},
        
        # Balanced with more description weight
        {'w_kalori': 0.30, 'w_lauk': 0.25, 'w_karbo': 0.20, 'w_deskripsi': 0.25},
        
        # Extreme focus on kategori
        {'w_kalori': 0.25, 'w_lauk': 0.40, 'w_karbo': 0.25, 'w_deskripsi': 0.10},
    ]
    # Kombinasi Promising (tambahkan ke tune_weights_advanced.py)
    promising_weights = [
        # Top dari PHASE 1
        {'w_kalori': 0.30, 'w_lauk': 0.25, 'w_karbo': 0.20, 'w_deskripsi': 0.25},
        
        # Varian di sekitarnya
        {'w_kalori': 0.30, 'w_lauk': 0.26, 'w_karbo': 0.19, 'w_deskripsi': 0.25},
        {'w_kalori': 0.29, 'w_lauk': 0.25, 'w_karbo': 0.20, 'w_deskripsi': 0.26},
        {'w_kalori': 0.31, 'w_lauk': 0.24, 'w_karbo': 0.20, 'w_deskripsi': 0.25},
        
        # Extreme deskripsi
        {'w_kalori': 0.28, 'w_lauk': 0.25, 'w_karbo': 0.18, 'w_deskripsi': 0.29},
        {'w_kalori': 0.30, 'w_lauk': 0.23, 'w_karbo': 0.20, 'w_deskripsi': 0.27},
    ]
    
    quick_results = test_specific_weights(evaluator, promising_weights, top_n=5)
    
    # Strategy 2: Full grid search
    print("\n📍 PHASE 2: Grid Search for Optimal Weights")
    user_input = input("\nRun full grid search? (y/n): ")
    
    if user_input.lower() == 'y':
        best_weights, best_f1 = grid_search_weights(evaluator, top_n=5)
        
        print("\n" + "="*70)
        print("🎉 TUNING COMPLETE!")
        print("="*70)
        print(f"\nUpdate your mccbf_engine.py with these weights:")
        print(f"""
weights = {{
    'w_kalori': {best_weights['w_kalori']},
    'w_lauk': {best_weights['w_lauk']},
    'w_karbo': {best_weights['w_karbo']},
    'w_deskripsi': {best_weights['w_deskripsi']:.2f}
}}
        """)
    else:
        print("\n✅ Quick test complete. Use top result from PHASE 1!")