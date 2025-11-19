import pandas as pd
import numpy as np
from evaluate_system import MCCBFEvaluator

def comprehensive_weight_tuning():
    """
    Comprehensive weight tuning dengan grid search
    """
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth.csv',
        model_dir='model'
    )
    
    # Grid search scenarios
    scenarios = []
    
    # Skenario 1-4: Variasi bobot deskripsi
    for desc_weight in [0.30, 0.35, 0.40, 0.45]:
        remaining = 1.0 - desc_weight
        scenarios.append({
            'name': f'Deskripsi {int(desc_weight*100)}%',
            'weights': {
                'deskripsi': desc_weight,
                'kategori': remaining * 0.35,
                'karbohidrat': remaining * 0.25,
                'kalori': remaining * 0.40
            }
        })
    
    # Skenario 5-8: Variasi bobot kategori
    for cat_weight in [0.30, 0.35, 0.40]:
        remaining = 1.0 - cat_weight
        scenarios.append({
            'name': f'Kategori {int(cat_weight*100)}%',
            'weights': {
                'deskripsi': remaining * 0.35,
                'kategori': cat_weight,
                'karbohidrat': remaining * 0.25,
                'kalori': remaining * 0.40
            }
        })
    
    # Skenario 9: Balance optimal (hipotesis)
    scenarios.append({
        'name': 'Optimal Balance',
        'weights': {
            'deskripsi': 0.35,
            'kategori': 0.30,
            'karbohidrat': 0.20,
            'kalori': 0.15
        }
    })
    
    results = []
    best_f1 = 0
    best_scenario = None
    
    for idx, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*70}")
        print(f"🧪 [{idx}/{len(scenarios)}] Testing: {scenario['name']}")
        print(f"    Bobot: Desk={scenario['weights']['deskripsi']:.2f}, "
              f"Kat={scenario['weights']['kategori']:.2f}, "
              f"Karbo={scenario['weights']['karbohidrat']:.2f}, "
              f"Kal={scenario['weights']['kalori']:.2f}")
        print(f"{'='*70}")
        
        # Override default weights
        original_method = evaluator.engine.get_recommendations
        
        def get_recs_with_weights(*args, **kwargs):
            kwargs['weights'] = scenario['weights']
            return original_method(*args, **kwargs)
        
        evaluator.engine.get_recommendations = get_recs_with_weights
        
        # Evaluate
        df_results, avg_metrics = evaluator.evaluate_all(top_n=5)
        
        results.append({
            'Scenario': scenario['name'],
            'Precision': avg_metrics['avg_precision'],
            'Recall': avg_metrics['avg_recall'],
            'F1-Score': avg_metrics['avg_f1_score'],
            'Bobot_Deskripsi': scenario['weights']['deskripsi'],
            'Bobot_Kategori': scenario['weights']['kategori'],
            'Bobot_Karbohidrat': scenario['weights']['karbohidrat'],
            'Bobot_Kalori': scenario['weights']['kalori']
        })
        
        print(f"📊 Hasil: Precision={avg_metrics['avg_precision']:.4f}, "
              f"Recall={avg_metrics['avg_recall']:.4f}, "
              f"F1={avg_metrics['avg_f1_score']:.4f}")
        
        # Track best
        if avg_metrics['avg_f1_score'] > best_f1:
            best_f1 = avg_metrics['avg_f1_score']
            best_scenario = scenario
    
    # Summary
    df_comparison = pd.DataFrame(results)
    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
    
    print(f"\n{'='*70}")
    print("🏆 HASIL AKHIR - RANKING BY F1-SCORE")
    print(f"{'='*70}\n")
    print(df_comparison.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"✨ BEST SCENARIO: {best_scenario['name']}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Bobot Optimal:")
    for key, val in best_scenario['weights'].items():
        print(f"      • {key}: {val:.2f}")
    print(f"{'='*70}")
    
    # Save
    df_comparison.to_csv('model/weight_tuning_comprehensive.csv', index=False)
    print(f"\n✅ Hasil disimpan ke: model/weight_tuning_comprehensive.csv")
    
    return df_comparison, best_scenario

if __name__ == "__main__":
    df_results, best_config = comprehensive_weight_tuning()