# utils/tune_weights.py
import pandas as pd
from evaluate_system import MCCBFEvaluator

def test_weight_scenarios():
    """
    Test berbagai kombinasi bobot
    """
    evaluator = MCCBFEvaluator(
        ground_truth_path='data/ground_truth.csv',
        model_dir='model'
    )
    
    scenarios = [
        # Skenario 1: Bobot Seimbang (Default)
        {
            'name': 'Seimbang (Default)',
            'weights': {'deskripsi': 0.25, 'kategori': 0.25, 'karbohidrat': 0.20, 'kalori': 0.30}
        },
        # Skenario 2: Prioritas Kategori & Karbo
        {
            'name': 'Prioritas Kategori',
            'weights': {'deskripsi': 0.15, 'kategori': 0.35, 'karbohidrat': 0.30, 'kalori': 0.20}
        },
        # Skenario 3: Prioritas Deskripsi
        {
            'name': 'Prioritas Deskripsi',
            'weights': {'deskripsi': 0.40, 'kategori': 0.20, 'karbohidrat': 0.15, 'kalori': 0.25}
        },
        # Skenario 4: Prioritas Kalori
        {
            'name': 'Prioritas Kalori',
            'weights': {'deskripsi': 0.20, 'kategori': 0.20, 'karbohidrat': 0.15, 'kalori': 0.45}
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {scenario['name']}")
        print(f"    Bobot: {scenario['weights']}")
        print(f"{'='*70}")
        
        # Modifikasi engine weights
        evaluator.engine.default_weights = scenario['weights']
        
        # Run evaluation
        df_results, avg_metrics = evaluator.evaluate_all(top_n=5)
        
        results.append({
            'Scenario': scenario['name'],
            'Precision': avg_metrics['avg_precision'],
            'Recall': avg_metrics['avg_recall'],
            'F1-Score': avg_metrics['avg_f1_score']
        })
        
        print(f"\n📊 Hasil:")
        print(f"   Precision: {avg_metrics['avg_precision']:.4f}")
        print(f"   Recall:    {avg_metrics['avg_recall']:.4f}")
        print(f"   F1-Score:  {avg_metrics['avg_f1_score']:.4f}")
    
    # Summary comparison
    df_comparison = pd.DataFrame(results)
    print(f"\n{'='*70}")
    print("📊 PERBANDINGAN SEMUA SKENARIO")
    print(f"{'='*70}\n")
    print(df_comparison.to_string(index=False))
    
    # Save results
    df_comparison.to_csv('model/weight_tuning_results.csv', index=False)
    print(f"\n✅ Hasil disimpan ke: model/weight_tuning_results.csv")

if __name__ == "__main__":
    test_weight_scenarios()