import pandas as pd
import numpy as np
from mccbf_engine import MCCBFEngine
from evaluate_system import MCCBFEvaluator
import json

def optimize_fokus_lauk():
    """
    Optimize mode 'fokus_lauk'
    Target: Boost dari 77.41% ke >80%
    
    Strategy:
    - Boost w_lauk (current: 0.50)
    - Test range: 0.50-0.60
    - Adjust other weights accordingly
    """
    print("="*70)
    print("🔍 OPTIMIZING MODE: FOKUS_LAUK")
    print("="*70)
    print("Current: F1=77.41%, Target: >80%")
    print()
    
    # Define search space untuk fokus_lauk
    # Format: (w_deskripsi, w_lauk, w_karbo, w_kalori)
    candidates = [
        # Baseline (current)
        (0.20, 0.50, 0.20, 0.10),
        
        # Strategy 1: Boost lauk lebih tinggi
        (0.15, 0.55, 0.20, 0.10),
        (0.15, 0.55, 0.15, 0.15),
        (0.10, 0.60, 0.20, 0.10),
        
        # Strategy 2: Balance lauk + karbo (seperti seimbang yang sukses)
        (0.20, 0.45, 0.25, 0.10),
        (0.15, 0.50, 0.25, 0.10),
        (0.15, 0.45, 0.30, 0.10),
        
        # Strategy 3: Reduce kalori weight, boost lauk+karbo
        (0.15, 0.55, 0.25, 0.05),
        (0.10, 0.55, 0.30, 0.05),
    ]
    
    results = []
    best_f1 = 0.7741  # Current score
    best_config = None
    
    print(f"Testing {len(candidates)} configurations...\n")
    
    for idx, (w_desc, w_lauk, w_karbo, w_kal) in enumerate(candidates, 1):
        weights = {
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal
        }
        
        # Create engine with new weights
        engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
        engine.modes['fokus_lauk'] = weights
        
        # Evaluate
        evaluator = MCCBFEvaluator(
            ground_truth_path='data/ground_truth_v4.csv',
            data_path='data/Preprocessing/data_preprocessed.csv',
            engine=engine
        )
        
        _, metrics = evaluator.evaluate_mode(mode='fokus_lauk', top_n=5, verbose=False)
        
        f1 = metrics['avg_f1']
        precision = metrics['avg_precision']
        recall = metrics['avg_recall']
        
        results.append({
            'config_id': idx,
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Print progress
        status = "✨ NEW BEST!" if f1 > best_f1 else ""
        print(f"[{idx:2d}] {w_desc:.2f}|{w_lauk:.2f}|{w_karbo:.2f}|{w_kal:.2f} → "
              f"F1={f1:.4f} (P={precision:.4f}, R={recall:.4f}) {status}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = weights
    
    # Save results
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    df.to_csv('model/optimization_fokus_lauk.csv', index=False)
    
    print(f"\n{'='*70}")
    print("🏆 BEST CONFIGURATION (Fokus Lauk)")
    print(f"{'='*70}")
    print(f"F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"Improvement: {(best_f1 - 0.7741)*100:+.2f}%")
    print(f"\nWeights:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    # Save config
    with open('model/optimized_config_fokus_lauk.json', 'w') as f:
        json.dump({
            'mode': 'fokus_lauk',
            'weights': best_config,
            'sigma': 30,
            'f1_score': best_f1,
            'improvement': (best_f1 - 0.7741)
        }, f, indent=2)
    
    print(f"\n💾 Saved: model/optimization_fokus_lauk.csv")
    print(f"💾 Saved: model/optimized_config_fokus_lauk.json")
    
    return best_config, best_f1


def optimize_fokus_deskripsi():
    """
    Optimize mode 'fokus_deskripsi'
    Target: Boost dari 72.60% ke >77%
    
    Strategy:
    - Boost w_deskripsi (current: 0.50)
    - BUT: Seimbang menunjukkan w_desc=0.35 lebih baik!
    - Test lower deskripsi, higher lauk+karbo
    """
    print("\n" + "="*70)
    print("🔍 OPTIMIZING MODE: FOKUS_DESKRIPSI")
    print("="*70)
    print("Current: F1=72.60%, Target: >77%")
    print()
    
    # Define search space untuk fokus_deskripsi
    # INSIGHT: Mode seimbang dengan w_desc=0.35 score LEBIH TINGGI dari mode fokus_desc!
    # Artinya: "fokus deskripsi" seharusnya TIDAK terlalu extreme
    candidates = [
        # Baseline (current)
        (0.50, 0.20, 0.20, 0.10),
        
        # Strategy 1: Reduce deskripsi drastis (follow seimbang pattern)
        (0.40, 0.25, 0.25, 0.10),
        (0.38, 0.27, 0.25, 0.10),
        (0.35, 0.30, 0.25, 0.10),  # Similar to seimbang
        
        # Strategy 2: Moderate deskripsi dengan boost karbo
        (0.45, 0.20, 0.25, 0.10),
        (0.42, 0.23, 0.25, 0.10),
        (0.40, 0.25, 0.25, 0.10),
        
        # Strategy 3: Balanced approach
        (0.38, 0.28, 0.24, 0.10),
        (0.36, 0.29, 0.25, 0.10),
    ]
    
    results = []
    best_f1 = 0.7260  # Current score
    best_config = None
    
    print(f"Testing {len(candidates)} configurations...\n")
    
    for idx, (w_desc, w_lauk, w_karbo, w_kal) in enumerate(candidates, 1):
        weights = {
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal
        }
        
        # Create engine with new weights
        engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
        engine.modes['fokus_deskripsi'] = weights
        
        # Evaluate
        evaluator = MCCBFEvaluator(
            ground_truth_path='data/ground_truth_v4.csv',
            data_path='data/Preprocessing/data_preprocessed.csv',
            engine=engine
        )
        
        _, metrics = evaluator.evaluate_mode(mode='fokus_deskripsi', top_n=5, verbose=False)
        
        f1 = metrics['avg_f1']
        precision = metrics['avg_precision']
        recall = metrics['avg_recall']
        
        results.append({
            'config_id': idx,
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Print progress
        status = "✨ NEW BEST!" if f1 > best_f1 else ""
        print(f"[{idx:2d}] {w_desc:.2f}|{w_lauk:.2f}|{w_karbo:.2f}|{w_kal:.2f} → "
              f"F1={f1:.4f} (P={precision:.4f}, R={recall:.4f}) {status}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = weights
    
    # Save results
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    df.to_csv('model/optimization_fokus_deskripsi.csv', index=False)
    
    print(f"\n{'='*70}")
    print("🏆 BEST CONFIGURATION (Fokus Deskripsi)")
    print(f"{'='*70}")
    print(f"F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"Improvement: {(best_f1 - 0.7260)*100:+.2f}%")
    print(f"\nWeights:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    # Save config
    with open('model/optimized_config_fokus_deskripsi.json', 'w') as f:
        json.dump({
            'mode': 'fokus_deskripsi',
            'weights': best_config,
            'sigma': 30,
            'f1_score': best_f1,
            'improvement': (best_f1 - 0.7260)
        }, f, indent=2)
    
    print(f"\n💾 Saved: model/optimization_fokus_deskripsi.csv")
    print(f"💾 Saved: model/optimized_config_fokus_deskripsi.json")
    
    return best_config, best_f1


def apply_optimized_configs():
    """
    Apply optimized configs to engine and re-evaluate
    """
    print("\n" + "="*70)
    print("📝 APPLYING OPTIMIZED CONFIGS TO ENGINE")
    print("="*70)
    
    # Load optimized configs
    try:
        with open('model/optimized_config_fokus_lauk.json', 'r') as f:
            config_lauk = json.load(f)
        
        with open('model/optimized_config_fokus_deskripsi.json', 'r') as f:
            config_desc = json.load(f)
        
        print("\n✅ Loaded optimized configs")
        print(f"   Fokus Lauk: F1={config_lauk['f1_score']:.4f}")
        print(f"   Fokus Deskripsi: F1={config_desc['f1_score']:.4f}")
        
        # Update mccbf_engine.py
        print("\n📝 To apply these configs, update mccbf_engine.py:")
        print("\nself.modes = {")
        print("    'seimbang': {")
        print("        'w_deskripsi': 0.35,")
        print("        'w_lauk': 0.30,")
        print("        'w_karbo': 0.25,")
        print("        'w_kalori': 0.10")
        print("    },")
        print("    'fokus_deskripsi': {")
        for k, v in config_desc['weights'].items():
            print(f"        '{k}': {v},")
        print("    },")
        print("    'fokus_lauk': {")
        for k, v in config_lauk['weights'].items():
            print(f"        '{k}': {v},")
        print("    }")
        print("}")
        
    except FileNotFoundError:
        print("⚠️ Optimized config files not found. Run optimization first.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    print("="*70)
    print("🚀 OPTIMIZE FOKUS_LAUK & FOKUS_DESKRIPSI MODES")
    print("="*70)
    print()
    
    if mode in ["both", "lauk"]:
        config_lauk, f1_lauk = optimize_fokus_lauk()
    
    if mode in ["both", "deskripsi"]:
        config_desc, f1_desc = optimize_fokus_deskripsi()
    
    if mode == "both":
        # Summary
        print("\n" + "="*70)
        print("📊 OPTIMIZATION SUMMARY")
        print("="*70)
        
        summary_data = [
            {
                'Mode': 'Seimbang',
                'Before': '75.60%',
                'After': '80.01%',
                'Improvement': '+4.41%',
                'Status': '✅ Done'
            },
            {
                'Mode': 'Fokus Lauk',
                'Before': '77.41%',
                'After': f'{f1_lauk*100:.2f}%',
                'Improvement': f'{(f1_lauk-0.7741)*100:+.2f}%',
                'Status': '✅ Optimized'
            },
            {
                'Mode': 'Fokus Deskripsi',
                'Before': '72.60%',
                'After': f'{f1_desc*100:.2f}%',
                'Improvement': f'{(f1_desc-0.7260)*100:+.2f}%',
                'Status': '✅ Optimized'
            }
        ]
        
        df_summary = pd.DataFrame(summary_data)
        print()
        print(df_summary.to_string(index=False))
        
        # Save summary
        df_summary.to_csv('model/optimization_summary_all_modes.csv', index=False)
        print(f"\n💾 Saved: model/optimization_summary_all_modes.csv")
        
        # Show next steps
        apply_optimized_configs()
    
    print("\n✅ Optimization complete!")