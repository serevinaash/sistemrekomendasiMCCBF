import pandas as pd
import numpy as np
from utils.mccbf_engine import MCCBFEngine
from utils.evaluate_system import MCCBFEvaluator
from itertools import product
import json

class AdvancedWeightTuner:
    """
    Advanced Grid Search untuk mencari kombinasi optimal:
    1. Weight tuning (w_deskripsi, w_lauk, w_karbo, w_kalori)
    2. Gaussian sigma tuning
    3. Keyword boost tuning
    """
    
    def __init__(self, ground_truth_path, data_path):
        self.ground_truth_path = ground_truth_path
        self.data_path = data_path
        self.best_config = None
        self.best_f1 = 0.0
        self.results_history = []
    
    # =========================================================
    # GRID SEARCH: WEIGHT COMBINATIONS
    # =========================================================
    def grid_search_weights(self, mode="seimbang"):
        """
        Grid search untuk bobot optimal
        """
        print("="*70)
        print(f"🔍 GRID SEARCH: Weight Optimization untuk Mode {mode.upper()}")
        print("="*70)
        
        # Define search space (fokus pada range yang masuk akal)
        # Total weight harus = 1.0
        w_deskripsi_range = [0.35, 0.40, 0.45, 0.50, 0.55]
        w_lauk_range = [0.20, 0.25, 0.30]
        w_karbo_range = [0.15, 0.20, 0.25]
        w_kalori_range = [0.05, 0.10, 0.15]
        
        best_f1 = 0.0
        best_weights = None
        results = []
        
        total_combinations = len(w_deskripsi_range) * len(w_lauk_range) * len(w_karbo_range) * len(w_kalori_range)
        print(f"📊 Total kombinasi: {total_combinations}")
        
        tested = 0
        for w_desc in w_deskripsi_range:
            for w_lauk in w_lauk_range:
                for w_karbo in w_karbo_range:
                    for w_kal in w_kalori_range:
                        # Constraint: total weight harus = 1.0
                        total = w_desc + w_lauk + w_karbo + w_kal
                        if abs(total - 1.0) > 0.01:  # Toleransi 1%
                            continue
                        
                        tested += 1
                        
                        # Test kombinasi ini
                        weights = {
                            'w_deskripsi': w_desc,
                            'w_lauk': w_lauk,
                            'w_karbo': w_karbo,
                            'w_kalori': w_kal
                        }
                        
                        # Evaluate
                        engine = MCCBFEngine(data_path=self.data_path)
                        engine.modes[mode] = weights
                        
                        evaluator = MCCBFEvaluator(
                            ground_truth_path=self.ground_truth_path,
                            data_path=self.data_path,
                            engine=engine
                        )
                        
                        _, metrics = evaluator.evaluate_mode(mode=mode, top_n=5, verbose=False)
                        
                        f1 = metrics['avg_f1']
                        
                        results.append({
                            'w_deskripsi': w_desc,
                            'w_lauk': w_lauk,
                            'w_karbo': w_karbo,
                            'w_kalori': w_kal,
                            'precision': metrics['avg_precision'],
                            'recall': metrics['avg_recall'],
                            'f1': f1
                        })
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = weights
                            print(f"\n✨ New Best F1: {f1:.4f}")
                            print(f"   Weights: desc={w_desc}, lauk={w_lauk}, karbo={w_karbo}, kal={w_kal}")
                        
                        # Progress
                        if tested % 10 == 0:
                            print(f"⏳ Progress: {tested}/{total_combinations} tested...")
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST CONFIGURATION (Weights Only)")
        print(f"{'='*70}")
        print(f"F1-Score: {best_f1:.4f}")
        print(f"Weights: {best_weights}")
        
        # Save results
        df_results = pd.DataFrame(results).sort_values('f1', ascending=False)
        df_results.to_csv('model/grid_search_weights.csv', index=False)
        print(f"\n💾 Saved to: model/grid_search_weights.csv")
        
        return best_weights, best_f1, df_results
    
    # =========================================================
    # GRID SEARCH: GAUSSIAN SIGMA
    # =========================================================
    def tune_gaussian_sigma(self, best_weights, mode="seimbang"):
        """
        Tune Gaussian sigma untuk calorie scoring
        """
        print("\n" + "="*70)
        print("🔍 TUNING: Gaussian Sigma")
        print("="*70)
        
        # Sigma range: 30-100 kcal (semakin kecil = semakin strict)
        sigma_range = [30, 40, 50, 60, 70, 80]
        
        best_f1 = 0.0
        best_sigma = 50
        results = []
        
        for sigma in sigma_range:
            # Create custom engine with modified sigma
            engine = MCCBFEngine(data_path=self.data_path)
            engine.modes[mode] = best_weights
            
            # Override _calculate_calorie_score dengan sigma baru
            original_method = engine._calculate_calorie_score
            
            def custom_calorie_score(item_cal, user_cal, sig=sigma):
                return original_method(item_cal, user_cal, sigma=sig)
            
            engine._calculate_calorie_score = custom_calorie_score
            
            # Evaluate
            evaluator = MCCBFEvaluator(
                ground_truth_path=self.ground_truth_path,
                data_path=self.data_path,
                engine=engine
            )
            
            _, metrics = evaluator.evaluate_mode(mode=mode, top_n=5, verbose=False)
            
            f1 = metrics['avg_f1']
            
            results.append({
                'sigma': sigma,
                'precision': metrics['avg_precision'],
                'recall': metrics['avg_recall'],
                'f1': f1
            })
            
            print(f"Sigma={sigma:3d} → F1={f1:.4f} (P={metrics['avg_precision']:.4f}, R={metrics['avg_recall']:.4f})")
            
            if f1 > best_f1:
                best_f1 = f1
                best_sigma = sigma
        
        print(f"\n🏆 Best Sigma: {best_sigma} (F1={best_f1:.4f})")
        
        return best_sigma, best_f1
    
    # =========================================================
    # GRID SEARCH: KEYWORD BOOST
    # =========================================================
    def tune_keyword_boost(self, best_weights, best_sigma, mode="seimbang"):
        """
        Tune keyword boost multiplier & max cap
        """
        print("\n" + "="*70)
        print("🔍 TUNING: Keyword Boost")
        print("="*70)
        
        # Current: 0.08 per keyword, max 0.3
        # Test range
        boost_per_keyword = [0.06, 0.08, 0.10, 0.12]
        max_boost = [0.25, 0.30, 0.35, 0.40]
        
        best_f1 = 0.0
        best_boost = (0.08, 0.3)
        results = []
        
        for boost_val in boost_per_keyword:
            for max_val in max_boost:
                # Create engine with modified boost logic
                # (Ini perlu modifikasi engine, untuk sekarang kita skip dulu)
                # Placeholder untuk future implementation
                pass
        
        print("⚠️ Keyword boost tuning requires engine modification")
        print("   Using current values: boost=0.08, max=0.3")
        
        return best_boost, best_f1
    
    # =========================================================
    # COMBINED OPTIMIZATION
    # =========================================================
    def optimize_all(self, mode="seimbang"):
        """
        Run full optimization pipeline
        """
        print("\n" + "="*70)
        print("🚀 FULL OPTIMIZATION PIPELINE")
        print("="*70)
        
        # Step 1: Weight optimization
        print("\n📍 Step 1/3: Optimizing Weights...")
        best_weights, f1_weights, _ = self.grid_search_weights(mode=mode)
        
        # Step 2: Sigma optimization
        print("\n📍 Step 2/3: Optimizing Gaussian Sigma...")
        best_sigma, f1_sigma = self.tune_gaussian_sigma(best_weights, mode=mode)
        
        # Step 3: Keyword boost (placeholder)
        print("\n📍 Step 3/3: Keyword Boost (using defaults)...")
        
        # Final evaluation with best config
        print("\n" + "="*70)
        print("🎯 FINAL EVALUATION WITH OPTIMIZED CONFIG")
        print("="*70)
        
        # Create optimized engine
        engine = MCCBFEngine(data_path=self.data_path)
        engine.modes[mode] = best_weights
        
        # Apply sigma (requires engine modification to accept sigma parameter)
        # For now, we document it
        
        evaluator = MCCBFEvaluator(
            ground_truth_path=self.ground_truth_path,
            data_path=self.data_path,
            engine=engine
        )
        
        _, final_metrics = evaluator.evaluate_mode(mode=mode, top_n=5, verbose=False)
        
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Precision: {final_metrics['avg_precision']:.4f}")
        print(f"   Recall:    {final_metrics['avg_recall']:.4f}")
        print(f"   F1-Score:  {final_metrics['avg_f1']:.4f}")
        
        # Save config
        config = {
            'mode': mode,
            'weights': best_weights,
            'sigma': best_sigma,
            'keyword_boost_per': 0.08,
            'keyword_boost_max': 0.3,
            'final_f1': final_metrics['avg_f1'],
            'final_precision': final_metrics['avg_precision'],
            'final_recall': final_metrics['avg_recall']
        }
        
        with open(f'model/optimized_config_{mode}.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Config saved to: model/optimized_config_{mode}.json")
        
        return config


# =========================================================
# SIMPLE TUNER (FASTER, TARGETED)
# =========================================================
def quick_tune_targeted():
    """
    Quick tuning dengan fokus pada range yang paling promising
    Berdasarkan analisis: deskripsi weight paling berpengaruh
    """
    print("="*70)
    print("⚡ QUICK TUNE (Targeted Search)")
    print("="*70)
    
    tuner = AdvancedWeightTuner(
        ground_truth_path='data/ground_truth_v4.csv',
        data_path='data/Preprocessing/data_preprocessed.csv'
    )
    
    # Focused search: boost deskripsi weight
    print("\n🎯 Hypothesis: Increase w_deskripsi will improve F1")
    print("   Current best: w_deskripsi=0.45 → F1=0.7560")
    print("   Testing: w_deskripsi=0.50-0.55")
    
    candidates = [
        # Format: (w_desc, w_lauk, w_karbo, w_kal)
        (0.50, 0.25, 0.15, 0.10),  # Boost desc, reduce karbo
        (0.52, 0.23, 0.15, 0.10),  # Boost desc more
        (0.48, 0.27, 0.15, 0.10),  # Balance desc+lauk
        (0.50, 0.20, 0.20, 0.10),  # Boost desc, keep karbo
        (0.48, 0.25, 0.17, 0.10),  # Slight boost all
    ]
    
    results = []
    
    for w_desc, w_lauk, w_karbo, w_kal in candidates:
        weights = {
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal
        }
        
        engine = MCCBFEngine(data_path='data/Preprocessing/data_preprocessed.csv')
        engine.modes['seimbang'] = weights
        
        evaluator = MCCBFEvaluator(
            ground_truth_path='data/ground_truth_v4.csv',
            data_path='data/Preprocessing/data_preprocessed.csv',
            engine=engine
        )
        
        _, metrics = evaluator.evaluate_mode(mode='seimbang', top_n=5, verbose=False)
        
        results.append({
            'w_deskripsi': w_desc,
            'w_lauk': w_lauk,
            'w_karbo': w_karbo,
            'w_kalori': w_kal,
            'precision': metrics['avg_precision'],
            'recall': metrics['avg_recall'],
            'f1': metrics['avg_f1']
        })
        
        print(f"\n{w_desc:.2f}|{w_lauk:.2f}|{w_karbo:.2f}|{w_kal:.2f} → F1={metrics['avg_f1']:.4f} (P={metrics['avg_precision']:.4f}, R={metrics['avg_recall']:.4f})")
    
    # Find best
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    best = df.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST CONFIG:")
    print(f"{'='*70}")
    print(f"F1-Score: {best['f1']:.4f}")
    print(f"Weights: desc={best['w_deskripsi']}, lauk={best['w_lauk']}, karbo={best['w_karbo']}, kal={best['w_kalori']}")
    
    df.to_csv('model/quick_tune_results.csv', index=False)
    print(f"\n💾 Saved to: model/quick_tune_results.csv")
    
    return df


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    
    if mode == "quick":
        # Quick targeted search (recommended untuk test dulu)
        quick_tune_targeted()
    
    elif mode == "full":
        # Full grid search (slow, tapi comprehensive)
        tuner = AdvancedWeightTuner(
            ground_truth_path='data/ground_truth_v4.csv',
            data_path='data/Preprocessing/data_preprocessed.csv'
        )
        
        config = tuner.optimize_all(mode='seimbang')
        print(f"\n✅ Optimization complete! Check model/optimized_config_seimbang.json")
    
    else:
        print("Usage: python tune_weights_advanced.py [quick|full]")
        print("  quick: Fast targeted search (recommended)")
        print("  full:  Comprehensive grid search (slow)")