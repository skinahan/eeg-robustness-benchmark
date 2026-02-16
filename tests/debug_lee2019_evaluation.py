#!/usr/bin/env python3
"""
Debug script to investigate Lee2019 SSVEP evaluation issues.
Focuses on noise application, per-class performance, and evaluation methodology.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def debug_noise_application():
    """Debug noise application in test_perturb mode."""
    print("=" * 60)
    print("DEBUGGING NOISE APPLICATION")
    print("=" * 60)
    
    print("Key debugging code to add to your evaluation pipeline:")
    print()
    
    debug_code = '''
# Add this to session_evaluator.py in the CV loop (around line 180-184)

# BEFORE applying noise
original_power = np.mean(X_to_corrupt[valid_idx]**2)
original_std = np.std(X_to_corrupt[valid_idx])

print(f"Fold {i} - BEFORE noise:")
print(f"  Original signal power: {original_power:.6f}")
print(f"  Original signal std: {original_std:.6f}")

# Apply noise
X_to_corrupt[valid_idx] = noise_augmentor.transform(X_to_corrupt[valid_idx])

# AFTER applying noise
noisy_power = np.mean(X_to_corrupt[valid_idx]**2)
noisy_std = np.std(X_to_corrupt[valid_idx])

print(f"Fold {i} - AFTER noise:")
print(f"  Noisy signal power: {noisy_power:.6f}")
print(f"  Noisy signal std: {noisy_std:.6f}")
print(f"  Power ratio: {noisy_power/original_power:.3f}")
print(f"  Std ratio: {noisy_std/original_std:.3f}")

# Check if noise actually changed the signal
if abs(noisy_power - original_power) < 1e-10:
    print("⚠️  WARNING: Signal power unchanged - noise may not be applied!")
if abs(noisy_std - original_std) < 1e-10:
    print("⚠️  WARNING: Signal std unchanged - noise may not be applied!")
'''
    
    print(debug_code)
    
    print("\nExpected behavior:")
    print("- Power ratio should be > 1.0 (noise increases power)")
    print("- Std ratio should be > 1.0 (noise increases variability)")
    print("- If ratios are ~1.0, noise is not being applied!")

def debug_per_class_performance():
    """Debug per-class performance analysis."""
    print("=" * 60)
    print("DEBUGGING PER-CLASS PERFORMANCE")
    print("=" * 60)
    
    debug_code = '''
# Add this after model evaluation in your pipeline

def analyze_per_class_performance(y_true, y_pred, y_pred_proba=None, class_names=None):
    """Analyze per-class performance in detail."""
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    
    print("\\n" + "="*50)
    print("PER-CLASS PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    print(f"Accuracy difference: {accuracy - balanced_acc:.3f}")
    
    if abs(accuracy - balanced_acc) > 0.05:
        print("⚠️  WARNING: Large difference between accuracy and balanced accuracy!")
        print("   This suggests class imbalance or poor performance on some classes.")
    
    # Per-class analysis
    print("\\nPer-class performance:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    print("\\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\\nPer-class accuracy:")
    for i in range(len(class_names)):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
            print(f"  {class_names[i]}: {class_acc:.3f} ({class_mask.sum()} samples)")
    
    # ROC-AUC analysis (if probabilities available)
    if y_pred_proba is not None:
        print("\\nROC-AUC Analysis:")
        
        # Overall ROC-AUC (OvR)
        try:
            overall_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            print(f"Overall ROC-AUC (OvR): {overall_auc:.3f}")
        except Exception as e:
            print(f"Overall ROC-AUC calculation failed: {e}")
        
        # Per-class ROC-AUC
        print("Per-class ROC-AUC:")
        for i in range(len(class_names)):
            try:
                # Binary ROC-AUC for class i vs. rest
                y_binary = (y_true == i).astype(int)
                class_auc = roc_auc_score(y_binary, y_pred_proba[:, i])
                print(f"  {class_names[i]}: {class_auc:.3f}")
            except Exception as e:
                print(f"  {class_names[i]}: Failed ({e})")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': cm
    }

# Usage in your evaluation pipeline:
# results = analyze_per_class_performance(y_test, y_pred, y_pred_proba, 
#                                       class_names=['12.0Hz', '5.45Hz', '6.67Hz', '8.57Hz'])
'''
    
    print(debug_code)

def debug_evaluation_methodology():
    """Debug evaluation methodology issues."""
    print("=" * 60)
    print("DEBUGGING EVALUATION METHODOLOGY")
    print("=" * 60)
    
    debug_code = '''
# Add this to check for evaluation methodology issues

def debug_cv_consistency(X, y, cv, model, noise_augmentor=None, intensity=None):
    """Debug cross-validation consistency and independence."""
    
    print("\\n" + "="*50)
    print("CROSS-VALIDATION CONSISTENCY CHECK")
    print("="*50)
    
    fold_scores = []
    fold_predictions = []
    fold_true_labels = []
    
    for i, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        print(f"\\nFold {i}:")
        print(f"  Train size: {len(train_idx)}, Valid size: {len(valid_idx)}")
        
        # Check class distribution in fold
        train_classes = np.bincount(y[train_idx])
        valid_classes = np.bincount(y[valid_idx])
        
        print(f"  Train class distribution: {train_classes}")
        print(f"  Valid class distribution: {valid_classes}")
        
        # Train model
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation set
        X_valid_fold = X[valid_idx]
        y_valid_fold = y[valid_idx]
        
        # Apply noise if specified
        if noise_augmentor is not None:
            print(f"  Applying noise with intensity: {intensity}")
            X_valid_fold = noise_augmentor.transform(X_valid_fold)
        
        # Get predictions
        y_pred_fold = model.predict(X_valid_fold)
        y_pred_proba_fold = model.predict_proba(X_valid_fold)
        
        # Calculate metrics
        fold_acc = accuracy_score(y_valid_fold, y_pred_fold)
        fold_balanced_acc = balanced_accuracy_score(y_valid_fold, y_pred_fold)
        
        # ROC-AUC
        try:
            fold_auc = roc_auc_score(y_valid_fold, y_pred_proba_fold, multi_class='ovr')
        except:
            fold_auc = 0.0
        
        print(f"  Fold accuracy: {fold_acc:.3f}")
        print(f"  Fold balanced accuracy: {fold_balanced_acc:.3f}")
        print(f"  Fold ROC-AUC: {fold_auc:.3f}")
        
        fold_scores.append(fold_auc)
        fold_predictions.extend(y_pred_fold)
        fold_true_labels.extend(y_valid_fold)
    
    # Overall analysis
    print(f"\\nOverall CV Results:")
    print(f"  Mean ROC-AUC: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
    print(f"  Score range: {np.min(fold_scores):.3f} - {np.max(fold_scores):.3f}")
    
    if np.std(fold_scores) > 0.1:
        print("⚠️  WARNING: High variance across CV folds!")
    
    # Check for data leakage
    print(f"\\nData Leakage Check:")
    print(f"  Total unique samples: {len(np.unique(X.reshape(X.shape[0], -1), axis=0))}")
    print(f"  Total samples: {X.shape[0]}")
    
    return fold_scores

# Usage:
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# scores = debug_cv_consistency(X, y, cv, model, noise_augmentor, intensity)
'''
    
    print(debug_code)

def debug_ssvep_specific_issues():
    """Debug SSVEP-specific issues."""
    print("=" * 60)
    print("DEBUGGING SSVEP-SPECIFIC ISSUES")
    print("=" * 60)
    
    debug_code = '''
# Add this to check SSVEP-specific issues

def debug_ssvep_patterns(X, y, class_names=None):
    """Debug SSVEP pattern distinctiveness."""
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    
    print("\\n" + "="*50)
    print("SSVEP PATTERN ANALYSIS")
    print("="*50)
    
    # Analyze signal characteristics per class
    for i, class_name in enumerate(class_names):
        class_mask = y == i
        if class_mask.sum() > 0:
            X_class = X[class_mask]
            
            # Signal statistics
            mean_power = np.mean(X_class**2)
            mean_std = np.mean(np.std(X_class, axis=(1,2)))
            mean_range = np.mean(np.ptp(X_class, axis=(1,2)))
            
            print(f"\\n{class_name} ({class_mask.sum()} samples):")
            print(f"  Mean power: {mean_power:.6f}")
            print(f"  Mean std: {mean_std:.6f}")
            print(f"  Mean range: {mean_range:.6f}")
    
    # Check for trivial separability
    print("\\nTrivial Separability Check:")
    
    # Check if classes have very different power levels
    class_powers = []
    for i in range(len(class_names)):
        class_mask = y == i
        if class_mask.sum() > 0:
            class_power = np.mean(X[class_mask]**2)
            class_powers.append(class_power)
    
    if len(class_powers) > 1:
        power_ratio = max(class_powers) / min(class_powers)
        print(f"  Power ratio between classes: {power_ratio:.3f}")
        
        if power_ratio > 10:
            print("⚠️  WARNING: Classes have very different power levels!")
            print("   Classification might be trivial based on signal power.")
        
        if power_ratio < 1.1:
            print("✅ Classes have similar power levels - good for fair evaluation.")

# Usage:
# debug_ssvep_patterns(X, y, ['12.0Hz', '5.45Hz', '6.67Hz', '8.57Hz'])
'''
    
    print(debug_code)

def create_integrated_debug_script():
    """Create a complete debugging script."""
    print("=" * 60)
    print("COMPLETE DEBUGGING SCRIPT")
    print("=" * 60)
    
    complete_script = '''
#!/usr/bin/env python3
"""
Complete debugging script for Lee2019 SSVEP evaluation issues.
Run this script to identify the root cause of suspiciously high results.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def comprehensive_debug(X, y, model, noise_augmentor=None, intensity=None, class_names=None):
    """
    Comprehensive debugging of Lee2019 SSVEP evaluation pipeline.
    
    Args:
        X: Input data (n_samples, n_channels, n_times)
        y: Labels
        model: Trained model
        noise_augmentor: Noise augmentor (optional)
        intensity: Noise intensity (optional)
        class_names: Class names (optional)
    """
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(np.unique(y)))]
    
    print("=" * 60)
    print("COMPREHENSIVE LEE2019 SSVEP DEBUG")
    print("=" * 60)
    
    # 1. Check data characteristics
    print("\\n1. DATA CHARACTERISTICS:")
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # 2. Check signal characteristics
    print("\\n2. SIGNAL CHARACTERISTICS:")
    overall_power = np.mean(X**2)
    overall_std = np.std(X)
    print(f"   Overall signal power: {overall_power:.6f}")
    print(f"   Overall signal std: {overall_std:.6f}")
    
    # 3. Per-class analysis
    print("\\n3. PER-CLASS ANALYSIS:")
    for i, class_name in enumerate(class_names):
        class_mask = y == i
        if class_mask.sum() > 0:
            X_class = X[class_mask]
            class_power = np.mean(X_class**2)
            class_std = np.std(X_class)
            print(f"   {class_name}: power={class_power:.6f}, std={class_std:.6f}")
    
    # 4. Model evaluation
    print("\\n4. MODEL EVALUATION:")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Balanced Accuracy: {balanced_acc:.3f}")
    
    # 5. Per-class performance
    print("\\n5. PER-CLASS PERFORMANCE:")
    print(classification_report(y, y_pred, target_names=class_names))
    
    # 6. Confusion matrix
    print("\\n6. CONFUSION MATRIX:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # 7. ROC-AUC analysis
    print("\\n7. ROC-AUC ANALYSIS:")
    try:
        overall_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr')
        print(f"   Overall ROC-AUC (OvR): {overall_auc:.3f}")
    except Exception as e:
        print(f"   Overall ROC-AUC failed: {e}")
    
    # 8. Noise application test (if applicable)
    if noise_augmentor is not None and intensity is not None:
        print(f"\\n8. NOISE APPLICATION TEST (intensity={intensity}):")
        
        # Apply noise
        X_noisy = noise_augmentor.transform(X)
        
        # Check signal changes
        noisy_power = np.mean(X_noisy**2)
        noisy_std = np.std(X_noisy)
        
        print(f"   Original power: {overall_power:.6f}")
        print(f"   Noisy power: {noisy_power:.6f}")
        print(f"   Power ratio: {noisy_power/overall_power:.3f}")
        
        if abs(noisy_power - overall_power) < 1e-10:
            print("   ⚠️  WARNING: Noise not applied - signal unchanged!")
        else:
            print("   ✅ Noise successfully applied")
        
        # Evaluate on noisy data
        y_pred_noisy = model.predict(X_noisy)
        noisy_accuracy = accuracy_score(y, y_pred_noisy)
        noisy_balanced_acc = balanced_accuracy_score(y, y_pred_noisy)
        
        print(f"   Noisy accuracy: {noisy_accuracy:.3f}")
        print(f"   Noisy balanced accuracy: {noisy_balanced_acc:.3f}")
        print(f"   Performance drop: {(accuracy - noisy_accuracy)/accuracy*100:.1f}%")
        
        if noisy_accuracy > 0.9:
            print("   ⚠️  WARNING: Still very high performance with noise!")
    
    # 9. Summary and recommendations
    print("\\n9. SUMMARY AND RECOMMENDATIONS:")
    
    if accuracy > 0.95:
        print("   ⚠️  WARNING: Accuracy > 95% - suspiciously high!")
    
    if abs(accuracy - balanced_acc) > 0.05:
        print("   ⚠️  WARNING: Large accuracy vs balanced accuracy difference!")
    
    if accuracy > 0.9 and noise_augmentor is not None:
        print("   ⚠️  WARNING: High performance even with noise - check noise application!")
    
    print("\\n   Recommendations:")
    print("   - Check if noise is actually being applied")
    print("   - Verify per-class performance is balanced")
    print("   - Compare with expected SSVEP baselines (70-90%)")
    print("   - Consider using balanced accuracy instead of raw accuracy")

# Usage example:
# comprehensive_debug(X, y, model, noise_augmentor, intensity=10.0, 
#                   class_names=['12.0Hz', '5.45Hz', '6.67Hz', '8.57Hz'])
'''
    
    print(complete_script)

def main():
    """Main debugging guide."""
    print("Lee2019 SSVEP Evaluation Debugging Guide")
    print("=" * 60)
    
    debug_noise_application()
    debug_per_class_performance()
    debug_evaluation_methodology()
    debug_ssvep_specific_issues()
    create_integrated_debug_script()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Add the debugging code to your evaluation pipeline")
    print("2. Run the comprehensive debug script")
    print("3. Check the output for warnings and issues")
    print("4. Focus on the most likely culprits:")
    print("   - Noise application verification")
    print("   - Per-class performance analysis")
    print("   - Comparison with expected baselines")
    print("\nThe issue is likely in noise application or evaluation methodology,")
    print("not in class imbalance (which is perfect).")

if __name__ == "__main__":
    main()
