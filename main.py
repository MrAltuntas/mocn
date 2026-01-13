import os
import sys
import json
import warnings
from datetime import datetime

from src.data_processing import load_data, get_dataset_info
from src.svm import ClassicalSVM
from src.trainer import cross_validate_experiment, compare_models, save_model
from src.evaluation import print_results
from config import *
from src.qsvm import QuantumSVM

# Suppress warnings
warnings.filterwarnings('ignore')


def main():
    # ========================================
    # STEP 1: Load Data
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    try:
        X, y = load_data(DATASET_NAME)
        print(f"\nDataset loaded successfully!")

        # Show dataset info
        info = get_dataset_info(X, y)
        print(f"\nDataset Information (Population for CV):")
        print(f"  Total samples:     {info['n_samples']}")
        print(f"  Features (Raw):    {info['n_features']}")
        print(f"  Classes:           {info['n_classes']}")
        print(f"  Class distribution: {info['class_distribution']}")

    except Exception as e:
        print(f"\nERROR: Failed to load data: {str(e)}")
        sys.exit(1)

    # ========================================
    # STEP 2: Cross-Validation (Classical SVM)
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: CLASSICAL SVM (Cross-Validation)")
    print("="*70)

    try:
        # Define Factory
        def create_classical_svm():
            return ClassicalSVM()

        # Run CV
        svm_results = cross_validate_experiment(
            model_factory=create_classical_svm,
            X=X,
            y=y,
            model_name='Classical SVM'
        )

    except Exception as e:
        print(f"\nERROR: Classical SVM failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================
    # STEP 3: Cross-Validation (Quantum SVM)
    # ========================================
    if not SKIP_QUANTUM:
        print("\n" + "="*70)
        print("STEP 3: QUANTUM SVM (Cross-Validation)")
        print("="*70)

        try:
            # Define Factory
            def create_quantum_svm():
                return QuantumSVM()
            
            # Show circuit info (just once for info)
            dummy_qsvm = QuantumSVM()
            circuit_info = dummy_qsvm.get_circuit_info()
            print(f"\nQuantum Circuit Information:")
            print(f"  Qubits:     {circuit_info['num_qubits']}")
            print(f"  Parameters: {circuit_info['num_parameters']}")
            print(f"  Depth:      {circuit_info['depth']}")
            
            # Run CV
            qsvm_results = cross_validate_experiment(
                model_factory=create_quantum_svm,
                X=X,
                y=y,
                model_name='Quantum SVM'
            )

        except Exception as e:
            print(f"\nERROR: Quantum SVM failed: {str(e)}")
            import traceback
            traceback.print_exc()
            qsvm_results = None
    else:
        print("\n[SKIPPED] Quantum SVM training skipped (SKIP_QUANTUM=True in config)")
        qsvm_results = None

    # ========================================
    # STEP 4: Compare Results
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: COMPARISON & RESULTS")
    print("="*70)

    # Prepare results list
    all_results = [svm_results]
    if qsvm_results is not None:
        all_results.append(qsvm_results)

    # Compare models
    if len(all_results) > 1:
        comparison = compare_models(all_results)
    else:
        comparison = None
        print("\nOnly one model was trained. Skipping comparison.")

    # ========================================
    # STEP 5: Save Results
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: SAVING RESULTS")
    print("="*70)

    try:
        # Create results directory
        os.makedirs(METRICS_DIR, exist_ok=True)

        # Prepare results JSON
        results_json = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset': DATASET_NAME,
                'cv_folds': CV_FOLDS,
                'total_samples': len(X),
                'normalization': NORMALIZATION_METHOD,
                'use_pca': USE_PCA,
            },
            'classical_svm': svm_results,
            'quantum_svm': qsvm_results if qsvm_results else None,
            'comparison': comparison
        }

        # Save to JSON
        results_file = os.path.join(METRICS_DIR, 'experiment_results_cv.json')
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=4)

        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"\nWARNING: Failed to save results: {str(e)}")

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)

    print("\nFinal Summary (Average Accuracy):")
    print(f"  Classical SVM: {svm_results['metrics']['accuracy']:.4f} (+/- {svm_results['metrics']['accuracy_std']:.4f})")
    if qsvm_results:
        print(f"  Quantum SVM:   {qsvm_results['metrics']['accuracy']:.4f} (+/- {qsvm_results['metrics']['accuracy_std']:.4f})")
        accuracy_diff = qsvm_results['metrics']['accuracy'] - svm_results['metrics']['accuracy']
        print(f"  Difference:    {accuracy_diff:+.4f}")

    print(f"\nResults saved to: {METRICS_DIR}")
    print(f"Experiment finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
