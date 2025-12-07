import os
import sys
import json
import warnings
from datetime import datetime

from src.data_processing import load_data, preprocess_data, get_dataset_info
from src.svm import ClassicalSVM
from src.trainer import train_and_evaluate, compare_models, save_model
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
        print(f"\nDataset Information:")
        print(f"  Total samples:     {info['n_samples']}")
        print(f"  Features:          {info['n_features']}")
        print(f"  Classes:           {info['n_classes']}")
        print(f"  Class distribution: {info['class_distribution']}")

    except Exception as e:
        print(f"\nERROR: Failed to load data: {str(e)}")
        sys.exit(1)

    # ========================================
    # STEP 2: Preprocess Data
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING DATA")
    print("="*70)

    try:
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        print(f"\nPreprocessing completed successfully!")

    except Exception as e:
        print(f"\nERROR: Failed to preprocess data: {str(e)}")
        sys.exit(1)

    # ========================================
    # STEP 3: Train Classical SVM
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: CLASSICAL SVM")
    print("="*70)

    try:
        # Initialize Classical SVM
        classical_svm = ClassicalSVM()
        print(f"\nModel initialized: {classical_svm}")

        # Train and evaluate
        svm_results = train_and_evaluate(
            model=classical_svm,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name='Classical SVM'
        )

        # Print results
        if VERBOSE:
            print_results(svm_results, detailed=True)

    except Exception as e:
        print(f"\nERROR: Classical SVM failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================
    # STEP 4: Train Quantum SVM
    # ========================================
    if not SKIP_QUANTUM:
        print("\n" + "="*70)
        print("STEP 4: QUANTUM SVM")
        print("="*70)

        try:
            # Initialize Quantum SVM
            quantum_svm = QuantumSVM()
            print(f"\nModel initialized: {quantum_svm}")

            # Show circuit info
            circuit_info = quantum_svm.get_circuit_info()
            print(f"\nQuantum Circuit Information:")
            print(f"  Qubits:     {circuit_info['num_qubits']}")
            print(f"  Parameters: {circuit_info['num_parameters']}")
            print(f"  Depth:      {circuit_info['depth']}")
            print(f"  Size:       {circuit_info['size']} gates")

            # Train and evaluate
            qsvm_results = train_and_evaluate(
                model=quantum_svm,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name='Quantum SVM'
            )

            # Print results
            if VERBOSE:
                print_results(qsvm_results, detailed=True)

        except Exception as e:
            print(f"\nERROR: Quantum SVM failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nNote: Quantum SVM requires Qiskit to be properly installed.")
            print("Continuing with Classical SVM results only...")
            qsvm_results = None
    else:
        print("\n[SKIPPED] Quantum SVM training skipped (SKIP_QUANTUM=True in config)")
        qsvm_results = None

    # ========================================
    # STEP 5: Compare Results
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: COMPARISON & RESULTS")
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
    # STEP 6: Save Results
    # ========================================
    print("\n" + "="*70)
    print("STEP 6: SAVING RESULTS")
    print("="*70)

    try:
        # Create results directory
        os.makedirs(METRICS_DIR, exist_ok=True)

        # Prepare results JSON
        results_json = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset': DATASET_NAME,
                'random_seed': RANDOM_SEED,
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'n_features_original': info['n_features'],
                'n_features_final': X_train.shape[1],
            },
            'classical_svm': svm_results,
            'quantum_svm': qsvm_results if qsvm_results else None,
            'comparison': comparison
        }

        # Save to JSON
        results_file = os.path.join(METRICS_DIR, 'experiment_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_json, f, indent=4)

        print(f"\nResults saved to: {results_file}")

        # Save models if requested
        if SAVE_MODELS:
            print("\nSaving trained models...")

            svm_path = save_model(classical_svm, 'classical_svm')
            print(f"  Classical SVM: {svm_path}")

            if qsvm_results is not None:
                qsvm_path = save_model(quantum_svm, 'quantum_svm')
                print(f"  Quantum SVM:   {qsvm_path}")

    except Exception as e:
        print(f"\nWARNING: Failed to save results: {str(e)}")

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)

    print("\nFinal Summary:")
    print(f"  Classical SVM Accuracy: {svm_results['metrics']['accuracy']:.4f}")
    if qsvm_results:
        print(f"  Quantum SVM Accuracy:   {qsvm_results['metrics']['accuracy']:.4f}")
        accuracy_diff = qsvm_results['metrics']['accuracy'] - svm_results['metrics']['accuracy']
        print(f"  Accuracy Difference:    {accuracy_diff:+.4f}")

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
