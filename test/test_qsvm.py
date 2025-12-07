"""
Quick Test Script for Quantum SVM
Tests all major functionality
"""

import numpy as np
from src.qsvm import QuantumSVM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def test_qsvm():
    """Run comprehensive tests on QSVM implementation"""

    print("=" * 70)
    print("QUANTUM SVM - FUNCTIONALITY TEST")
    print("=" * 70)

    # Test 1: Initialization
    print("\n[TEST 1] Initialization...")
    try:
        qsvm = QuantumSVM(n_qubits=4, reps=2)
        print("✓ PASSED: Model initialized successfully")
        print(f"  {qsvm}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 2: Generate and preprocess data
    print("\n[TEST 2] Data Preparation...")
    try:
        X, y = make_classification(
            n_samples=80,
            n_features=8,
            n_informative=6,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=4)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print("✓ PASSED: Data prepared successfully")
        print(f"  Train: {X_train_pca.shape}, Test: {X_test_pca.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 3: Training
    print("\n[TEST 3] Model Training...")
    try:
        qsvm.train(X_train_pca, y_train, verbose=False)
        print("✓ PASSED: Model trained successfully")
        assert qsvm.is_trained == True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 4: Prediction
    print("\n[TEST 4] Predictions...")
    try:
        y_pred = qsvm.predict(X_test_pca)
        assert len(y_pred) == len(y_test)
        print("✓ PASSED: Predictions generated")
        print(f"  Predicted {len(y_pred)} samples")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 5: Probability predictions
    print("\n[TEST 5] Probability Predictions...")
    try:
        y_proba = qsvm.predict_proba(X_test_pca)
        assert y_proba.shape == (len(y_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)
        print("✓ PASSED: Probabilities computed correctly")
        print(f"  Shape: {y_proba.shape}, Sum check: OK")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 6: Scoring
    print("\n[TEST 6] Model Scoring...")
    try:
        accuracy = qsvm.score(X_test_pca, y_test)
        assert 0 <= accuracy <= 1
        print("✓ PASSED: Scoring works")
        print(f"  Test Accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 7: Get parameters
    print("\n[TEST 7] Get Parameters...")
    try:
        params = qsvm.get_params()
        assert 'n_qubits' in params
        assert 'reps' in params
        print("✓ PASSED: Parameters retrieved")
        print(f"  Parameters: {params}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 8: Circuit info
    print("\n[TEST 8] Circuit Information...")
    try:
        circuit_info = qsvm.get_circuit_info()
        assert 'num_qubits' in circuit_info
        assert 'depth' in circuit_info
        print("✓ PASSED: Circuit info retrieved")
        print(f"  Qubits: {circuit_info['num_qubits']}, Depth: {circuit_info['depth']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 9: Save model
    print("\n[TEST 9] Save Model...")
    try:
        qsvm.save("test_model.pkl")
        print("✓ PASSED: Model saved")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 10: Load model
    print("\n[TEST 10] Load Model...")
    try:
        loaded_qsvm = QuantumSVM.load("test_model.pkl")
        assert loaded_qsvm.is_trained == True
        assert loaded_qsvm.n_qubits == 4
        print("✓ PASSED: Model loaded successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 11: Predictions with loaded model
    print("\n[TEST 11] Loaded Model Predictions...")
    try:
        y_pred_loaded = loaded_qsvm.predict(X_test_pca)
        assert np.array_equal(y_pred, y_pred_loaded)
        print("✓ PASSED: Loaded model produces same predictions")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

    # Test 12: Error handling - wrong dimensions
    print("\n[TEST 12] Error Handling...")
    try:
        wrong_X = np.random.randn(10, 5)  # Wrong number of features
        try:
            qsvm.predict(wrong_X)
            print("✗ FAILED: Should have raised ValueError")
            return False
        except ValueError:
            print("✓ PASSED: Proper error handling for wrong dimensions")
    except Exception as e:
        print(f"✗ FAILED: Unexpected error: {e}")
        return False

    # Test 13: Untrained model error
    print("\n[TEST 13] Untrained Model Protection...")
    try:
        untrained_qsvm = QuantumSVM(n_qubits=4, reps=2)
        try:
            untrained_qsvm.predict(X_test_pca)
            print("✗ FAILED: Should have raised RuntimeError")
            return False
        except RuntimeError:
            print("✓ PASSED: Prevents prediction on untrained model")
    except Exception as e:
        print(f"✗ FAILED: Unexpected error: {e}")
        return False

    # Cleanup
    import os
    try:
        os.remove("test_model.pkl")
        print("\n✓ Cleanup complete")
    except:
        pass

    return True


def main():
    """Run all tests"""
    print("\n" + "🔬 " * 35)
    print("STARTING COMPREHENSIVE TEST SUITE")
    print("🔬 " * 35 + "\n")

    success = test_qsvm()

    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 70)
        print("\nYour Quantum SVM implementation is working correctly!")
        print("\nNext steps:")
        print("  1. Run 'python example_usage.py' for a complete workflow")
        print("  2. Check README.md for detailed documentation")
        print("  3. Customize config.py for your use case")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease review the errors above and check your setup.")

    print()


if __name__ == "__main__":
    main()