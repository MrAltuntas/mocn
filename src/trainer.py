import time
import numpy as np
from src.evaluation import evaluate_model


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name='Model'):
    """
    Train a model and evaluate its performance

    Args:
        model: Model instance (ClassicalSVM or QuantumSVM)
        X_train (ndarray): Training features
        X_test (ndarray): Test features
        y_train (ndarray): Training labels
        y_test (ndarray): Test labels
        model_name (str): Name of the model for display

    Returns:
        results (dict): Dictionary containing:
            - model_name: Name of the model
            - training_time: Time taken to train (seconds)
            - inference_time: Time taken for predictions (seconds)
            - metrics: Dictionary of evaluation metrics
            - predictions: Predicted labels
            - model_params: Model parameters
    """
    print("\n" + "="*60)
    print(f"TRAINING: {model_name}")
    print("="*60)

    # Display training set info
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}")
    print(f"Features:         {X_train.shape[1]}")
    print(f"Classes:          {len(np.unique(y_train))}")

    # Training phase
    print(f"\nStarting training...")
    start_time = time.time()

    # Optimization: Subsample for Quantum SVM if dataset is too large
    try:
        from config import QSVM_MAX_SAMPLES
        from src.qsvm import QuantumSVM
        
        if isinstance(model, QuantumSVM) and X_train.shape[0] > QSVM_MAX_SAMPLES:
            print(f"\n[OPTIMIZATION] Subsampling training data for QSVM...")
            print(f"  Original: {X_train.shape[0]} samples")
            
            # Use stratified sampling to maintain class distribution
            from sklearn.model_selection import train_test_split
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train, y_train, 
                train_size=QSVM_MAX_SAMPLES, 
                stratify=y_train, 
                random_state=42
            )
            
            X_train = X_train_sub
            y_train = y_train_sub
            print(f"  Subsampled: {X_train.shape[0]} samples (stratified)")
            
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] Could not subsample for QSVM: {e}")

    start_time = time.time()

    try:
        model.train(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        raise

    # Inference phase
    print(f"\nStarting inference...")
    start_time = time.time()

    try:
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.4f} seconds")

    except Exception as e:
        print(f"ERROR during inference: {str(e)}")
        raise

    # Get probability scores (for ROC/PR curves)
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        # Model doesn't support predict_proba (e.g., QuantumSVM)
        y_proba = None

    # Evaluation phase
    print(f"\nEvaluating model...")
    metrics = evaluate_model(y_test, y_pred, y_proba=y_proba)

    # Quick results preview
    print(f"\n{'─'*60}")
    print(f"Quick Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"{'─'*60}")

    # Get model parameters
    try:
        model_params = model.get_params()
    except AttributeError:
        model_params = {}

    # Compile results
    results = {
        'model_name': model_name,
        'training_time': training_time,
        'inference_time': inference_time,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_features': X_train.shape[1],
        'metrics': metrics,
        'predictions': y_pred.tolist(),  # Convert to list for JSON serialization
        'y_proba': y_proba.tolist() if y_proba is not None else None,  # Probability scores for ROC/PR curves
        'y_true': y_test.tolist(),  # True labels for curve plotting
        'model_params': model_params
    }

    return results


def compare_models(results_list):
    """
    Compare multiple model results

    Args:
        results_list (list): List of result dictionaries from train_and_evaluate

    Returns:
        comparison (dict): Comparison metrics
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    comparison = {}

    # Extract metrics for comparison
    for results in results_list:
        model_name = results['model_name']
        comparison[model_name] = {
            'accuracy': results['metrics']['accuracy'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1': results['metrics']['f1'],
            'training_time': results['training_time'],
            'inference_time': results['inference_time']
        }

    # Print comparison table
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Train Time':<15} {'Inference Time':<15}")
    print("─" * 80)

    for model_name, metrics in comparison.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['training_time']:<15.2f} "
              f"{metrics['inference_time']:<15.4f}")

    # Determine best model by accuracy
    best_model = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model (by accuracy): {best_model[0]} ({best_model[1]['accuracy']:.4f})")

    # Determine fastest model
    fastest_model = min(comparison.items(), key=lambda x: x[1]['training_time'])
    print(f"Fastest training: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}s)")

    return comparison


from sklearn.model_selection import StratifiedKFold
from src.data_processing import prepare_fold_data
from config import CV_FOLDS, RANDOM_SEED

def cross_validate_experiment(model_factory, X, y, model_name='Model'):
    """
    Perform K-Fold Cross Validation.
    
    Args:
        model_factory (callable): Function that returns a new model instance
        X (ndarray): Feature matrix (Raw, before specific preprocessing)
        y (ndarray): Labels
        model_name (str): Name for display
        
    Returns:
        results (dict): Aggregated results
    """
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION: {model_name}")
    print("="*60)
    
    # Initialize K-Fold
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_metrics = []
    training_times = []
    inference_times = []
    
    print(f"Starting {CV_FOLDS}-Fold Cross-Validation on {len(X)} samples...")
    print(f"{'Fold':<6} {'Accuracy':<10} {'F1':<10} {'Train Time':<12}")
    print("-" * 50)
    
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        # Split Raw Data
        X_train_raw, X_val_raw = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Prepare Data (Fit on Train, Transform val) - Prevents Leakage
        # Note: prepare_fold_data applies SMOTE, Scale, PCA
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = prepare_fold_data(
            X_train_raw, y_train, X_val_raw, y_val
        )
        
        # Instantiate fresh model
        model = model_factory()
        
        # Train
        t0 = time.time()
        model.train(X_train_fold, y_train_fold)
        t_train = time.time() - t0
        training_times.append(t_train)
        
        # Predict
        t0 = time.time()
        y_pred = model.predict(X_val_fold)
        t_infer = time.time() - t0
        inference_times.append(t_infer)
        
        # Evaluate
        try:
             y_proba = model.predict_proba(X_val_fold)
        except:
             y_proba = None
             
        metrics = evaluate_model(y_val_fold, y_pred, y_proba)
        fold_metrics.append(metrics)
        
        print(f"{fold_idx+1}/{CV_FOLDS:<4} {metrics['accuracy']:.4f}     {metrics['f1']:.4f}     {t_train:.2f}s")

    # Aggregate Results
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    
    avg_train_time = np.mean(training_times)
    
    print("-" * 50)
    print(f"Average Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"Average F1 Score: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Average Train Time: {avg_train_time:.2f}s")
    
    results = {
        'model_name': model_name,
        'metrics': {
            'accuracy': avg_accuracy,
            'accuracy_std': std_accuracy,
            'f1': avg_f1,
            'f1_std': std_f1,
            'precision': np.mean([m['precision'] for m in fold_metrics]),
            'recall': np.mean([m['recall'] for m in fold_metrics])
        },
        'training_time': np.sum(training_times), # Total time
        'avg_training_time': avg_train_time,
        'inference_time': np.sum(inference_times),
        'fold_metrics': fold_metrics
    }
    
    return results

def save_model(model, model_name, save_dir=None):
    """
    Save trained model to disk
    
    Args:
        model: Trained model instance
        model_name (str): Name for the saved model file
        save_dir (str): Directory to save model (default: from config)
    """
    from config import MODELS_DIR
    import os

    save_dir = save_dir or MODELS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Create filename
    filename = f"{model_name.lower().replace(' ', '_')}.pkl"
    filepath = os.path.join(save_dir, filename)

    # Save model
    model.save(filepath)

    return filepath
