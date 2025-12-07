import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef
)
import warnings


def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Compute comprehensive evaluation metrics

    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels
        y_proba (ndarray, optional): Predicted probabilities for ROC-AUC

    Returns:
        metrics (dict): Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Weighted precision
            - recall: Weighted recall (sensitivity)
            - f1: Weighted F1 score
            - fpr: False Positive Rate
            - fnr: False Negative Rate
            - tnr: True Negative Rate (specificity)
            - tpr: True Positive Rate (sensitivity)
            - confusion_matrix: Confusion matrix
            - mcc: Matthews Correlation Coefficient
            - roc_auc: ROC-AUC score (if probabilities provided)
    """
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization

    # Compute FPR, FNR, TPR, TNR from confusion matrix
    # Handle both binary and multi-class cases
    if cm.shape == (2, 2):
        # Binary classification
        tn, fp, fn, tp = cm.ravel()

        # False Positive Rate (FPR) = FP / (FP + TN)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False Negative Rate (FNR) = FN / (FN + TP)
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # True Negative Rate (TNR) = Specificity = TN / (TN + FP)
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # True Positive Rate (TPR) = Sensitivity = TP / (TP + FN)
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0

    else:
        # Multi-class: compute average FPR and FNR
        n_classes = cm.shape[0]
        fpr_list = []
        fnr_list = []

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            fpr_list.append(fpr)
            fnr_list.append(fnr)

        metrics['fpr'] = np.mean(fpr_list)
        metrics['fnr'] = np.mean(fnr_list)
        metrics['tnr'] = 1 - metrics['fpr']
        metrics['tpr'] = 1 - metrics['fnr']

    # Matthews Correlation Coefficient (MCC)
    # Good for imbalanced datasets
    try:
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics['mcc'] = 0.0

    # ROC-AUC (if probabilities are provided)
    if y_proba is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            # For multi-class
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception as e:
            warnings.warn(f"Could not compute ROC-AUC: {str(e)}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    return metrics


def print_results(results, detailed=True):
    """
    Print evaluation results in a formatted way

    Args:
        results (dict): Results dictionary from train_and_evaluate
        detailed (bool): Whether to print detailed metrics
    """
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS: {results['model_name']}")
    print("="*60)

    metrics = results['metrics']

    # Main metrics
    print("\nMain Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    if detailed:
        # Additional metrics
        print("\nDetailed Metrics:")
        print(f"  FPR (False Positive Rate): {metrics['fpr']:.4f}")
        print(f"  FNR (False Negative Rate): {metrics['fnr']:.4f}")
        print(f"  TPR (True Positive Rate):  {metrics['tpr']:.4f}")
        print(f"  TNR (True Negative Rate):  {metrics['tnr']:.4f}")
        print(f"  MCC (Matthews Corr. Coef): {metrics['mcc']:.4f}")

        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC:                   {metrics['roc_auc']:.4f}")

    # Performance metrics
    print("\nPerformance:")
    print(f"  Training Time:   {results['training_time']:.2f} seconds")
    print(f"  Inference Time:  {results['inference_time']:.4f} seconds")

    if detailed:
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print_confusion_matrix(cm)

        # Model parameters
        if results.get('model_params'):
            print("\nModel Parameters:")
            for key, value in results['model_params'].items():
                print(f"  {key}: {value}")

    print("="*60)


def print_confusion_matrix(cm):
    """
    Print confusion matrix in a nice format

    Args:
        cm (ndarray): Confusion matrix
    """
    n_classes = cm.shape[0]

    # Header
    print("              ", end="")
    for i in range(n_classes):
        print(f"Pred {i:2d}  ", end="")
    print()

    # Matrix
    for i in range(n_classes):
        print(f"  True {i:2d}:   ", end="")
        for j in range(n_classes):
            print(f"{cm[i, j]:6d}  ", end="")
        print()


def generate_classification_report(y_true, y_pred, class_names=None):
    """
    Generate detailed classification report

    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels
        class_names (list): Names of classes

    Returns:
        report (str): Classification report string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    return report


def calculate_detection_rate(y_true, y_pred):
    """
    Calculate attack detection rate (for IDS)
    Assumes label 1 = attack, 0 = normal

    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels

    Returns:
        detection_rate (float): Percentage of attacks detected
    """
    # Find attack samples (label = 1)
    attack_mask = y_true == 1

    if attack_mask.sum() == 0:
        return 0.0

    # Calculate how many attacks were correctly detected
    detected = (y_pred[attack_mask] == 1).sum()
    total_attacks = attack_mask.sum()

    detection_rate = detected / total_attacks

    return detection_rate


def calculate_false_alarm_rate(y_true, y_pred):
    """
    Calculate false alarm rate (for IDS)
    Assumes label 0 = normal, 1 = attack

    Args:
        y_true (ndarray): True labels
        y_pred (ndarray): Predicted labels

    Returns:
        false_alarm_rate (float): Percentage of normal samples misclassified as attacks
    """
    # Find normal samples (label = 0)
    normal_mask = y_true == 0

    if normal_mask.sum() == 0:
        return 0.0

    # Calculate how many normal samples were classified as attacks
    false_alarms = (y_pred[normal_mask] == 1).sum()
    total_normal = normal_mask.sum()

    false_alarm_rate = false_alarms / total_normal

    return false_alarm_rate
