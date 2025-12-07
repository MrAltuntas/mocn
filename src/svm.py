from sklearn.svm import SVC
from config import *
import pickle
import os


class ClassicalSVM:
    """
    Classical Support Vector Machine classifier using scikit-learn

    Attributes:
        model: SVC instance from scikit-learn
    """

    def __init__(self, kernel=None, C=None, gamma=None, verbose=None):
        """
        Initialize Classical SVM

        Args:
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            verbose (int or bool): Verbosity level (0/False=silent, 1/True=progress, 2+=detailed)
        """
        # Use config values if not provided
        kernel = kernel or SVM_KERNEL
        C = C or SVM_C
        gamma = gamma or SVM_GAMMA

        # Convert boolean to int if necessary, use config value if not provided
        if verbose is None:
            self.verbose = 1 if SVM_ITERATION_LOG else 0
        elif isinstance(verbose, bool):
            self.verbose = 1 if verbose else 0
        else:
            self.verbose = verbose
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=RANDOM_SEED,
            probability=True,  # Enable probability estimates
            class_weight='balanced',  # Handle class imbalance
            verbose=self.verbose  # Enable verbose output
        )

        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the SVM model

        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
        """
        if self.verbose > 0:
            print(f"\n[SVM Training] Starting training...")
            print(f"[SVM Training] Dataset size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"[SVM Training] Kernel: {self.model.kernel}, C: {self.model.C}, Gamma: {self.model.gamma}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        if self.verbose > 0:
            print(f"[SVM Training] Training completed!")
            if hasattr(self.model, 'n_support_'):
                print(f"[SVM Training] Support vectors: {sum(self.model.n_support_)}")

    def predict(self, X_test):
        """
        Make predictions

        Args:
            X_test (ndarray): Test features

        Returns:
            y_pred (ndarray): Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if self.verbose > 0:
            print(f"\n[SVM Prediction] Predicting {X_test.shape[0]} samples...")

        predictions = self.model.predict(X_test)

        if self.verbose > 0:
            print(f"[SVM Prediction] Prediction completed!")

        return predictions

    def predict_proba(self, X_test):
        """
        Predict class probabilities

        Args:
            X_test (ndarray): Test features

        Returns:
            probabilities (ndarray): Class probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        if self.verbose > 0:
            print(f"\n[SVM Prediction] Predicting probabilities for {X_test.shape[0]} samples...")

        probabilities = self.model.predict_proba(X_test)

        if self.verbose > 0:
            print(f"[SVM Prediction] Probability prediction completed!")

        return probabilities

    def save(self, path):
        """
        Save model to disk

        Args:
            path (str): File path to save model
        """
        if not self.is_trained:
            print("WARNING: Saving untrained model")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved to: {path}")

    def load(self, path):
        """
        Load model from disk

        Args:
            path (str): File path to load model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        print(f"Model loaded from: {path}")

    def get_params(self):
        """
        Get model parameters

        Returns:
            params (dict): Model parameters
        """
        return {
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma
        }

    def __str__(self):
        """String representation"""
        params = self.get_params()
        return f"ClassicalSVM(kernel={params['kernel']}, C={params['C']}, gamma={params['gamma']})"
