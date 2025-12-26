"""
Quantum Support Vector Machine Implementation
Uses Qiskit's statevector kernel for quantum feature mapping
"""

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.svm import SVC
import numpy as np
import pickle
import os
import warnings
from tqdm import tqdm

# Try to import config, but provide defaults if not available
try:
    from config import QSVM_N_QUBITS, QSVM_SHOTS, QSVM_FEATURE_MAP_REPS
except ImportError:
    QSVM_N_QUBITS = 4
    QSVM_SHOTS = None
    QSVM_FEATURE_MAP_REPS = 2
    warnings.warn("Config not found, using default values")


class QuantumSVM:
    """
    Quantum Support Vector Machine using Qiskit
    
    This implementation uses a quantum kernel based on the ZZFeatureMap
    for encoding classical data into quantum states, combined with 
    classical SVM optimization.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit
        reps (int): Number of repetitions in the feature map
        feature_map: Quantum feature map circuit (ZZFeatureMap)
        quantum_kernel: Quantum kernel for computing similarity
        model (SVC): Scikit-learn SVM with quantum kernel
        is_trained (bool): Whether the model has been trained
    """

    def __init__(self, n_qubits=None, reps=None):
        """
        Initialize Quantum SVM

        Args:
            n_qubits (int, optional): Number of qubits. Defaults to QSVM_N_QUBITS.
            reps (int, optional): Feature map repetitions. Defaults to QSVM_FEATURE_MAP_REPS.
        """
        # Set parameters
        self.n_qubits = n_qubits if n_qubits is not None else QSVM_N_QUBITS
        self.reps = reps if reps is not None else QSVM_FEATURE_MAP_REPS

        # Validate parameters
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if self.reps < 1:
            raise ValueError("reps must be at least 1")

        # Initialize quantum components
        self.feature_map = None
        self.quantum_kernel = None
        self.model = None
        self.is_trained = False

        self._initialize_quantum_components()

        print(f"Initialized QuantumSVM with {self.n_qubits} qubits and {self.reps} repetitions")

    def _initialize_quantum_components(self):
        """Initialize quantum feature map and kernel"""

        # Create ZZ Feature Map
        # This map applies rotation gates and entangling gates
        self.feature_map = ZZFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            entanglement='full',  # Full entanglement between qubits
            insert_barriers=False  # No barriers for cleaner circuit
        )

        # Create statevector-based fidelity kernel
        # This computes |<φ(x)|φ(y)>|² between quantum states
        self.quantum_kernel = FidelityStatevectorKernel(
            feature_map=self.feature_map,
            enforce_psd=True  # Ensure positive semi-definite kernel matrix
        )

        # Create SVM with quantum kernel
        # We use the _quantum_kernel_function method as the kernel
        self.model = SVC(
            kernel=self._quantum_kernel_function,
            probability=True,  # Enable probability predictions
            class_weight='balanced',  # Handle imbalanced classes
            cache_size=1000  # Increase cache for faster training
        )

    def _quantum_kernel_function(self, X, Y):
        """
        Compute quantum kernel matrix between X and Y
        
        This is a separate method (not nested) so it can be pickled.
        
        Args:
            X (ndarray): First set of data points
            Y (ndarray): Second set of data points
            
        Returns:
            ndarray: Kernel matrix
        """
        # Ensure inputs are 2D numpy arrays
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        # Compute kernel matrix
        # Compute kernel matrix in batches
        n_samples = X.shape[0]
        
        # Use config batch size or default to 100
        try:
            from config import BATCH_QSVM
            batch_size = BATCH_QSVM
        except ImportError:
            batch_size = 100

        kernel_matrix = []
        
        # Display progress bar only if n_samples spans multiple batches
        disable_tqdm = n_samples <= batch_size
        
        desc = "Computing Quantum Kernel"
        # If we can infer what step this is (Train vs Predict)
        if hasattr(self, 'model') and hasattr(self.model, 'support_vectors_'):
             # If support vectors exist and Y matches them, checking dimensions might help,
             # but keeping it generic is safer.
             pass

        # Create iterator
        iterator = tqdm(range(0, n_samples, batch_size), desc=desc, disable=disable_tqdm)
        
        for i in iterator:
            X_batch = X[i:i + batch_size]
            # Compute similarity of this batch against ALL Y
            # FidelityStatevectorKernel.evaluate(x_vec, y_vec) returns matrix of shape (len(x_vec), len(y_vec))
            batch_kernel = self.quantum_kernel.evaluate(x_vec=X_batch, y_vec=Y)
            kernel_matrix.append(batch_kernel)
            
        # Concatenate all batch results
        return np.vstack(kernel_matrix)

    def train(self, X_train, y_train, verbose=True):
        """
        Train the Quantum SVM
        
        Args:
            X_train (ndarray): Training features [n_samples, n_features]
            y_train (ndarray): Training labels [n_samples]
            verbose (bool): Print training information
            
        Raises:
            ValueError: If feature dimension doesn't match n_qubits
        """
        # Validate input
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples. "
                f"Got {X_train.shape[0]} and {y_train.shape[0]}"
            )

        if X_train.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension ({X_train.shape[1]}) must match "
                f"n_qubits ({self.n_qubits}). "
                f"Apply PCA with n_components={self.n_qubits} first."
            )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Quantum SVM")
            print(f"{'='*60}")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features (qubits): {X_train.shape[1]}")
            print(f"Classes: {len(np.unique(y_train))}")
            print(f"Feature map: ZZFeatureMap with {self.reps} reps")
            print(f"Entanglement: full")
            print(f"Kernel: Statevector Fidelity")
            print(f"{'='*60}\n")

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        if verbose:
            print("✓ Training complete\n")

    def predict(self, X_test):
        """
        Predict labels for test data
        
        Args:
            X_test (ndarray): Test features [n_samples, n_features]
            
        Returns:
            ndarray: Predicted labels [n_samples]
            
        Raises:
            RuntimeError: If model not trained
            ValueError: If feature dimension doesn't match
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        # Validate input
        X_test = np.asarray(X_test)

        if X_test.ndim != 2:
            raise ValueError(f"X_test must be 2D, got shape {X_test.shape}")

        if X_test.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension ({X_test.shape[1]}) must match "
                f"n_qubits ({self.n_qubits})"
            )

        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Predict class probabilities
        
        Args:
            X_test (ndarray): Test features [n_samples, n_features]
            
        Returns:
            ndarray: Class probabilities [n_samples, n_classes]
            
        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        # Validate input
        X_test = np.asarray(X_test)

        if X_test.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension ({X_test.shape[1]}) must match "
                f"n_qubits ({self.n_qubits})"
            )

        return self.model.predict_proba(X_test)

    def score(self, X_test, y_test):
        """
        Compute accuracy score
        
        Args:
            X_test (ndarray): Test features
            y_test (ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before scoring")

        return self.model.score(X_test, y_test)

    def save(self, path):
        """
        Save model to disk
        
        Args:
            path (str): File path to save model
        """
        if not self.is_trained:
            warnings.warn("Saving untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save the entire object (Qiskit objects pickle well in recent versions)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print(f"✓ Model saved to: {path}")

    @classmethod
    def load(cls, path):
        """
        Load model from disk
        
        Args:
            path (str): File path to load model from
            
        Returns:
            QuantumSVM: Loaded model instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        print(f"✓ Model loaded from: {path}")
        return model

    def get_params(self):
        """
        Get model parameters
        
        Returns:
            dict: Model parameters
        """
        return {
            'n_qubits': self.n_qubits,
            'reps': self.reps,
            'feature_map': 'ZZFeatureMap',
            'entanglement': 'full',
            'kernel': 'FidelityStatevectorKernel',
            'is_trained': self.is_trained
        }

    def get_circuit_info(self):
        """
        Get quantum circuit information
        
        Returns:
            dict: Circuit statistics
        """
        return {
            'num_qubits': self.feature_map.num_qubits,
            'num_parameters': self.feature_map.num_parameters,
            'depth': self.feature_map.depth(),
            'size': self.feature_map.size(),
            'num_gates': sum(self.feature_map.count_ops().values())
        }

    def visualize_circuit(self, output_file=None):
        """
        Visualize the quantum feature map circuit
        
        Args:
            output_file (str, optional): Path to save the circuit diagram
            
        Returns:
            Figure: matplotlib figure object (if matplotlib is available)
        """
        try:
            from qiskit.visualization import circuit_drawer

            fig = circuit_drawer(
                self.feature_map,
                output='mpl',
                style={'backgroundcolor': '#FFFFFF'}
            )

            if output_file:
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✓ Circuit diagram saved to: {output_file}")

            return fig
        except ImportError:
            warnings.warn("matplotlib not available for circuit visualization")
            return None

    def __repr__(self):
        """String representation"""
        status = "trained" if self.is_trained else "untrained"
        return (
            f"QuantumSVM(n_qubits={self.n_qubits}, "
            f"reps={self.reps}, "
            f"status={status})"
        )

    def __str__(self):
        """Detailed string representation"""
        info = [
            "Quantum SVM",
            "-" * 40,
            f"Qubits: {self.n_qubits}",
            f"Feature Map: ZZFeatureMap (reps={self.reps})",
            f"Kernel: Statevector Fidelity",
            f"Status: {'Trained' if self.is_trained else 'Untrained'}",
            ]

        if self.is_trained:
            info.extend([
                f"Support Vectors: {len(self.model.support_vectors_)}",
                f"Classes: {len(self.model.classes_)}"
            ])

        return "\n".join(info)


# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("Generating sample data...")
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Preprocess: Scale and reduce dimensions
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Reduced features from {X.shape[1]} to {X_train_pca.shape[1]}")

    # Create and train quantum SVM
    qsvm = QuantumSVM(n_qubits=4, reps=2)
    print(qsvm)

    qsvm.train(X_train_pca, y_train)

    # Make predictions
    y_pred = qsvm.predict(X_test_pca)
    accuracy = qsvm.score(X_test_pca, y_test)

    print(f"\nTest Accuracy: {accuracy:.2%}")

    # Get probabilities
    y_proba = qsvm.predict_proba(X_test_pca)
    print(f"Prediction probabilities shape: {y_proba.shape}")

    # Show circuit info
    print("\nCircuit Information:")
    for key, value in qsvm.get_circuit_info().items():
        print(f"  {key}: {value}")

    # Save and load
    qsvm.save("qsvm_model.pkl")
    loaded_qsvm = QuantumSVM.load("qsvm_model.pkl")
    print(f"\nLoaded model: {loaded_qsvm}")