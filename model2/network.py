"""
Neural Network & Decision Tree for Model 2: Learning-Based Moral AI
====================================================================
Implements both a neural network and decision tree classifier
that learn moral reasoning from human judgment data.
"""

import numpy as np
import os, json, pickle
from typing import Dict, List, Tuple, Optional


class SimpleNeuralNetwork:
    """
    A feedforward neural network implemented in pure numpy.
    Learns moral preferences from human judgment data.
    
    Architecture: Input(55) -> Dense(128,ReLU) -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)
    """
    
    def __init__(self, input_size: int = 55, hidden_sizes: List[int] = None,
                 learning_rate: float = 0.01, seed: int = 42):
        self.lr = learning_rate
        self.seed = seed
        np.random.seed(seed)
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.n_layers = len(self.weights)
        self.training_history = {"loss": [], "accuracy": []}
        self.is_trained = False
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass through the network."""
        activations = [X]
        current = X
        
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self._relu(z)
            activations.append(current)
        
        # Output layer with sigmoid
        z = current @ self.weights[-1] + self.biases[-1]
        output = self._sigmoid(z)
        activations.append(output)
        
        return output, activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict moral preference scores."""
        output, _ = self.forward(X)
        return output.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of being the preferred action."""
        return self.predict(X)
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, batch_size: int = 32,
              validation_split: float = 0.2) -> Dict:
        """
        Train the network using backpropagation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 1 for preferred, 0 for not
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dict
        """
        n = len(X)
        val_n = int(n * validation_split)
        
        # Shuffle and split
        indices = np.random.permutation(n)
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Mini-batch SGD
            perm = np.random.permutation(len(X_train))
            epoch_loss = 0
            n_batches = 0
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = perm[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward
                output, activations = self.forward(X_batch)
                
                # Loss (binary cross-entropy)
                eps = 1e-8
                loss = -np.mean(
                    y_batch * np.log(output + eps) +
                    (1 - y_batch) * np.log(1 - output + eps)
                )
                epoch_loss += loss
                n_batches += 1
                
                # Backward
                delta = output - y_batch  # gradient of BCE + sigmoid
                
                for i in range(self.n_layers - 1, -1, -1):
                    dw = activations[i].T @ delta / len(X_batch)
                    db = np.mean(delta, axis=0, keepdims=True)
                    
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * self._relu_deriv(activations[i])
                    
                    # Update with gradient clipping
                    dw = np.clip(dw, -1.0, 1.0)
                    self.weights[i] -= self.lr * dw
                    self.biases[i] -= self.lr * db
            
            avg_loss = epoch_loss / n_batches
            
            # Validation
            val_pred = self.predict(X_val)
            val_acc = np.mean((val_pred > 0.5).astype(float) == y_val.flatten())
            
            self.training_history["loss"].append(float(avg_loss))
            self.training_history["accuracy"].append(float(val_acc))
            
            # Early stopping
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        self.is_trained = True
        
        return {
            "final_loss": self.training_history["loss"][-1],
            "final_accuracy": self.training_history["accuracy"][-1],
            "epochs_trained": len(self.training_history["loss"]),
            "best_loss": best_val_loss,
        }
    
    def get_feature_importance(self, X: np.ndarray,
                                feature_names: List[str]) -> Dict[str, float]:
        """
        Estimate feature importance using gradient-based method.
        Computes average absolute gradient w.r.t. input features.
        """
        if not self.is_trained:
            return {}
        
        # Simple perturbation-based importance
        base_pred = self.predict(X)
        importances = {}
        
        for i, name in enumerate(feature_names):
            X_perturbed = X.copy()
            X_perturbed[:, i] += 0.1
            new_pred = self.predict(X_perturbed)
            importance = float(np.mean(np.abs(new_pred - base_pred)))
            importances[name] = importance
        
        # Normalize
        total = sum(importances.values()) or 1
        importances = {k: v/total for k, v in importances.items()}
        
        return dict(sorted(importances.items(), key=lambda x: -x[1]))
    
    def save(self, filepath: str):
        """Save model to file."""
        data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "history": self.training_history,
            "lr": self.lr,
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load model from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]
        self.training_history = data["history"]
        self.lr = data["lr"]
        self.is_trained = data["is_trained"]
        self.n_layers = len(self.weights)


class DecisionTreeMoral:
    """
    A simple decision tree for interpretable moral reasoning.
    Implemented from scratch for educational/research transparency.
    """
    
    def __init__(self, max_depth: int = 8, min_samples_split: int = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.is_trained = False
        self.feature_importances_ = None
    
    def _gini(self, y):
        if len(y) == 0:
            return 0
        p1 = np.mean(y)
        return 1 - p1**2 - (1-p1)**2
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feat = 0
        best_thresh = 0
        
        parent_gini = self._gini(y)
        n = len(y)
        
        # Sample features for efficiency
        n_features = X.shape[1]
        feat_indices = np.random.choice(n_features, min(20, n_features), replace=False)
        
        for feat in feat_indices:
            thresholds = np.percentile(X[:, feat], [25, 50, 75])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                
                if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
                    continue
                
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                gain = parent_gini - (n_left/n * left_gini + n_right/n * right_gini)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
        
        return best_feat, best_thresh, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n = len(y)
        
        # Stopping conditions
        if depth >= self.max_depth or n < self.min_samples_split or self._gini(y) < 0.01:
            return {"leaf": True, "prediction": float(np.mean(y)), "count": n}
        
        feat, thresh, gain = self._best_split(X, y)
        
        if gain <= 0:
            return {"leaf": True, "prediction": float(np.mean(y)), "count": n}
        
        left_mask = X[:, feat] <= thresh
        
        return {
            "leaf": False,
            "feature": int(feat),
            "threshold": float(thresh),
            "gain": float(gain),
            "left": self._build_tree(X[left_mask], y[left_mask], depth+1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth+1),
            "count": n,
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train the decision tree."""
        np.random.seed(42)
        self.tree = self._build_tree(X, y)
        self.is_trained = True
        
        # Compute feature importances
        self.feature_importances_ = np.zeros(X.shape[1])
        self._compute_importance(self.tree, X.shape[1])
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ /= total
        
        # Compute accuracy
        pred = self.predict(X)
        acc = np.mean((pred > 0.5).astype(float) == y)
        
        return {
            "final_accuracy": float(acc),
            "tree_depth": self._tree_depth(self.tree),
            "n_leaves": self._count_leaves(self.tree),
        }
    
    def _compute_importance(self, node, n_features):
        if node["leaf"]:
            return
        feat = node["feature"]
        self.feature_importances_[feat] += node.get("gain", 0) * node["count"]
        self._compute_importance(node["left"], n_features)
        self._compute_importance(node["right"], n_features)
    
    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["prediction"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)
    
    def _tree_depth(self, node):
        if node["leaf"]:
            return 0
        return 1 + max(self._tree_depth(node["left"]), self._tree_depth(node["right"]))
    
    def _count_leaves(self, node):
        if node["leaf"]:
            return 1
        return self._count_leaves(node["left"]) + self._count_leaves(node["right"])
    
    def get_feature_importance(self, X, feature_names):
        if self.feature_importances_ is None:
            return {}
        imp = {}
        for i, name in enumerate(feature_names):
            if i < len(self.feature_importances_):
                imp[name] = float(self.feature_importances_[i])
        return dict(sorted(imp.items(), key=lambda x: -x[1]))
    
    def get_decision_path(self, x: np.ndarray, feature_names: List[str]) -> List[str]:
        """Get human-readable decision path for a single sample."""
        path = []
        node = self.tree
        while not node["leaf"]:
            feat = node["feature"]
            thresh = node["threshold"]
            name = feature_names[feat] if feat < len(feature_names) else f"feature_{feat}"
            val = x[feat]
            if val <= thresh:
                path.append(f"{name} = {val:.3f} <= {thresh:.3f} -> LEFT")
                node = node["left"]
            else:
                path.append(f"{name} = {val:.3f} > {thresh:.3f} -> RIGHT")
                node = node["right"]
        path.append(f"Prediction: {node['prediction']:.3f}")
        return path
    
    def save(self, filepath: str):
        data = {"tree": self.tree, "importances": self.feature_importances_.tolist() if self.feature_importances_ is not None else None, "is_trained": self.is_trained}
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.tree = data["tree"]
        self.feature_importances_ = np.array(data["importances"]) if data["importances"] else None
        self.is_trained = data["is_trained"]


class MoralNetwork:
    """
    Wrapper that manages both neural network and decision tree models.
    Provides a unified interface for training and prediction.
    """
    
    def __init__(self, model_type: str = "neural_network"):
        self.model_type = model_type
        if model_type == "neural_network":
            self.model = SimpleNeuralNetwork()
        else:
            self.model = DecisionTreeMoral()
        self.training_results = None
    
    def train(self, X, y, **kwargs):
        self.training_results = self.model.train(X, y, **kwargs)
        return self.training_results
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    @property
    def is_trained(self):
        return self.model.is_trained
    
    def get_feature_importance(self, X, feature_names):
        return self.model.get_feature_importance(X, feature_names)
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model.load(filepath)
