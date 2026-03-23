"""
Reward Model for Model 3: RLHF Moral AI
========================================
Learns to predict human preferences from pairwise feedback.
"""

import numpy as np
from typing import Dict, List, Tuple
import os, json


class RewardModel:
    """
    Neural network that learns to predict human preference scores.
    
    Trained on pairwise comparisons: given two responses,
    learns which one humans prefer and by how much.
    
    Architecture: Input(55) -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Linear)
    Output is an unbounded reward score.
    """
    
    def __init__(self, input_size: int = 55, learning_rate: float = 0.005, seed: int = 42):
        self.lr = learning_rate
        np.random.seed(seed)
        
        layers = [input_size, 64, 32, 1]
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.n_layers = len(self.weights)
        self.is_trained = False
        self.training_history = {"loss": [], "accuracy": []}
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass returning raw reward score."""
        activations = [X]
        current = X
        
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self._relu(z)
            activations.append(current)
        
        # Linear output (unbounded reward)
        z = current @ self.weights[-1] + self.biases[-1]
        activations.append(z)
        
        return z, activations
    
    def predict_reward(self, X: np.ndarray) -> np.ndarray:
        """Predict reward score for given features."""
        output, _ = self.forward(X)
        return output.flatten()
    
    def train_on_pairs(self, pairs: List[Dict], epochs: int = 100,
                        batch_size: int = 16) -> Dict:
        """
        Train reward model using pairwise preference data.
        
        Uses the Bradley-Terry loss:
        L = -log(sigmoid(r_better - r_worse))
        
        This teaches the model to assign higher rewards to preferred actions.
        """
        if not pairs:
            return {"error": "No training pairs"}
        
        # Extract feature pairs
        X_better = np.array([p["better_features"] for p in pairs])
        X_worse = np.array([p["worse_features"] for p in pairs])
        
        n = len(pairs)
        best_loss = float('inf')
        
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0
            epoch_correct = 0
            n_batches = 0
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                bs = len(idx)
                
                # Forward pass for both
                r_better, acts_better = self.forward(X_better[idx])
                r_worse, acts_worse = self.forward(X_worse[idx])
                
                # Bradley-Terry: P(better > worse) = sigmoid(r_better - r_worse)
                diff = r_better - r_worse
                prob = self._sigmoid(diff)
                
                # Loss = -log(prob)
                eps = 1e-8
                loss = -np.mean(np.log(prob + eps))
                epoch_loss += loss
                n_batches += 1
                
                # Accuracy: how often r_better > r_worse
                epoch_correct += np.sum(diff > 0)
                
                # Backward pass (gradient of Bradley-Terry loss)
                # d_loss/d_diff = -(1 - sigmoid(diff)) = sigmoid(diff) - 1
                d_diff = (prob - 1.0) / bs
                
                # Backprop through better branch (positive gradient)
                self._backprop(acts_better, d_diff)
                
                # Backprop through worse branch (negative gradient)
                self._backprop(acts_worse, -d_diff)
            
            avg_loss = epoch_loss / max(n_batches, 1)
            accuracy = epoch_correct / n
            
            self.training_history["loss"].append(float(avg_loss))
            self.training_history["accuracy"].append(float(accuracy))
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        self.is_trained = True
        
        return {
            "final_loss": self.training_history["loss"][-1],
            "final_accuracy": self.training_history["accuracy"][-1],
            "epochs_trained": epochs,
            "total_pairs": n,
        }
    
    def _backprop(self, activations: List, d_output: np.ndarray):
        """Backpropagate gradients through the network."""
        delta = d_output
        
        for i in range(self.n_layers - 1, -1, -1):
            dw = activations[i].T @ delta
            db = np.mean(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_deriv(activations[i])
            
            dw = np.clip(dw, -1.0, 1.0)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db
    
    def save(self, filepath: str):
        data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "history": self.training_history,
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]
        self.training_history = data["history"]
        self.is_trained = data["is_trained"]
        self.n_layers = len(self.weights)
