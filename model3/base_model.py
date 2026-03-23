"""
Base Moral Model for Model 3: RLHF Moral AI
============================================
The initial policy network that generates moral decisions.
Starts imperfect, then improves through human feedback.
"""

import numpy as np
from typing import Dict, List, Tuple
import os, json

# Reuse feature extraction from Model 2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model2.features import FeatureExtractor, CONSEQUENCE_KEYS


class BaseMoralModel:
    """
    Base policy network for RLHF training.
    
    Architecture: Input(55) -> Dense(96,ReLU) -> Dense(48,ReLU) -> Dense(1,Sigmoid)
    
    This model starts with random/noisy weights and is refined
    through the RLHF loop to align with human preferences.
    """
    
    def __init__(self, input_size: int = 55, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.feature_extractor = FeatureExtractor()
        
        # Smaller architecture than Model 2 (policy network)
        self.layer_sizes = [input_size, 96, 48, 1]
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * \
                np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.n_layers = len(self.weights)
        self.is_trained = False
        self.training_history = {
            "reward": [], "loss": [], "accuracy": [],
            "sycophancy_score": [], "epoch": []
        }
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """Forward pass, returns output and activations for backprop."""
        activations = [X]
        current = X
        
        for i in range(self.n_layers - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self._relu(z)
            activations.append(current)
        
        z = current @ self.weights[-1] + self.biases[-1]
        output = self._sigmoid(z)
        activations.append(output)
        
        return output, activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get preference scores for actions."""
        output, _ = self.forward(X)
        return output.flatten()
    
    def generate_responses(self, scenario: Dict) -> List[Dict]:
        """
        Generate scored responses for each action in a scenario.
        Returns list of {action_id, description, score, features}.
        """
        actions = scenario.get("actions", [])
        responses = []
        
        for idx, action in enumerate(actions):
            feat = self.feature_extractor.extract_action_features(scenario, action, idx)
            score = float(self.predict(feat.reshape(1, -1))[0])
            responses.append({
                "action_id": action["id"],
                "action_idx": idx,
                "description": action["description"],
                "score": score,
                "features": feat,
            })
        
        return responses
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get all weights and biases as flat arrays (for RL update)."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.copy())
            params.append(b.copy())
        return params
    
    def set_parameters(self, params: List[np.ndarray]):
        """Set all weights and biases from flat arrays."""
        idx = 0
        for i in range(self.n_layers):
            self.weights[i] = params[idx].copy()
            self.biases[i] = params[idx + 1].copy()
            idx += 2
    
    def save(self, filepath: str):
        data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "history": self.training_history,
            "is_trained": self.is_trained,
            "layer_sizes": self.layer_sizes,
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
