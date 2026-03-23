"""
Training Pipeline for Model 2: Learning-Based Moral AI
======================================================
Handles data preparation, model training, and saving.
"""

import numpy as np
import os
from typing import Dict, List, Optional
from .features import FeatureExtractor
from .labels import get_all_labels
from .network import MoralNetwork


class MoralTrainer:
    """
    Trains the moral learning model on the AMR-220 dataset
    with human judgment labels.
    """
    
    def __init__(self, model_type: str = "neural_network"):
        self.feature_extractor = FeatureExtractor()
        self.model = MoralNetwork(model_type)
        self.model_type = model_type
        self.X_train = None
        self.y_train = None
        self.scenario_ids = None
        self.is_prepared = False
    
    def prepare_data(self, scenarios: List[Dict]) -> Dict:
        """Prepare training data from scenarios and human labels."""
        labels = get_all_labels()
        
        self.X_train, self.y_train, self.scenario_ids = \
            self.feature_extractor.extract_dataset(scenarios, labels)
        
        self.is_prepared = True
        
        n_positive = int(np.sum(self.y_train))
        n_negative = len(self.y_train) - n_positive
        
        return {
            "total_samples": len(self.X_train),
            "positive_samples": n_positive,
            "negative_samples": n_negative,
            "feature_size": self.X_train.shape[1],
            "unique_scenarios": len(set(self.scenario_ids)),
        }
    
    def train(self, epochs: int = 200, batch_size: int = 32,
              learning_rate: float = 0.01) -> Dict:
        """
        Train the model on prepared data.
        
        Returns:
            Training results dictionary
        """
        if not self.is_prepared:
            raise RuntimeError("Call prepare_data() first")
        
        if self.model_type == "neural_network":
            self.model.model.lr = learning_rate
        
        results = self.model.train(
            self.X_train, self.y_train,
            epochs=epochs, batch_size=batch_size
        )
        
        # Add feature importance
        importance = self.model.get_feature_importance(
            self.X_train, self.feature_extractor.feature_names
        )
        results["top_features"] = dict(list(importance.items())[:15])
        
        return results
    
    def get_training_history(self) -> Dict:
        """Get training history (for neural network)."""
        if self.model_type == "neural_network":
            return self.model.model.training_history
        return {}
    
    def save_model(self, directory: str):
        """Save trained model to directory."""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"model2_{self.model_type}.json")
        self.model.save(filepath)
        return filepath
    
    def load_model(self, directory: str):
        """Load trained model from directory."""
        filepath = os.path.join(directory, f"model2_{self.model_type}.json")
        if os.path.exists(filepath):
            self.model.load(filepath)
            return True
        return False
