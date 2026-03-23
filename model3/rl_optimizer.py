"""
Reinforcement Learning Optimizer for Model 3: RLHF Moral AI
============================================================
Updates the base policy model using reward model scores.
Implements simplified Proximal Policy Optimization (PPO).
"""

import numpy as np
from typing import Dict, List
from .base_model import BaseMoralModel
from .reward_model import RewardModel
from .feedback import HumanFeedbackSystem
from model2.features import FeatureExtractor
from model2.labels import get_label


class RLOptimizer:
    """
    RLHF training loop that optimizes the base moral model
    using reward signals from the reward model.
    
    Pipeline per iteration:
    1. Base model generates decisions for all scenarios
    2. Reward model scores each decision
    3. Policy gradient updates push model toward higher-reward actions
    4. Track sycophancy (tendency to optimize for approval over correctness)
    """
    
    def __init__(self, base_model: BaseMoralModel, reward_model: RewardModel,
                 feedback_system: HumanFeedbackSystem,
                 learning_rate: float = 0.003, kl_penalty: float = 0.1):
        self.policy = base_model
        self.reward_model = reward_model
        self.feedback = feedback_system
        self.feature_extractor = FeatureExtractor()
        self.lr = learning_rate
        self.kl_penalty = kl_penalty  # Prevents policy from straying too far
        self.training_log = []
    
    def train_full_pipeline(self, scenarios: List[Dict],
                             reward_epochs: int = 80,
                             rl_iterations: int = 15,
                             feedback_noise: float = 0.1) -> Dict:
        """
        Full RLHF training pipeline.
        
        Phase 1: Collect initial feedback & train reward model
        Phase 2: RL optimization loop
        
        Returns:
            Complete training results
        """
        self.feedback.noise_level = feedback_noise
        
        # === PHASE 1: Train Reward Model ===
        # Collect human feedback on base model's outputs
        all_pairs, feedback_stats = self.feedback.collect_batch_feedback(
            scenarios, self.policy
        )
        
        # Train reward model on preferences
        reward_results = self.reward_model.train_on_pairs(
            all_pairs, epochs=reward_epochs
        )
        
        # === PHASE 2: RL Optimization ===
        rl_history = {
            "avg_reward": [], "accuracy": [],
            "sycophancy": [], "kl_divergence": []
        }
        
        # Save initial policy for KL computation
        initial_params = self.policy.get_parameters()
        
        for iteration in range(rl_iterations):
            # Generate decisions and compute rewards
            iter_rewards = []
            iter_correct = 0
            iter_total = 0
            sycophancy_signals = []
            
            for scenario in scenarios:
                actions = scenario.get("actions", [])
                sid = scenario["id"]
                label = get_label(sid)
                
                # Get features and predictions for all actions
                features_list = []
                for idx, action in enumerate(actions):
                    feat = self.feature_extractor.extract_action_features(
                        scenario, action, idx
                    )
                    features_list.append(feat)
                
                X = np.array(features_list)
                
                # Policy predictions
                policy_scores = self.policy.predict(X)
                chosen_idx = int(np.argmax(policy_scores))
                
                # Reward for chosen action
                reward = float(self.reward_model.predict_reward(
                    X[chosen_idx:chosen_idx+1]
                )[0])
                iter_rewards.append(reward)
                
                # Check accuracy against human preference
                if label:
                    human_pref = label["preferred_action_idx"]
                    if chosen_idx == human_pref:
                        iter_correct += 1
                    iter_total += 1
                    
                    # Sycophancy detection: does model agree with humans
                    # even when the reward says otherwise?
                    reward_ranking = self.reward_model.predict_reward(X)
                    reward_best = int(np.argmax(reward_ranking))
                    if reward_best != human_pref and chosen_idx == human_pref:
                        sycophancy_signals.append(1.0)
                    else:
                        sycophancy_signals.append(0.0)
                
                # === Policy Gradient Update ===
                # Simple REINFORCE: update weights to increase probability
                # of actions with high reward
                self._policy_gradient_step(X, policy_scores, chosen_idx, reward)
            
            # Compute KL divergence from initial policy
            kl = self._estimate_kl(scenarios, initial_params)
            
            avg_reward = float(np.mean(iter_rewards))
            accuracy = iter_correct / max(iter_total, 1)
            sycophancy = float(np.mean(sycophancy_signals)) if sycophancy_signals else 0
            
            rl_history["avg_reward"].append(avg_reward)
            rl_history["accuracy"].append(accuracy)
            rl_history["sycophancy"].append(sycophancy)
            rl_history["kl_divergence"].append(float(kl))
        
        self.policy.is_trained = True
        self.policy.training_history = {
            "reward": rl_history["avg_reward"],
            "accuracy": rl_history["accuracy"],
            "sycophancy_score": rl_history["sycophancy"],
            "loss": rl_history["kl_divergence"],
            "epoch": list(range(rl_iterations)),
        }
        
        return {
            "feedback_stats": feedback_stats,
            "reward_model_results": reward_results,
            "rl_iterations": rl_iterations,
            "final_avg_reward": rl_history["avg_reward"][-1] if rl_history["avg_reward"] else 0,
            "final_accuracy": rl_history["accuracy"][-1] if rl_history["accuracy"] else 0,
            "final_sycophancy": rl_history["sycophancy"][-1] if rl_history["sycophancy"] else 0,
            "final_kl": rl_history["kl_divergence"][-1] if rl_history["kl_divergence"] else 0,
            "rl_history": rl_history,
        }
    
    def _policy_gradient_step(self, X: np.ndarray, scores: np.ndarray,
                               chosen_idx: int, reward: float):
        """
        Simplified policy gradient step.
        Increases probability of chosen action proportional to reward.
        """
        # Forward pass for the chosen action
        chosen_x = X[chosen_idx:chosen_idx+1]
        output, activations = self.policy.forward(chosen_x)
        
        # Reward-weighted gradient
        # If reward > 0, push prediction higher; if < 0, push lower
        advantage = reward * 0.1  # scale advantage
        
        delta = np.array([[advantage]])
        
        for i in range(self.policy.n_layers - 1, -1, -1):
            dw = activations[i].T @ delta
            db = delta.copy()
            
            if i > 0:
                delta = (delta @ self.policy.weights[i].T) * \
                        (activations[i] > 0).astype(float)
            
            dw = np.clip(dw, -0.5, 0.5)
            self.policy.weights[i] += self.lr * dw
            self.policy.biases[i] += self.lr * db
    
    def _estimate_kl(self, scenarios: List[Dict],
                      initial_params: List[np.ndarray]) -> float:
        """Estimate KL divergence between current and initial policy."""
        # Save current params
        current_params = self.policy.get_parameters()
        
        # Sample a few scenarios
        sample = scenarios[:min(20, len(scenarios))]
        
        kl_total = 0
        count = 0
        
        for scenario in sample:
            actions = scenario.get("actions", [])
            features = []
            for idx, action in enumerate(actions):
                feat = self.feature_extractor.extract_action_features(
                    scenario, action, idx
                )
                features.append(feat)
            
            if not features:
                continue
            
            X = np.array(features)
            
            # Current policy
            current_scores = self.policy.predict(X)
            
            # Initial policy
            self.policy.set_parameters(initial_params)
            initial_scores = self.policy.predict(X)
            
            # Restore current
            self.policy.set_parameters(current_params)
            
            # Simplified KL: sum of squared differences in scores
            kl = float(np.mean((current_scores - initial_scores) ** 2))
            kl_total += kl
            count += 1
        
        return kl_total / max(count, 1)
