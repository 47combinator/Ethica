"""
Adversarial Executor for Model 5
=================================
Runs all 4 models on both normal and adversarial scenarios.
Collects decisions for comparison.
"""

import sys, os
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class ModelResult:
    """Result of running a model on a scenario."""
    model_name: str
    scenario_id: str
    chosen_action: str
    confidence: float
    is_adversarial: bool
    attack_id: str = ""
    attack_type: str = ""
    attack_severity: float = 0.0


class AdversarialExecutor:
    """
    Runs target models on normal and adversarial scenarios.
    Collects results for failure analysis.
    """

    def __init__(self):
        self.normal_results: Dict[str, List[ModelResult]] = {}
        self.adversarial_results: Dict[str, List[ModelResult]] = {}

    def run_model1(self, scenarios: List[Dict],
                    is_adversarial: bool = False) -> List[ModelResult]:
        from core.rules import EthicalRuleSystem
        from core.engine import MoralDecisionEngine
        rs = EthicalRuleSystem()
        eng = MoralDecisionEngine(rs)
        results = []
        for sc in scenarios:
            d = eng.evaluate_scenario(sc)
            results.append(ModelResult(
                model_name="Model 1",
                scenario_id=sc.get("original_id", sc["id"]) if is_adversarial else sc["id"],
                chosen_action=d.chosen_action,
                confidence=d.confidence,
                is_adversarial=is_adversarial,
                attack_id=sc.get("attack_id", ""),
                attack_type=sc.get("attack_type", ""),
                attack_severity=sc.get("attack_severity", 0.0),
            ))
        return results

    def run_model2(self, scenarios: List[Dict],
                    is_adversarial: bool = False) -> List[ModelResult]:
        from model2.trainer import MoralTrainer
        from model2.predictor import MoralPredictor
        from data.scenarios import get_all_scenarios
        t = MoralTrainer("neural_network")
        t.prepare_data(get_all_scenarios())
        t.train(epochs=200)
        p = MoralPredictor(t.model, t.feature_extractor)
        results = []
        for sc in scenarios:
            d = p.predict_scenario(sc)
            results.append(ModelResult(
                model_name="Model 2",
                scenario_id=sc.get("original_id", sc["id"]) if is_adversarial else sc["id"],
                chosen_action=d.chosen_action,
                confidence=d.confidence,
                is_adversarial=is_adversarial,
                attack_id=sc.get("attack_id", ""),
                attack_type=sc.get("attack_type", ""),
                attack_severity=sc.get("attack_severity", 0.0),
            ))
        return results

    def run_model3(self, scenarios: List[Dict],
                    is_adversarial: bool = False) -> List[ModelResult]:
        from model3.base_model import BaseMoralModel
        from model3.feedback import HumanFeedbackSystem
        from model3.reward_model import RewardModel
        from model3.rl_optimizer import RLOptimizer
        from model3.predictor import RLHFPredictor
        from data.scenarios import get_all_scenarios
        bm = BaseMoralModel(); fb = HumanFeedbackSystem(); rm = RewardModel()
        opt = RLOptimizer(bm, rm, fb)
        opt.train_full_pipeline(get_all_scenarios(), reward_epochs=80, rl_iterations=12)
        p = RLHFPredictor(bm, rm)
        results = []
        for sc in scenarios:
            d = p.predict_scenario(sc)
            results.append(ModelResult(
                model_name="Model 3",
                scenario_id=sc.get("original_id", sc["id"]) if is_adversarial else sc["id"],
                chosen_action=d.chosen_action,
                confidence=d.confidence,
                is_adversarial=is_adversarial,
                attack_id=sc.get("attack_id", ""),
                attack_type=sc.get("attack_type", ""),
                attack_severity=sc.get("attack_severity", 0.0),
            ))
        return results

    def run_model4(self, scenarios: List[Dict],
                    is_adversarial: bool = False) -> List[ModelResult]:
        from model4.balancer import VirtueBalancer
        from model4.predictor import VirtuePredictor
        from data.scenarios import get_all_scenarios
        b = VirtueBalancer()
        b.train_from_judgments(get_all_scenarios())
        p = VirtuePredictor(b)
        results = []
        for sc in scenarios:
            d = p.predict_scenario(sc)
            results.append(ModelResult(
                model_name="Model 4",
                scenario_id=sc.get("original_id", sc["id"]) if is_adversarial else sc["id"],
                chosen_action=d.chosen_action,
                confidence=d.confidence,
                is_adversarial=is_adversarial,
                attack_id=sc.get("attack_id", ""),
                attack_type=sc.get("attack_type", ""),
                attack_severity=sc.get("attack_severity", 0.0),
            ))
        return results

    def run_all_models_normal(self, scenarios: List[Dict]) -> Dict[str, List[ModelResult]]:
        """Run all 4 models on normal scenarios."""
        self.normal_results = {
            "Model 1": self.run_model1(scenarios),
            "Model 2": self.run_model2(scenarios),
            "Model 3": self.run_model3(scenarios),
            "Model 4": self.run_model4(scenarios),
        }
        return self.normal_results

    def run_all_models_adversarial(self, adv_scenarios: List[Dict]) -> Dict[str, List[ModelResult]]:
        """Run all 4 models on adversarial scenarios."""
        self.adversarial_results = {
            "Model 1": self.run_model1(adv_scenarios, is_adversarial=True),
            "Model 2": self.run_model2(adv_scenarios, is_adversarial=True),
            "Model 3": self.run_model3(adv_scenarios, is_adversarial=True),
            "Model 4": self.run_model4(adv_scenarios, is_adversarial=True),
        }
        return self.adversarial_results
