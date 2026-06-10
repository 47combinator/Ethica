# ═══════════════════════════════════════════════════════════════════════
# Copyright (c) 2025 Pratyush Chaudhari. All rights reserved.
#
# This source code is part of the Ethica project.
# Research paper: https://zenodo.org/records/20544025
#
# LICENSE: This code is provided for academic study and personal
# learning ONLY. Commercial use, corporate deployment, or any use
# intended to generate revenue is strictly prohibited without
# explicit written permission from the author.
# ═══════════════════════════════════════════════════════════════════════

# Model 5: Adversarial Moral Robustness AI
from .attacks import AttackLibrary, Attack, AttackType
from .generator import AdversarialGenerator
from .executor import AdversarialExecutor
from .detector import FailureDetector
from .scorer import RobustnessScorer
