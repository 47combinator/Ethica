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

"""
AMR-1000 Dataset Expansion Pipeline
=====================================
Converts experimental datasets (Moral Machine + Scruples) into
the AMR scenario format used by the Ethica framework.

Usage:
    python -m expansion.convert_moral_machine   # Moral Machine -> AMR AV scenarios
    python -m expansion.convert_scruples        # Scruples -> AMR multi-category
    python -m expansion.validate                # Validate all scenarios
"""
