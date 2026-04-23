# Ethica

<p align="center">
  <strong>A comprehensive research system for studying machine ethics through 5 distinct moral AI models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Models-5-green" alt="Models">
  <img src="https://img.shields.io/badge/Scenarios-1020-orange" alt="Scenarios">
  <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License">
</p>

---

## Overview

This project implements **5 fundamentally different approaches to AI morality** and compares them against a dataset of **1020 ethical dilemmas (AMR-1020)** spanning 11 real-world categories. Each model represents a different philosophy of machine ethics — from rigid rule systems to adversarial stress testing.

**The core research question:**
> *How do different ethical frameworks shape AI moral reasoning — and where do they break?*

This is the same class of problems studied by **OpenAI**, **Anthropic**, and **Google DeepMind** in AI safety research.

---

## The 5 Models

| Model | Approach | Philosophy | Score |
|-------|----------|-----------|-------|
| **Model 1** | Rule-Based (Top-Down) | Deontological ethics — 22 hierarchical moral rules | 0.608 |
| **Model 2** | Learning-Based (Bottom-Up) | Empirical ethics — neural network trained on human judgments | 0.821 |
| **Model 3** | RLHF (Alignment) | Reinforcement Learning from Human Feedback with sycophancy detection | 0.717 |
| **Model 4** | Virtue Ethics (Hybrid) | Aristotelian virtues with context-adaptive phronesis | 0.855 |
| **Model 5** | Adversarial Robustness | Stress-tests Models 1–4 with 18 attacks across 7 categories | — |

### Decision Agreement Matrix (1020 scenarios)

|  | M1 | M2 | M3 | M4 |
|--|----|----|----|----|
| **M1** | 100% | 92.7% | 69.1% | 81.8% |
| **M2** | — | 100% | 71.4% | 85.5% |
| **M3** | — | — | 100% | 72.3% |
| **M4** | — | — | — | 100% |
| **All 4** | | | **60.9%** | |

---

## AMR-1020 Dataset

**1020 ethical dilemmas** across **11 real-world categories**:

| Category | Count | Examples |
|----------|-------|---------|
| Autonomous Vehicles | 30 | Trolley problems, self-sacrifice decisions |
| Healthcare AI | 30 | Triage, organ allocation, treatment bias |
| Hiring & Bias | 20 | Algorithmic discrimination, diversity quotas |
| Military AI | 20 | Drone targeting, civilian protection |
| Privacy & Surveillance | 20 | Data collection, facial recognition |
| Financial AI | 20 | Lending bias, algorithmic trading |
| Disaster Response | 20 | Resource allocation, rescue prioritization |
| Human-AI Interaction | 20 | Emotional manipulation, consent |
| Corporate Pressure | 20 | Whistleblowing, profit vs. safety |
| Moral Ambiguity | 20 | Edge cases, philosophical paradoxes |
| Education AI | 80+ | Academic integrity, generative AI policies |

*Note: The dataset incorporates the original 220 hand-crafted scenarios alongside 800 additional dilemmas converted from the Moral Machine and Scruples datasets.*

Each scenario includes:
- Textual description of the dilemma
- 2–3 possible actions with detailed consequences
- Consequence scores across 10+ ethical dimensions
- Human judgment labels (preferred action + distribution)

---

## Architecture

```
Ethica/
├── app.py                      # Unified Streamlit dashboard (all 5 models)
├── main.py                     # CLI runner for Model 1
├── requirements.txt            # Python dependencies
│
├── core/                       # Model 1: Rule-Based Ethics
│   ├── rules.py                # 22 hierarchical moral rules
│   ├── engine.py               # Moral decision engine with conflict resolution
│   ├── explanation.py          # Rule-based explanation generator
│   └── evaluation.py           # Consistency, fairness, harm metrics
│
├── model2/                     # Model 2: Learning-Based Ethics
│   ├── features.py             # 55-dimensional feature extractor
│   ├── labels.py               # Human moral judgments for all 220 scenarios
│   ├── network.py              # From-scratch neural network + decision tree
│   ├── trainer.py              # Training pipeline
│   ├── predictor.py            # Prediction with confidence and agreement
│   ├── explainer.py            # Learned-pattern explanations
│   └── evaluator.py            # Accuracy, bias, consistency metrics
│
├── model3/                     # Model 3: RLHF Ethics
│   ├── base_model.py           # Base policy network
│   ├── feedback.py             # Human feedback simulator (pairwise rankings)
│   ├── reward_model.py         # Bradley-Terry preference learning
│   ├── rl_optimizer.py         # Policy gradient with KL penalty
│   ├── predictor.py            # RLHF-aligned prediction
│   ├── explainer.py            # Sycophancy-aware explanations
│   └── evaluator.py            # Sycophancy detection, robustness metrics
│
├── model4/                     # Model 4: Virtue Ethics
│   ├── virtues.py              # 8 Aristotelian virtues with conflict detection
│   ├── context.py              # Phronesis: 10 context profiles + urgency modifiers
│   ├── balancer.py             # Virtue balancing engine with learned adjustments
│   ├── predictor.py            # Virtue-based decision generator
│   ├── explainer.py            # Virtue justification + conflict analysis
│   └── evaluator.py            # Balance (Shannon entropy), context sensitivity
│
├── model5/                     # Model 5: Adversarial Robustness
│   ├── attacks.py              # 18 attacks across 7 categories
│   ├── generator.py            # Adversarial scenario mutation (220 → 1100+)
│   ├── executor.py             # Runs all models on normal + adversarial data
│   ├── detector.py             # Moral failure classification
│   └── scorer.py               # Robustness scoring (5 sub-metrics)
│
├── neo4j_engine/               # Knowledge Graph Reasoning Engine
│   ├── schema.py               # Graph schema (5 node types, 6 relationships)
│   ├── connector.py            # Neo4j Aura cloud database connection
│   ├── queries.py              # Graph queries (scoring, path finding)
│   ├── reasoning.py            # Graph-based reasoning (standalone + hybrid)
│   ├── explanation.py          # Graph explanation generator
│   └── test_offline.py         # Full offline test suite
│
├── expansion/                  # Dataset Expansion Pipeline
│   ├── convert_moral_machine.py # Moral Machine CSV → AMR format
│   ├── convert_scruples.py     # Scruples JSONL → AMR format
│   ├── validate.py             # Dataset validator + merger
│   ├── generate_data_file.py   # Generates cat_expanded.py
│   └── verify_all.py           # 26-test verification suite
│
└── data/                       # AMR-1020 Dataset
    ├── scenarios.py            # Unified scenario loader
    ├── cat_vehicles.py         # Autonomous vehicle scenarios
    ├── cat_healthcare.py       # Healthcare AI scenarios
    ├── cat_hiring.py           # Hiring & bias scenarios
    ├── cat_military_privacy_finance.py  # Military, privacy, finance scenarios
    ├── cat_remaining.py        # Disaster, interaction, corporate, ambiguity
    └── cat_expanded.py         # 800 expanded scenarios (Moral Machine + Scruples)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/47combinator/Ethica.git
cd Ethica
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 4. Run from CLI (Model 1 only)

```bash
python main.py
```

### 5. Run verification tests (26 tests)

```bash
python expansion/verify_all.py
```

---

## Dashboard Features

### Model Selection
Choose any of the 5 models from the sidebar. Each model has its own themed interface:

- **Model 1** (🟢 Green) — Dashboard, Scenario Explorer, Full Evaluation, Rule System
- **Model 2** (🟣 Purple) — Dashboard with training curves, Scenario Explorer, Training Insights
- **Model 3** (🟡 Gold) — RLHF training progress, sycophancy tracking, reward analysis
- **Model 4** (🩷 Pink) — Virtue radar charts, context adaptation heatmap, conflict analysis
- **Model 5** (🔴 Red) — Attack Library, Stress Test, Failure Analysis, 5-Model Comparison

### Key Pages

| Page | Description |
|------|------------|
| **Scenario Explorer** | Pick any scenario, run a model, see the full decision + explanation |
| **Full Evaluation** | Run all 1020 scenarios, see accuracy, consistency, fairness metrics |
| **Stress Test** | Configure attacks (scenarios × attacks), see robustness rankings |
| **Failure Analysis** | Examine critical failures, severity distribution, model vulnerabilities |
| **5-Model Comparison** | Normal scores + adversarial robustness side-by-side |

---

## Model Details

### Model 1: Rule-Based (Deontological)
- **22 hierarchical rules** with priority levels (1–10) and weights
- Conflict resolution when rules disagree
- 100% transparent — every decision traces back to specific rules
- **Weakness**: Rigid — fails with context-dependent dilemmas

### Model 2: Learning-Based (Empirical)
- **55-dimensional feature vectors** per action (consequences + category + meta-features)
- Pure NumPy neural network: `Input(55) → Dense(128) → Dense(64) → Dense(32) → Dense(1)`
- Trained on human judgment labels for all 1020 scenarios
- **Weakness**: Inherits and may amplify biases from training data

### Model 3: RLHF (Alignment)
- **3-component pipeline**: Base Policy → Reward Model → RL Optimizer
- Bradley-Terry pairwise preference learning
- Policy gradient with KL divergence penalty
- Built-in **sycophancy detection** (optimizing for approval vs. correctness)
- **Weakness**: May learn to say what humans want to hear rather than what's right

### Model 4: Virtue Ethics (Hybrid)
- **8 Aristotelian virtues**: Compassion, Justice, Honesty, Responsibility, Courage, Prudence, Temperance, Benevolence
- **Phronesis (practical wisdom)**: 10 domain-specific context profiles dynamically adjust virtue weights
- Virtue conflict detection and resolution
- **Weakness**: Virtue definitions are culturally dependent

### Model 5: Adversarial Robustness (Stress Testing)
- **18 attacks** across **7 categories**: Prompt Manipulation, Authority Pressure, Emotional Manipulation, Ambiguity, Bias Injection, Reward Hacking, Rule Exploitation
- Generates **5100+ adversarial variants** from the 1020-scenario dataset
- **Moral Robustness Score** = (Consistency + Resistance + Fairness + Harm Avoidance + Confidence Stability) / 5
- Detects and classifies **critical moral failures**

---

## Research Findings

Our experiments reveal:

1. **Rule-based systems** (Model 1) are fully transparent but fail in complex, context-dependent situations
2. **Learning-based systems** (Model 2) capture human moral patterns but inherit and amplify data biases
3. **RLHF systems** (Model 3) align with human preferences but risk **sycophancy** — optimizing for approval over truth
4. **Virtue ethics systems** (Model 4) provide the most balanced decisions through context-adaptive reasoning
5. **No model is immune to adversarial attacks** — all exhibit measurable vulnerabilities under pressure

> **Key insight**: The choice of ethical framework fundamentally shapes AI moral reasoning. No single approach is sufficient — each captures different aspects of human morality.

---

## Technical Requirements

- **Python**: 3.10 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended for stress testing)
- **No GPU required** — all models use pure NumPy (no PyTorch/TensorFlow)
- **No external APIs** — everything runs locally

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.30.0 | Interactive web dashboard |
| `pandas` | ≥2.0.0 | Data manipulation |
| `plotly` | ≥5.18.0 | Interactive visualizations |
| `numpy` | ≥1.24.0 | Core computation (all models) |

All ML models are implemented **from scratch** using only NumPy — no scikit-learn, PyTorch, or TensorFlow required.

---

## Citation

If you use this research in academic work, please cite:

```bibtex
@software{Ethica_2026,
  title     = {Ethica: Comparing Five Approaches to Machine Ethics},
  author    = {Pratyush},
  year      = {2026},
  url       = {https://github.com/47combinator/Ethica},
  note      = {5-model framework comparing rule-based, learning-based, RLHF, virtue ethics, and adversarial approaches to AI morality}
}
```

---

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Areas of interest:

- [ ] Additional ethical dilemma scenarios
- [ ] New attack types for Model 5
- [ ] Cross-cultural virtue definitions for Model 4
- [ ] Real human judgment data (crowdsourced)
- [ ] Integration with LLM-based moral reasoning

---

<p align="center">
  <strong>Built for AI Ethics Research</strong><br>
  <em>Exploring how different moral philosophies shape artificial intelligence</em>
</p>
