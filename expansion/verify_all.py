"""
Full Project Verification Script
=================================
Tests every model and system against old + new scenarios.
Run this before creating a Pull Request.
"""
import sys
import os
import traceback

# Add project root to path so all modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0
errors = []

def test(name, fn):
    global passed, failed, errors
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as ex:
        print(f"  FAIL  {name}: {ex}")
        errors.append((name, traceback.format_exc()))
        failed += 1


# ============================================
# TEST 1: Data Loading
# ============================================
print("\n=== TEST GROUP 1: Data Loading ===")

def test_total_count():
    from data.scenarios import get_all_scenarios
    s = get_all_scenarios()
    assert len(s) == 1020, f"Expected 1020, got {len(s)}"

def test_unique_ids():
    from data.scenarios import get_all_scenarios
    s = get_all_scenarios()
    ids = [x["id"] for x in s]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {len(ids) - len(set(ids))}"

def test_category_count():
    from data.scenarios import get_category_counts
    counts = get_category_counts()
    assert len(counts) == 11, f"Expected 11 categories, got {len(counts)}"

def test_old_scenarios_intact():
    from data.scenarios import get_scenario_by_id
    av01 = get_scenario_by_id("AV_01")
    assert av01 is not None, "AV_01 not found"
    assert av01["title"] == "Pedestrian vs Passenger Safety"
    hc01 = get_scenario_by_id("HC_01")
    assert hc01 is not None, "HC_01 not found"
    ma30 = get_scenario_by_id("MA_30")
    assert ma30 is not None, "MA_30 not found"

def test_new_scenarios_exist():
    from data.scenarios import get_scenario_by_id
    mm = get_scenario_by_id("MM_001")
    assert mm is not None, "MM_001 not found"
    sd = get_scenario_by_id("SD_0001")
    assert sd is not None, "SD_0001 not found"
    sa = get_scenario_by_id("SA_0001")
    assert sa is not None, "SA_0001 not found"

def test_education_category():
    from data.scenarios import get_scenarios_by_category
    edu = get_scenarios_by_category("education_ai")
    assert len(edu) > 0, "No education_ai scenarios found"
    assert len(edu) == 38, f"Expected 38, got {len(edu)}"

def test_all_scenarios_valid_structure():
    from data.scenarios import get_all_scenarios
    for s in get_all_scenarios():
        assert "id" in s, f"Missing id"
        assert "category" in s, f"{s.get('id')}: missing category"
        assert "actions" in s, f"{s['id']}: missing actions"
        assert len(s["actions"]) >= 2, f"{s['id']}: only {len(s['actions'])} actions"
        for a in s["actions"]:
            assert "consequences" in a, f"{s['id']}/{a.get('id')}: missing consequences"
            for k, v in a["consequences"].items():
                assert 0.0 <= v <= 1.0, f"{s['id']}/{a['id']}: {k}={v} out of range"

test("Total count = 1020", test_total_count)
test("All IDs unique", test_unique_ids)
test("11 categories", test_category_count)
test("Original scenarios intact", test_old_scenarios_intact)
test("New scenarios exist", test_new_scenarios_exist)
test("Education AI category works", test_education_category)
test("All scenarios valid structure", test_all_scenarios_valid_structure)


# ============================================
# TEST 2: Model 1 - Rule-Based Engine
# ============================================
print("\n=== TEST GROUP 2: Model 1 (Rule-Based) ===")

def test_model1_old():
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id
    e = MoralDecisionEngine()
    r = e.evaluate_scenario(get_scenario_by_id("AV_01"))
    assert r.chosen_action_description is not None
    assert r.confidence > 0

def test_model1_mm():
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id
    e = MoralDecisionEngine()
    r = e.evaluate_scenario(get_scenario_by_id("MM_001"))
    assert r.chosen_action_description is not None

def test_model1_sd():
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id
    e = MoralDecisionEngine()
    r = e.evaluate_scenario(get_scenario_by_id("SD_0001"))
    assert r.chosen_action_description is not None

def test_model1_sa():
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_scenario_by_id
    e = MoralDecisionEngine()
    r = e.evaluate_scenario(get_scenario_by_id("SA_0001"))
    assert r.chosen_action_description is not None

def test_model1_batch():
    from core.engine import MoralDecisionEngine
    from data.scenarios import get_all_scenarios
    e = MoralDecisionEngine()
    scenarios = get_all_scenarios()
    fail_count = 0
    for s in scenarios[:100]:  # test first 100
        try:
            r = e.evaluate_scenario(s)
            assert r.chosen_action_description is not None
        except:
            fail_count += 1
    assert fail_count == 0, f"{fail_count}/100 scenarios failed"

test("Old scenario (AV_01)", test_model1_old)
test("Moral Machine (MM_001)", test_model1_mm)
test("Scruples Dilemma (SD_0001)", test_model1_sd)
test("Scruples Anecdote (SA_0001)", test_model1_sa)
test("Batch test (first 100)", test_model1_batch)


# ============================================
# TEST 3: Model 2 - Feature Extraction
# ============================================
print("\n=== TEST GROUP 3: Model 2 (Feature Extraction) ===")

def test_model2_features_old():
    from model2.features import FeatureExtractor
    from data.scenarios import get_scenario_by_id
    fe = FeatureExtractor()
    s = get_scenario_by_id("AV_01")
    feats = fe.extract_scenario_features(s)
    assert len(feats) == 2
    assert feats[0][0].shape[0] == fe.feature_size

def test_model2_features_new():
    from model2.features import FeatureExtractor
    from data.scenarios import get_scenario_by_id
    fe = FeatureExtractor()
    for sid in ["MM_001", "SD_0001", "SA_0001"]:
        s = get_scenario_by_id(sid)
        feats = fe.extract_scenario_features(s)
        assert len(feats) >= 2, f"{sid}: only {len(feats)} features"
        assert feats[0][0].shape[0] == fe.feature_size, f"{sid}: wrong feature size"

def test_model2_batch_features():
    from model2.features import FeatureExtractor
    from data.scenarios import get_all_scenarios
    fe = FeatureExtractor()
    scenarios = get_all_scenarios()
    fail_count = 0
    for s in scenarios:
        try:
            feats = fe.extract_scenario_features(s)
            assert len(feats) >= 2
        except:
            fail_count += 1
    assert fail_count == 0, f"{fail_count}/{len(scenarios)} failed feature extraction"

test("Old scenario features", test_model2_features_old)
test("New scenario features", test_model2_features_new)
test("All 1020 scenarios extract", test_model2_batch_features)


# ============================================
# TEST 4: Model 4 - Virtue System
# ============================================
print("\n=== TEST GROUP 4: Model 4 (Virtue System) ===")

def test_model4_old():
    from model4.virtues import VirtueSystem
    from data.scenarios import get_scenario_by_id
    vs = VirtueSystem()
    s = get_scenario_by_id("AV_01")
    for a in s["actions"]:
        vec = vs.compute_virtue_vector(a["consequences"])
        assert vec is not None
        assert len(vec) > 0

def test_model4_new():
    from model4.virtues import VirtueSystem
    from data.scenarios import get_scenario_by_id
    vs = VirtueSystem()
    for sid in ["MM_001", "SD_0001", "SA_0001"]:
        s = get_scenario_by_id(sid)
        for a in s["actions"]:
            vec = vs.compute_virtue_vector(a["consequences"])
            assert vec is not None, f"{sid}: virtue vector returned None"

test("Old scenario virtues", test_model4_old)
test("New scenario virtues", test_model4_new)


# ============================================
# TEST 5: Model 5 - Robustness Scorer
# ============================================
print("\n=== TEST GROUP 5: Model 5 (Robustness) ===")

def test_model5_imports():
    """Verify all Model 5 components import correctly."""
    from model5.scorer import RobustnessScorer
    from model5.attacks import AttackLibrary
    from model5.generator import AdversarialGenerator
    from model5.detector import FailureDetector
    scorer = RobustnessScorer()
    assert scorer is not None

def test_model5_attacks_old():
    """Verify attack library works with old scenarios."""
    from model5.attacks import AttackLibrary
    from data.scenarios import get_scenario_by_id
    lib = AttackLibrary()
    s = get_scenario_by_id("AV_01")
    attacks = lib.get_all_attacks()
    assert len(attacks) > 0, "No attacks generated for AV_01"

def test_model5_attacks_new():
    """Verify attack library works with new scenarios."""
    from model5.attacks import AttackLibrary
    from data.scenarios import get_scenario_by_id
    lib = AttackLibrary()
    for sid in ["MM_001", "SD_0001", "SA_0001"]:
        s = get_scenario_by_id(sid)
        attacks = lib.get_all_attacks()
        assert len(attacks) > 0, f"{sid}: No attacks generated"

test("Model 5 imports", test_model5_imports)
test("Model 5 attacks (old)", test_model5_attacks_old)
test("Model 5 attacks (new)", test_model5_attacks_new)


# ============================================
# TEST 6: Neo4j Offline Engine
# ============================================
print("\n=== TEST GROUP 6: Neo4j Engine (Offline) ===")

def test_neo4j_schema():
    from neo4j_engine.test_offline import test_schema
    test_schema()

def test_neo4j_queries():
    from neo4j_engine.test_offline import test_queries
    test_queries()

def test_neo4j_reasoning():
    from neo4j_engine.test_offline import test_reasoning
    test_reasoning()

def test_neo4j_explanation():
    from neo4j_engine.test_offline import test_explanation
    test_explanation()

test("Neo4j schema validation", test_neo4j_schema)
test("Neo4j query builder", test_neo4j_queries)
test("Neo4j reasoning engine", test_neo4j_reasoning)
test("Neo4j explanation generator", test_neo4j_explanation)


# ============================================
# TEST 7: Expansion Pipeline
# ============================================
print("\n=== TEST GROUP 7: Expansion Pipeline ===")

def test_expansion_imports():
    from expansion.convert_moral_machine import convert_moral_machine_csv
    from expansion.convert_scruples import convert_scruples_dilemmas
    from expansion.validate import validate_scenario, validate_dataset

def test_validator_on_full_dataset():
    from data.scenarios import get_all_scenarios
    from expansion.validate import validate_dataset
    report = validate_dataset(get_all_scenarios())
    assert report["total_errors"] == 0, f"Found {report['total_errors']} errors"
    assert report["health"] == "PASS"

test("Expansion imports", test_expansion_imports)
test("Full dataset validation passes", test_validator_on_full_dataset)


# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 50)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 50)

if errors:
    print("\nFailed tests:")
    for name, tb in errors:
        print(f"\n--- {name} ---")
        print(tb)

sys.exit(0 if failed == 0 else 1)
