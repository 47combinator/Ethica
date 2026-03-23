"""
AI Morality Research Dashboard
==============================
Unified interface for all 5 models:
  Model 1: Rule-Based | Model 2: Learning-Based
  Model 3: RLHF      | Model 4: Virtue Ethics
  Model 5: Adversarial Robustness
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rules import EthicalRuleSystem
from core.engine import MoralDecisionEngine
from core.explanation import ExplanationGenerator
from core.evaluation import EvaluationSystem
from model2.trainer import MoralTrainer
from model2.predictor import MoralPredictor
from model2.explainer import Model2Explainer
from model2.evaluator import Model2Evaluator
from model3.base_model import BaseMoralModel
from model3.feedback import HumanFeedbackSystem
from model3.reward_model import RewardModel
from model3.rl_optimizer import RLOptimizer
from model3.predictor import RLHFPredictor
from model3.explainer import RLHFExplainer
from model3.evaluator import RLHFEvaluator
from model4.balancer import VirtueBalancer
from model4.predictor import VirtuePredictor
from model4.explainer import VirtueExplainer
from model4.evaluator import VirtueEvaluator
from model5.attacks import AttackLibrary, AttackType
from model5.generator import AdversarialGenerator
from model5.executor import AdversarialExecutor
from model5.detector import FailureDetector
from model5.scorer import RobustnessScorer
from data.scenarios import (
    get_all_scenarios, get_scenarios_by_category,
    get_scenario_by_id, CATEGORY_NAMES, get_category_counts
)

st.set_page_config(page_title="AI Morality Research", page_icon="brain", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.main-header p { color: #c4b5fd; font-size: 1rem; margin: 0.5rem 0 0 0; }
.m1-header h1 { background: linear-gradient(90deg, #34d399, #6ee7b7, #a7f3d0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.m2-header h1 { background: linear-gradient(90deg, #818cf8, #a78bfa, #c4b5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.m3-header h1 { background: linear-gradient(90deg, #f59e0b, #fbbf24, #fcd34d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.m4-header h1 { background: linear-gradient(90deg, #f472b6, #ec4899, #db2777); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.m5-header h1 { background: linear-gradient(90deg, #ef4444, #f97316, #eab308); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card { background: linear-gradient(145deg, #1e1b4b, #312e81); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid rgba(139,92,246,0.3); box-shadow: 0 4px 16px rgba(0,0,0,0.2); }
.metric-card h3 { color: #a78bfa; font-size: 0.8rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.5rem 0; }
.metric-card .value { color: #e0e7ff; font-size: 1.8rem; font-weight: 700; margin: 0; }
.db-m1 { background: linear-gradient(145deg, #064e3b, #065f46); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #34d399; margin: 1rem 0; }
.db-m2 { background: linear-gradient(145deg, #1e1b4b, #312e81); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #818cf8; margin: 1rem 0; }
.db-m3 { background: linear-gradient(145deg, #78350f, #92400e); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #fbbf24; margin: 1rem 0; }
.db-m4 { background: linear-gradient(145deg, #831843, #9d174d); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #f472b6; margin: 1rem 0; }
.db-m5 { background: linear-gradient(145deg, #7f1d1d, #991b1b); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #ef4444; margin: 1rem 0; }
div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1e1b4b 100%); }
</style>
""", unsafe_allow_html=True)

# --- Cached model init ---
@st.cache_resource
def init_m1():
    rs = EthicalRuleSystem()
    return rs, MoralDecisionEngine(rs), ExplanationGenerator(rs), EvaluationSystem()

@st.cache_resource
def init_m2():
    t = MoralTrainer("neural_network"); t.prepare_data(get_all_scenarios()); r = t.train(epochs=200)
    return t, MoralPredictor(t.model, t.feature_extractor), Model2Explainer(), Model2Evaluator(), r

@st.cache_resource
def init_m3():
    bm = BaseMoralModel(); fb = HumanFeedbackSystem(noise_level=0.1); rm = RewardModel()
    opt = RLOptimizer(bm, rm, fb); r = opt.train_full_pipeline(get_all_scenarios(), reward_epochs=80, rl_iterations=12)
    return bm, rm, opt, RLHFPredictor(bm, rm), RLHFExplainer(), RLHFEvaluator(), r

@st.cache_resource
def init_m4():
    b = VirtueBalancer(); stats = b.train_from_judgments(get_all_scenarios())
    return b, VirtuePredictor(b), VirtueExplainer(), VirtueEvaluator(), stats

# Helpers
counts = get_category_counts(); total = sum(counts.values())
def mc(col, l, v): col.markdown(f'<div class="metric-card"><h3>{l}</h3><p class="value">{v}</p></div>', unsafe_allow_html=True)
def hdr(t, s, c="main-header"): st.markdown(f'<div class="main-header {c}"><h1>{t}</h1><p>{s}</p></div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Select Model")
    model = st.radio("", [
        "Model 1: Rule-Based",
        "Model 2: Learning-Based",
        "Model 3: RLHF",
        "Model 4: Virtue Ethics",
        "Model 5: Adversarial"
    ], label_visibility="collapsed")
    st.markdown("---")
    if model == "Model 1: Rule-Based":
        page = st.radio("Pages", ["Dashboard", "Scenario Explorer", "Full Evaluation", "Rule System", "About"], label_visibility="collapsed")
    elif model == "Model 2: Learning-Based":
        page = st.radio("Pages", ["Dashboard", "Scenario Explorer", "Full Evaluation", "Training Insights", "About"], label_visibility="collapsed", key="m2p")
    elif model == "Model 3: RLHF":
        page = st.radio("Pages", ["Dashboard", "Scenario Explorer", "Full Evaluation", "RLHF Insights", "About"], label_visibility="collapsed", key="m3p")
    elif model == "Model 4: Virtue Ethics":
        page = st.radio("Pages", ["Dashboard", "Scenario Explorer", "Full Evaluation", "Virtue System", "About"], label_visibility="collapsed", key="m4p")
    else:
        page = st.radio("Pages", ["Dashboard", "Attack Library", "Run Stress Test", "Failure Analysis", "5-Model Comparison", "About"], label_visibility="collapsed", key="m5p")
    st.markdown("---")
    st.markdown(f"**{total}** dilemmas | **{len(CATEGORY_NAMES)}** categories")


# =============================================
# MODEL 1
# =============================================
if model == "Model 1: Rule-Based":
    rs, eng, expl, evl = init_m1()
    if page == "Dashboard":
        hdr("Model 1: Rule-Based Moral AI", "Top-Down Ethics | 22 Rules | AMR-220", "m1-header")
        cols = st.columns(4)
        for c, (l, v) in zip(cols, [("Dilemmas", total), ("Categories", len(CATEGORY_NAMES)), ("Rules", len(rs)), ("Priority Levels", 10)]): mc(c, l, v)
        cat_df = pd.DataFrame([{"Category": CATEGORY_NAMES.get(k,k), "Count": v} for k,v in counts.items()]).sort_values("Count", ascending=True)
        fig = px.bar(cat_df, x="Count", y="Category", orientation="h", color="Count", color_continuous_scale="Viridis", template="plotly_dark")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    elif page == "Scenario Explorer":
        hdr("Scenario Explorer - Model 1", "Rule engine analysis", "m1-header")
        c1, c2 = st.columns([1, 3])
        with c1:
            cat = st.selectbox("Category", list(CATEGORY_NAMES.keys()), format_func=lambda x: CATEGORY_NAMES[x])
            cs = get_scenarios_by_category(cat); sid = st.selectbox("Scenario", [s["id"] for s in cs], format_func=lambda x: f"{x} - {get_scenario_by_id(x)['title']}")
        sc = get_scenario_by_id(sid)
        with c2:
            if sc:
                st.markdown(f"### {sc['title']}\n*{sc['description']}*")
                for a in sc["actions"]: st.markdown(f"- **{a['id']}**: {a['description']}")
                if st.button("Run Analysis", type="primary"):
                    d = MoralDecisionEngine(rs).evaluate_scenario(sc)
                    st.markdown(f'<div class="db-m1"><h4 style="color:#34d399;margin:0">{d.chosen_action_description}</h4><p style="color:#a7f3d0">Confidence: {d.confidence:.0%}</p></div>', unsafe_allow_html=True)
                    for i, step in enumerate(d.reasoning_chain, 1): st.markdown(f"**{i}.** {step}")
                    with st.expander("Full Report"): st.code(expl.generate_full_explanation(d, sc), language="text")
    elif page == "Full Evaluation":
        hdr("Full Evaluation - Model 1", "All 220 dilemmas", "m1-header")
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Evaluating..."): decs = MoralDecisionEngine(rs).batch_evaluate(get_all_scenarios()); m = EvaluationSystem().evaluate_decisions(decs, get_all_scenarios())
            st.success(f"Overall: **{m['overall_score']:.3f}**")
            cols = st.columns(5)
            for c,(l,v) in zip(cols,[("Consistency",f"{m['moral_consistency']['average_consistency']:.0%}"),("Harm",f"{m['harm_minimization']['average_harm_minimization']:.0%}"),("Fairness",f"{m['fairness']['average_fairness']:.0%}"),("Transparency",f"{m['transparency']['transparency_score']:.0%}"),("Confidence",f"{m['confidence_distribution']['average_confidence']:.0%}")]): mc(c,l,v)
    elif page == "Rule System":
        hdr("Ethical Rule System", "22 hierarchical rules", "m1-header")
        for p, rules in rs.get_rule_hierarchy().items():
            st.markdown(f"### Priority {p}")
            for r in rules:
                with st.expander(f"**{r.rule_id}** - {r.name} (W:{r.weight})"): st.markdown(f"{r.description}\n\n**Category:** {r.category.value}")
    elif page == "About":
        hdr("About Model 1", "Rule-Based Moral AI", "m1-header")
        st.markdown("## Top-Down Ethics\nApplies **22 hierarchical moral rules**.\n\n### Limitations\n- Rigid rules fail with context\n- No learning\n- Cultural bias in rules")


# =============================================
# MODEL 2
# =============================================
elif model == "Model 2: Learning-Based":
    tr2, pr2, ex2, ev2, r2 = init_m2()
    if page == "Dashboard":
        hdr("Model 2: Learning-Based Moral AI", "Neural Network | Human Judgments", "m2-header")
        cols = st.columns(4)
        for c,(l,v) in zip(cols,[("Accuracy",f"{r2.get('final_accuracy',0):.0%}"),("Epochs",r2.get('epochs_trained',0)),("Dilemmas",total),("Features","55")]): mc(c,l,v)
        h = tr2.get_training_history()
        if h and h.get("loss"):
            c1,c2 = st.columns(2)
            with c1: fig = px.line(y=h["loss"],template="plotly_dark",labels={"x":"Epoch","y":"Loss"},title="Loss"); fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=40,b=0)); fig.update_traces(line_color="#818cf8"); st.plotly_chart(fig,use_container_width=True)
            with c2: fig = px.line(y=h["accuracy"],template="plotly_dark",labels={"x":"Epoch","y":"Accuracy"},title="Accuracy"); fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=40,b=0)); fig.update_traces(line_color="#34d399"); st.plotly_chart(fig,use_container_width=True)
    elif page == "Scenario Explorer":
        hdr("Scenario Explorer - Model 2", "Learned model analysis", "m2-header")
        c1,c2 = st.columns([1,3])
        with c1: cat = st.selectbox("Category",list(CATEGORY_NAMES.keys()),format_func=lambda x:CATEGORY_NAMES[x],key="m2c"); cs = get_scenarios_by_category(cat); sid = st.selectbox("Scenario",[s["id"] for s in cs],format_func=lambda x:f"{x} - {get_scenario_by_id(x)['title']}",key="m2s")
        sc = get_scenario_by_id(sid)
        with c2:
            if sc:
                st.markdown(f"### {sc['title']}\n*{sc['description']}*")
                for a in sc["actions"]: st.markdown(f"- **{a['id']}**: {a['description']}")
                if st.button("Run Analysis",type="primary",key="m2r"):
                    d = pr2.predict_scenario(sc)
                    st.markdown(f'<div class="db-m2"><h4 style="color:#818cf8;margin:0">{d.chosen_action_description}</h4><p style="color:#c4b5fd">Confidence: {d.confidence:.0%} | Agreement: {d.human_agreement:.0%}</p></div>', unsafe_allow_html=True)
                    with st.expander("Full Report"): st.code(ex2.generate_full_explanation(d, sc), language="text")
    elif page == "Full Evaluation":
        hdr("Full Evaluation - Model 2", "Learned morality", "m2-header")
        if st.button("Run Evaluation",type="primary",key="m2e"):
            with st.spinner("Evaluating..."): pr2.clear_log(); decs=pr2.batch_predict(get_all_scenarios()); m=ev2.evaluate_decisions(decs,get_all_scenarios())
            st.success(f"Overall: **{m['overall_score']:.3f}**")
            cols = st.columns(5)
            for c,(l,v) in zip(cols,[("Accuracy",f"{m['moral_accuracy']['accuracy']:.0%}"),("Bias",f"{1-m['bias_detection']['bias_score']:.0%}"),("Consistency",f"{m['consistency']['consistency_score']:.0%}"),("Context",f"{m['context_sensitivity']['sensitivity_score']:.0%}"),("Confidence",f"{m['confidence_analysis']['avg_confidence']:.0%}")]): mc(c,l,v)
    elif page == "Training Insights":
        hdr("Training Insights", "How Model 2 learned", "m2-header")
        st.code("Input(55) -> Dense(128,ReLU) -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)"); st.json(r2)
    elif page == "About":
        hdr("About Model 2", "Learning-Based Moral AI", "m2-header")
        st.markdown("## Bottom-Up Ethics\nLearns from **human judgments** via neural network.\n\n### Limitations\n- Bias amplification\n- Cultural variation\n- Less transparent")


# =============================================
# MODEL 3
# =============================================
elif model == "Model 3: RLHF":
    bm,rm_m,opt3,pr3,ex3,ev3,r3 = init_m3()
    if page == "Dashboard":
        hdr("Model 3: RLHF Moral AI", "Reinforcement Learning from Human Feedback", "m3-header")
        cols = st.columns(5); rl_h = r3.get("rl_history",{})
        for c,(l,v) in zip(cols,[("Accuracy",f"{r3.get('final_accuracy',0):.0%}"),("Sycophancy",f"{r3.get('final_sycophancy',0):.0%}"),("RL Iters",r3.get('rl_iterations',0)),("Reward Acc",f"{r3.get('reward_model_results',{}).get('final_accuracy',0):.0%}"),("Dilemmas",total)]): mc(c,l,v)
        if rl_h:
            c1,c2 = st.columns(2)
            with c1: fig = go.Figure(); fig.add_trace(go.Scatter(y=rl_h.get("avg_reward",[]),name="Reward",line=dict(color="#fbbf24"))); fig.add_trace(go.Scatter(y=rl_h.get("accuracy",[]),name="Accuracy",line=dict(color="#34d399"))); fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0)); st.plotly_chart(fig,use_container_width=True)
            with c2: fig = go.Figure(); fig.add_trace(go.Scatter(y=rl_h.get("sycophancy",[]),name="Sycophancy",line=dict(color="#f87171"))); fig.add_trace(go.Scatter(y=rl_h.get("kl_divergence",[]),name="KL",line=dict(color="#818cf8"))); fig.update_layout(template="plotly_dark",height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0)); st.plotly_chart(fig,use_container_width=True)
    elif page == "Scenario Explorer":
        hdr("Scenario Explorer - Model 3", "RLHF analysis", "m3-header")
        c1,c2 = st.columns([1,3])
        with c1: cat = st.selectbox("Category",list(CATEGORY_NAMES.keys()),format_func=lambda x:CATEGORY_NAMES[x],key="m3c"); cs = get_scenarios_by_category(cat); sid = st.selectbox("Scenario",[s["id"] for s in cs],format_func=lambda x:f"{x} - {get_scenario_by_id(x)['title']}",key="m3s")
        sc = get_scenario_by_id(sid)
        with c2:
            if sc:
                st.markdown(f"### {sc['title']}\n*{sc['description']}*")
                for a in sc["actions"]: st.markdown(f"- **{a['id']}**: {a['description']}")
                if st.button("Run RLHF Analysis",type="primary",key="m3r"):
                    d = pr3.predict_scenario(sc)
                    st.markdown(f'<div class="db-m3"><h4 style="color:#fbbf24;margin:0">{d.chosen_action_description}</h4><p style="color:#fcd34d">Confidence: {d.confidence:.0%} | Sycophancy: {d.sycophancy_risk:.0%}</p></div>', unsafe_allow_html=True)
                    with st.expander("Full Report"): st.code(ex3.generate_full_explanation(d, sc), language="text")
    elif page == "Full Evaluation":
        hdr("Full Evaluation - Model 3", "RLHF morality", "m3-header")
        if st.button("Run Evaluation",type="primary",key="m3e"):
            with st.spinner("Evaluating..."): pr3.clear_log(); decs=pr3.batch_predict(get_all_scenarios()); m=ev3.evaluate_decisions(decs,get_all_scenarios())
            st.success(f"Overall: **{m['overall_score']:.3f}**")
            cols = st.columns(5)
            for c,(l,v) in zip(cols,[("Accuracy",f"{m['moral_accuracy']['accuracy']:.0%}"),("Sycophancy",f"{m['sycophancy_analysis']['sycophancy_rate']:.0%}"),("Consistency",f"{m['consistency']['score']:.0%}"),("Robustness",f"{m['robustness']['score']:.0%}"),("Confidence",f"{m['confidence_analysis']['avg_confidence']:.0%}")]): mc(c,l,v)
            st.warning(m['sycophancy_analysis']['interpretation'])
    elif page == "RLHF Insights":
        hdr("RLHF Insights", "Alignment analysis", "m3-header")
        st.markdown("### Pipeline\n`Scenario -> Base Model -> Human Ranking -> Reward Model -> Policy Update`"); st.json(r3)
    elif page == "About":
        hdr("About Model 3", "RLHF Moral AI", "m3-header")
        st.markdown("## Human Feedback Alignment\nOptimizes decisions that **humans rate as ethical**.\n\n### Risks\n- Sycophancy\n- Bias inheritance\n- Preference manipulation")


# =============================================
# MODEL 4
# =============================================
elif model == "Model 4: Virtue Ethics":
    bal4, pr4, ex4, ev4, r4 = init_m4()
    if page == "Dashboard":
        hdr("Model 4: Virtue Ethics Moral AI", "Hybrid Model | 8 Aristotelian Virtues | Phronesis", "m4-header")
        cols = st.columns(5)
        for c,(l,v) in zip(cols,[("Accuracy",f"{r4.get('accuracy_after',0):.0%}"),("Improvement",f"{r4.get('improvement',0):+.0%}"),("Virtues","8"),("Categories","10"),("Dilemmas",total)]): mc(c,l,v)
        virtues = bal4.virtue_system.get_all_virtues()
        vdf = pd.DataFrame([{"Virtue":v.name,"Category":v.category.value,"Weight":v.base_weight} for v in virtues])
        fig = px.bar(vdf,x="Weight",y="Virtue",orientation="h",color="Category",color_discrete_map={"caring":"#f472b6","justice":"#818cf8","integrity":"#34d399","duty":"#fbbf24","fortitude":"#f87171","wisdom":"#a78bfa"},template="plotly_dark")
        fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),yaxis_title=""); st.plotly_chart(fig,use_container_width=True)
        from model4.context import CONTEXT_PROFILES
        pd_data = []
        for cat,prof in CONTEXT_PROFILES.items():
            for vid in ["V_COMP","V_JUST","V_HONE","V_RESP","V_COUR","V_PRUD"]:
                vname = bal4.virtue_system.get_virtue(vid).name
                pd_data.append({"Category":CATEGORY_NAMES.get(cat,cat),"Virtue":vname,"Weight":prof.get(vid,1.0)})
        pdf = pd.DataFrame(pd_data)
        fig = px.imshow(pdf.pivot(index="Category",columns="Virtue",values="Weight"),color_continuous_scale="RdYlGn",template="plotly_dark",aspect="auto")
        fig.update_layout(height=400,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0)); st.plotly_chart(fig,use_container_width=True)
    elif page == "Scenario Explorer":
        hdr("Scenario Explorer - Model 4", "Virtue ethics analysis", "m4-header")
        c1,c2 = st.columns([1,3])
        with c1: cat = st.selectbox("Category",list(CATEGORY_NAMES.keys()),format_func=lambda x:CATEGORY_NAMES[x],key="m4c"); cs = get_scenarios_by_category(cat); sid = st.selectbox("Scenario",[s["id"] for s in cs],format_func=lambda x:f"{x} - {get_scenario_by_id(x)['title']}",key="m4s")
        sc = get_scenario_by_id(sid)
        with c2:
            if sc:
                st.markdown(f"### {sc['title']}\n*{sc['description']}*")
                for a in sc["actions"]: st.markdown(f"- **{a['id']}**: {a['description']}")
                if st.button("Run Virtue Analysis",type="primary",key="m4r"):
                    d = pr4.predict_scenario(sc)
                    st.markdown(f'<div class="db-m4"><h4 style="color:#f472b6;margin:0">{d.chosen_action_description}</h4><p style="color:#fbcfe8">Confidence: {d.confidence:.0%} | Virtue: {d.dominant_virtue_name}</p></div>', unsafe_allow_html=True)
                    vp = d.virtue_profile; v_names = [bal4.virtue_system.get_virtue(vid).name for vid in vp.keys()]; v_scores = list(vp.values())
                    fig = go.Figure(data=go.Scatterpolar(r=v_scores+[v_scores[0]],theta=v_names+[v_names[0]],fill='toself',line_color='#f472b6',fillcolor='rgba(244,114,182,0.2)'))
                    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,1])),template="plotly_dark",height=400,paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=40,r=40,t=40,b=40)); st.plotly_chart(fig,use_container_width=True)
                    if d.conflicts:
                        st.markdown("### Virtue Conflicts")
                        for c in d.conflicts[:5]:
                            va = bal4.virtue_system.get_virtue(c["virtue_a"]); vb = bal4.virtue_system.get_virtue(c["virtue_b"]); dom = bal4.virtue_system.get_virtue(c["dominant"])
                            st.markdown(f"- **{va.name}** ({c['score_a']:.2f}) vs **{vb.name}** ({c['score_b']:.2f}) -> **{dom.name}** prevails")
                    with st.expander("Full Report"): st.code(ex4.generate_full_explanation(d, sc), language="text")
    elif page == "Full Evaluation":
        hdr("Full Evaluation - Model 4", "Virtue ethics assessment", "m4-header")
        if st.button("Run Evaluation",type="primary",key="m4e"):
            with st.spinner("Evaluating..."): pr4.clear_log(); decs=pr4.batch_predict(get_all_scenarios()); m=ev4.evaluate_decisions(decs,get_all_scenarios())
            st.success(f"Overall: **{m['overall_score']:.3f}**")
            cols = st.columns(5)
            for c,(l,v) in zip(cols,[("Accuracy",f"{m['moral_accuracy']['accuracy']:.0%}"),("Balance",f"{m['virtue_balance']['balance_score']:.0%}"),("Consistency",f"{m['consistency']['score']:.0%}"),("Context",f"{m['context_sensitivity']['score']:.0%}"),("Transparency",f"{m['transparency']['score']:.0%}")]): mc(c,l,v)
            vb = m['virtue_balance']['distribution']
            if vb:
                vbd = pd.DataFrame([{"Virtue":bal4.virtue_system.get_virtue(k).name if bal4.virtue_system.get_virtue(k) else k,"Proportion":v} for k,v in vb.items()])
                fig = px.pie(vbd,values="Proportion",names="Virtue",template="plotly_dark",color_discrete_sequence=["#f472b6","#818cf8","#34d399","#fbbf24","#f87171","#a78bfa","#6ee7b7","#fcd34d"])
                fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig,use_container_width=True)
    elif page == "Virtue System":
        hdr("Virtue System", "8 Aristotelian Virtues", "m4-header")
        hierarchy = bal4.virtue_system.get_virtue_hierarchy()
        for cat, virtues in hierarchy.items():
            st.markdown(f"### {cat.title()}")
            for v in virtues:
                with st.expander(f"**{v.id}** - {v.name} (Weight: {v.base_weight})"): st.markdown(f"**Description:** {v.description}\n\n**Opposing:** {', '.join(v.opposing_virtues)}")
    elif page == "About":
        hdr("About Model 4", "Virtue Ethics Moral AI", "m4-header")
        st.markdown("## Virtue Ethics (Hybrid)\nGuided by **8 Aristotelian virtues** with phronesis.\n\n### Strengths\n- Context-adaptive\n- Transparent reasoning\n- Handles moral ambiguity")


# =============================================
# MODEL 5 - ADVERSARIAL ROBUSTNESS
# =============================================
else:
    atk_lib = AttackLibrary()

    if page == "Dashboard":
        hdr("Model 5: Adversarial Moral Robustness AI", "Stress-Testing AI Ethics | 18 Attacks | 7 Categories", "m5-header")
        cols = st.columns(5)
        for c,(l,v) in zip(cols,[("Attacks",len(atk_lib)),("Attack Types","7"),("Target Models","4"),("Dilemmas",total),("Max Adv Cases",f"{total*5}+")]): mc(c,l,v)

        st.markdown("### Attack Type Distribution")
        type_counts = atk_lib.get_attack_type_counts()
        tdf = pd.DataFrame([{"Type": k.replace("_"," ").title(), "Count": v} for k,v in type_counts.items()])
        fig = px.bar(tdf, x="Count", y="Type", orientation="h", color="Count", color_continuous_scale="Reds", template="plotly_dark")
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=10), coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### How It Works")
        st.markdown("""
        ```
        Original Scenario -> Adversarial Modifier -> Modified Scenario
              |                                           |
          Model runs                                  Model runs
              |                                           |
         Normal Decision                          Adversarial Decision
              |___________________________________________|
                                    |
                          Failure Detection
                                    |
                         Robustness Score
        ```
        """)

        st.markdown("### Quick Stress Test")
        sample_size = st.slider("Scenarios to test:", 10, 60, 30, step=10)
        if st.button("Run Quick Test", type="primary"):
            scenarios = get_all_scenarios()[:sample_size]
            gen = AdversarialGenerator()
            adv = gen.generate_full_dataset(scenarios, attacks_per_scenario=5)
            exe = AdversarialExecutor()
            det = FailureDetector()
            scorer = RobustnessScorer()

            with st.spinner(f"Stress-testing all 4 models on {len(adv)} adversarial cases..."):
                normal = {
                    "Model 1": exe.run_model1(scenarios),
                    "Model 2": exe.run_model2(scenarios),
                    "Model 3": exe.run_model3(scenarios),
                    "Model 4": exe.run_model4(scenarios),
                }
                adversarial = {
                    "Model 1": exe.run_model1(adv, True),
                    "Model 2": exe.run_model2(adv, True),
                    "Model 3": exe.run_model3(adv, True),
                    "Model 4": exe.run_model4(adv, True),
                }
                failures = det.detect_failures(normal, adversarial)
                scores = scorer.compute_scores(normal, adversarial, failures)

            st.markdown("### Robustness Rankings")
            ranking = scorer.get_comparison_table()
            rdf = pd.DataFrame(ranking)
            st.dataframe(rdf, use_container_width=True)

            # Bar chart
            fig = go.Figure()
            clrs = {"Model 1":"#34d399","Model 2":"#818cf8","Model 3":"#fbbf24","Model 4":"#f472b6"}
            for row in ranking:
                fig.add_trace(go.Bar(x=["Overall","Consistency","Resistance","Fairness","Harm Avoid"],
                    y=[row["Robustness"],row["Consistency"],row["Resistance"],row["Fairness"],row["Harm Avoidance"]],
                    name=row["Model"], marker_color=clrs.get(row["Model"],"#888")))
            fig.update_layout(barmode="group",template="plotly_dark",height=400,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig, use_container_width=True)

            summary = det.get_failure_summary()
            st.markdown(f"### Failure Summary")
            st.markdown(f"- **Total Failures**: {summary['total_failures']}")
            st.markdown(f"- **Most Vulnerable**: {summary.get('most_vulnerable_model','N/A')}")
            st.markdown(f"- **Most Effective Attack**: {summary.get('most_effective_attack','N/A').replace('_',' ').title()}")

    elif page == "Attack Library":
        hdr("Attack Library", "18 adversarial attacks across 7 categories", "m5-header")
        for atype in AttackType:
            attacks = atk_lib.get_attacks_by_type(atype)
            if attacks:
                st.markdown(f"### {atype.value.replace('_',' ').title()} ({len(attacks)} attacks)")
                for a in attacks:
                    with st.expander(f"**{a.id}** - {a.name} (Severity: {a.severity:.0%})"):
                        st.markdown(f"**Description:** {a.description}")
                        st.markdown(f"**Targets:** {', '.join(a.targets)}")
                        st.markdown(f"**Modifier keys:** {', '.join(a.modifier.keys())}")

    elif page == "Run Stress Test":
        hdr("Full Stress Test", "Run all attacks against all models", "m5-header")
        st.markdown("Configure the stress test parameters:")
        c1, c2 = st.columns(2)
        with c1:
            n_scenarios = st.slider("Number of scenarios:", 20, 220, 50, step=10, key="st_n")
        with c2:
            atk_per = st.slider("Attacks per scenario:", 3, 7, 5, key="st_a")

        if st.button("Execute Full Stress Test", type="primary", key="st_run"):
            scenarios = get_all_scenarios()[:n_scenarios]
            gen = AdversarialGenerator()
            adv = gen.generate_full_dataset(scenarios, attacks_per_scenario=atk_per)
            gen_stats = gen.get_generation_stats(adv)
            exe = AdversarialExecutor()
            det = FailureDetector()
            scorer = RobustnessScorer()

            prog = st.progress(0, "Starting stress test...")
            prog.progress(5, "Running Model 1 (normal)...")
            n1 = exe.run_model1(scenarios)
            prog.progress(10, "Running Model 1 (adversarial)...")
            a1 = exe.run_model1(adv, True)
            prog.progress(25, "Running Model 2...")
            n2 = exe.run_model2(scenarios)
            a2 = exe.run_model2(adv, True)
            prog.progress(50, "Running Model 3...")
            n3 = exe.run_model3(scenarios)
            a3 = exe.run_model3(adv, True)
            prog.progress(75, "Running Model 4...")
            n4 = exe.run_model4(scenarios)
            a4 = exe.run_model4(adv, True)
            prog.progress(90, "Analyzing failures...")

            normal = {"Model 1":n1,"Model 2":n2,"Model 3":n3,"Model 4":n4}
            adversarial = {"Model 1":a1,"Model 2":a2,"Model 3":a3,"Model 4":a4}

            failures = det.detect_failures(normal, adversarial)
            scores = scorer.compute_scores(normal, adversarial, failures)
            prog.progress(100, "Complete!")

            st.success(f"Tested {n_scenarios} scenarios x {atk_per} attacks = **{gen_stats['total_adversarial']}** adversarial cases")

            st.markdown("### Robustness Scores")
            ranking = scorer.get_comparison_table()
            st.dataframe(pd.DataFrame(ranking), use_container_width=True)

            # Detailed per-model cards
            for name in ["Model 1","Model 2","Model 3","Model 4"]:
                s = scores[name]
                with st.expander(f"**{name}** - Robustness: {s['robustness_score']:.3f} - {s['interpretation']}"):
                    cols = st.columns(5)
                    mc(cols[0], "Consistency", f"{s['ethical_consistency']:.0%}")
                    mc(cols[1], "Resistance", f"{s['manipulation_resistance']:.0%}")
                    mc(cols[2], "Fairness", f"{s['fairness_preservation']:.0%}")
                    mc(cols[3], "Harm Avoid", f"{s['harm_avoidance']:.0%}")
                    mc(cols[4], "Conf Stability", f"{s['confidence_stability']:.0%}")

                    if s["attack_type_resistance"]:
                        atr_df = pd.DataFrame([{"Attack":k.replace("_"," ").title(),"Resistance":v} for k,v in s["attack_type_resistance"].items()])
                        fig = px.bar(atr_df,x="Resistance",y="Attack",orientation="h",color="Resistance",color_continuous_scale="RdYlGn",template="plotly_dark")
                        fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)

    elif page == "Failure Analysis":
        hdr("Failure Analysis", "Examine how models break under pressure", "m5-header")
        st.markdown("Run a focused failure analysis to identify critical vulnerabilities.")

        if st.button("Run Failure Analysis", type="primary", key="fa_run"):
            scenarios = get_all_scenarios()[:40]
            gen = AdversarialGenerator()
            adv = gen.generate_full_dataset(scenarios, attacks_per_scenario=5)
            exe = AdversarialExecutor()
            det = FailureDetector()
            scorer = RobustnessScorer()

            with st.spinner("Running failure analysis..."):
                normal = {"Model 1":exe.run_model1(scenarios),"Model 2":exe.run_model2(scenarios),"Model 3":exe.run_model3(scenarios),"Model 4":exe.run_model4(scenarios)}
                adversarial = {"Model 1":exe.run_model1(adv,True),"Model 2":exe.run_model2(adv,True),"Model 3":exe.run_model3(adv,True),"Model 4":exe.run_model4(adv,True)}
                failures = det.detect_failures(normal, adversarial)
                scores = scorer.compute_scores(normal, adversarial, failures)

            summary = det.get_failure_summary()

            st.markdown("### Failure Overview")
            cols = st.columns(4)
            mc(cols[0], "Total Failures", summary.get("total_failures",0))
            mc(cols[1], "Critical", summary.get("failures_by_severity",{}).get("critical",0))
            mc(cols[2], "Most Vulnerable", summary.get("most_vulnerable_model","N/A"))
            mc(cols[3], "Worst Attack", summary.get("most_effective_attack","N/A").replace("_"," ").title())

            # By model
            if summary.get("failures_by_model"):
                st.markdown("### Failures by Model")
                fdf = pd.DataFrame([{"Model":k,"Failures":v} for k,v in summary["failures_by_model"].items()])
                fig = px.bar(fdf,x="Model",y="Failures",color="Model",color_discrete_map={"Model 1":"#34d399","Model 2":"#818cf8","Model 3":"#fbbf24","Model 4":"#f472b6"},template="plotly_dark")
                fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),showlegend=False); st.plotly_chart(fig,use_container_width=True)

            # By attack type
            if summary.get("failures_by_attack_type"):
                st.markdown("### Failures by Attack Type")
                adf = pd.DataFrame([{"Attack":k.replace("_"," ").title(),"Failures":v} for k,v in summary["failures_by_attack_type"].items()])
                fig = px.bar(adf,x="Failures",y="Attack",orientation="h",color="Failures",color_continuous_scale="OrRd",template="plotly_dark")
                fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=10,b=0),coloraxis_showscale=False); st.plotly_chart(fig,use_container_width=True)

            # Critical failures list
            critical = det.get_critical_failures()
            if critical:
                st.markdown("### Critical Failures")
                for f in critical[:15]:
                    st.markdown(f'<div class="db-m5"><b>{f.model_name}</b> | {f.scenario_id}<br><span style="color:#fca5a5">{f.description}</span><br><small>Type: {f.failure_type} | Attack: {f.attack_type.replace("_"," ")}</small></div>', unsafe_allow_html=True)

            # Severity distribution
            if summary.get("failures_by_severity"):
                st.markdown("### Severity Distribution")
                sdf = pd.DataFrame([{"Severity":k.title(),"Count":v} for k,v in summary["failures_by_severity"].items()])
                fig = px.pie(sdf,values="Count",names="Severity",color="Severity",color_discrete_map={"Minor":"#fbbf24","Moderate":"#f97316","Critical":"#ef4444"},template="plotly_dark")
                fig.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)"); st.plotly_chart(fig,use_container_width=True)

    elif page == "5-Model Comparison":
        hdr("5-Model Comparison", "Normal Performance + Adversarial Robustness", "m5-header")
        if st.button("Run Complete 5-Model Comparison", type="primary", key="cmp5"):
            scenarios = get_all_scenarios()
            sample = scenarios[:40]

            with st.spinner("Running all 5 analyses across 220+200 scenarios..."):
                # Normal eval for all 4 models
                rs1=EthicalRuleSystem(); m1d=MoralDecisionEngine(rs1).batch_evaluate(scenarios); m1m=EvaluationSystem().evaluate_decisions(m1d,scenarios)
                _,_p2,_,_e2,_=init_m2(); _p2.clear_log(); m2d=_p2.batch_predict(scenarios); m2m=_e2.evaluate_decisions(m2d,scenarios)
                _,_,_,_p3,_,_e3,_=init_m3(); _p3.clear_log(); m3d=_p3.batch_predict(scenarios); m3m=_e3.evaluate_decisions(m3d,scenarios)
                _b4,_p4,_,_e4,_=init_m4(); _p4.clear_log(); m4d=_p4.batch_predict(scenarios); m4m=_e4.evaluate_decisions(m4d,scenarios)

                # Adversarial
                gen = AdversarialGenerator(); adv = gen.generate_full_dataset(sample, 5)
                exe = AdversarialExecutor(); det = FailureDetector(); scorer = RobustnessScorer()
                normal_r = {"Model 1":exe.run_model1(sample),"Model 2":exe.run_model2(sample),"Model 3":exe.run_model3(sample),"Model 4":exe.run_model4(sample)}
                adv_r = {"Model 1":exe.run_model1(adv,True),"Model 2":exe.run_model2(adv,True),"Model 3":exe.run_model3(adv,True),"Model 4":exe.run_model4(adv,True)}
                fails = det.detect_failures(normal_r, adv_r)
                rob_scores = scorer.compute_scores(normal_r, adv_r, fails)

            st.markdown("### Complete Research Summary")
            comp = pd.DataFrame({
                "Metric": ["Normal Score", "Robustness Score", "Failures", "Consistency", "Unique Strength"],
                "M1 Rules": [f"{m1m['overall_score']:.3f}", f"{rob_scores['Model 1']['robustness_score']:.3f}", rob_scores['Model 1']['total_failures'], f"{m1m['moral_consistency']['average_consistency']:.3f}", "Transparency"],
                "M2 Learned": [f"{m2m['overall_score']:.3f}", f"{rob_scores['Model 2']['robustness_score']:.3f}", rob_scores['Model 2']['total_failures'], f"{m2m['consistency']['consistency_score']:.3f}", "Pattern learning"],
                "M3 RLHF": [f"{m3m['overall_score']:.3f}", f"{rob_scores['Model 3']['robustness_score']:.3f}", rob_scores['Model 3']['total_failures'], f"{m3m['consistency']['score']:.3f}", "Human alignment"],
                "M4 Virtue": [f"{m4m['overall_score']:.3f}", f"{rob_scores['Model 4']['robustness_score']:.3f}", rob_scores['Model 4']['total_failures'], f"{m4m['consistency']['score']:.3f}", "Context balance"],
            })
            st.dataframe(comp, use_container_width=True)

            # Normal vs Adversarial chart
            fig = go.Figure()
            names = ["Model 1","Model 2","Model 3","Model 4"]
            normal_s = [m1m['overall_score'],m2m['overall_score'],m3m['overall_score'],m4m['overall_score']]
            robust_s = [rob_scores[n]['robustness_score'] for n in names]
            fig.add_trace(go.Bar(x=names,y=normal_s,name="Normal Score",marker_color="rgba(52,211,153,0.8)"))
            fig.add_trace(go.Bar(x=names,y=robust_s,name="Robustness Score",marker_color="rgba(239,68,68,0.8)"))
            fig.update_layout(barmode="group",template="plotly_dark",height=400,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=30,b=0),title="Normal Performance vs Adversarial Robustness")
            st.plotly_chart(fig, use_container_width=True)

            # Performance drop
            st.markdown("### Performance Drop Under Attack")
            for name,ns,rs_ in zip(names,normal_s,robust_s):
                drop = ns - rs_
                color = "#34d399" if drop < 0 else "#fbbf24" if drop < 0.2 else "#ef4444"
                st.markdown(f"- **{name}**: {ns:.3f} -> {rs_:.3f} (drop: **{drop:+.3f}**)")

            # Agreement matrix
            st.markdown("### Decision Agreement")
            n = len(scenarios)
            pairs = [("M1","M2",m1d,m2d),("M1","M3",m1d,m3d),("M1","M4",m1d,m4d),("M2","M3",m2d,m3d),("M2","M4",m2d,m4d),("M3","M4",m3d,m4d)]
            for n1,n2,d1,d2 in pairs:
                ag = sum(1 for a,b in zip(d1,d2) if a.chosen_action==b.chosen_action)/n
                st.markdown(f"- {n1} vs {n2}: **{ag:.1%}**")

            all4 = sum(1 for a,b,c,d in zip(m1d,m2d,m3d,m4d) if a.chosen_action==b.chosen_action==c.chosen_action==d.chosen_action)
            st.metric("All 4 Models Agree", f"{all4/n:.1%}")

            st.markdown("### Key Research Insights")
            most_robust = scorer.get_most_robust()
            most_vuln = scorer.get_most_vulnerable()
            fail_sum = det.get_failure_summary()
            st.markdown(f"""
            - **Most robust model**: {most_robust} ({rob_scores[most_robust]['robustness_score']:.3f})
            - **Most vulnerable model**: {most_vuln} ({rob_scores[most_vuln]['robustness_score']:.3f})
            - **Most effective attack**: {fail_sum.get('most_effective_attack','N/A').replace('_',' ').title()}
            - **Total failures detected**: {fail_sum.get('total_failures',0)}
            
            > **Conclusion**: No model is immune to adversarial attacks. 
            > Each ethical framework has unique vulnerabilities that emerge under pressure.
            > This proves that adversarial robustness testing is essential for AI safety.
            """)

    elif page == "About":
        hdr("About Model 5", "Adversarial Moral Robustness AI", "m5-header")
        st.markdown("""
        ## Stress-Testing AI Ethics
        Instead of building a "better" moral model, Model 5 **attacks** the other models to measure failure.
        
        ### 7 Attack Categories
        | Category | Goal | Primary Target |
        |----------|------|---------------|
        | Prompt Manipulation | Trick AI into unethical decisions | All |
        | Authority Pressure | Simulate power influence | Model 1, 4 |
        | Emotional Manipulation | Bias through emotional pressure | Model 3, 4 |
        | Ambiguity Attacks | Confuse with incomplete data | Model 1, 2 |
        | Bias Injection | Inject discriminatory framing | Model 2, 3 |
        | Reward Hacking | Exploit RLHF approval-seeking | Model 3 |
        | Rule Exploitation | Create rule conflicts | Model 1 |
        
        ### Why This Matters
        Most AI ethics research asks: *"Is AI ethical?"*
        
        This model asks: **"Is AI ethical when someone tries to break it?"**
        
        This is exactly what real-world systems face: misinformation, political pressure, 
        corporate manipulation, and adversarial exploitation.
        
        *Part of the AI Morality Research Project by Pratyush*
        """)
