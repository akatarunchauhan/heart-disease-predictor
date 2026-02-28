import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import joblib
import streamlit.components.v1 as components

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f0f14; color: #e8e4df; }

[data-testid="stSidebar"] {
    background: #16161f;
    border-right: 1px solid #2a2a38;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #a09898 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    line-height: 1.1;
    color: #f5f0ea;
    margin: 0;
}
.hero-title span { color: #e05c5c; font-style: italic; }
.hero-sub {
    color: #7a7585;
    font-size: 1rem;
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* metric cards now use inline styles */

.result-positive {
    background: linear-gradient(135deg, #3d1a1a 0%, #2a0f0f 100%);
    border: 1px solid #e05c5c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #0f2b1a 0%, #0a1f12 100%);
    border: 1px solid #3db87a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin: 0;
}
.result-positive .result-title { color: #e05c5c; }
.result-negative .result-title { color: #3db87a; }
.result-sub { color: #a09898; font-size: 0.88rem; margin-top: 0.5rem; }

.divider { border: none; border-top: 1px solid #2a2a38; margin: 1.5rem 0; }

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e05c5c;
    margin-bottom: 0.8rem;
}

.conf-bar-outer {
    background: #2a2a38;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.6rem;
}
.conf-bar-inner-pos {
    background: linear-gradient(90deg, #e05c5c, #ff8c8c);
    height: 100%; border-radius: 99px;
}
.conf-bar-inner-neg {
    background: linear-gradient(90deg, #3db87a, #6bdfa4);
    height: 100%; border-radius: 99px;
}

.bundle-info {
    background: #12181f;
    border: 1px solid #1e3a2a;
    border-left: 3px solid #3db87a;
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    font-size: 0.82rem;
    color: #7aaa8a;
    margin-bottom: 1rem;
}

.stButton > button {
    background: #e05c5c;
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: #c94a4a;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(224, 92, 92, 0.3);
}

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: #16161f;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    color: #7a7585;
    font-size: 0.85rem;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #e05c5c22 !important;
    border-color: #e05c5c !important;
    color: #e05c5c !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Feature Engineering (must match notebook exactly) ─────────────────────────
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Identical to engineer_features() in the notebook.
    Must stay in sync whenever the notebook version changes.
    """
    X = X.copy()
    X['age_thalach'] = X['age'] * X['thalach']
    X['chol_age']    = X['chol'] * X['age']
    X['age_group']   = pd.cut(
        X['age'], bins=[0, 40, 55, 65, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    return X


# ─── Load bundle ───────────────────────────────────────────────────────────────
BUNDLE_PATH = "model/heart_disease_bundle.pkl"

@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    return joblib.load(BUNDLE_PATH)

bundle = load_bundle()

# ─── Guard: bundle missing ─────────────────────────────────────────────────────
if bundle is None:
    st.markdown("""
    <h1 class="hero-title">Heart Disease<br><span>Risk Predictor</span></h1>
    """, unsafe_allow_html=True)
    st.error(
        f"**`{BUNDLE_PATH}` not found.**\n\n"
        "Please run the notebook first to generate the model bundle, "
        f"then place `{BUNDLE_PATH}` in the same folder as `app.py`.",
        icon="🚨",
    )
    st.markdown("""
    ```
    your-project/
    ├── app.py
    ├── heart_disease_bundle.pkl   ← generated by the notebook
    └── heart-disease.csv
    ```
    **Steps:**
    1. Open `end-to-end-heart-disease-classification.ipynb`
    2. Run all cells (Kernel → Restart & Run All)
    3. The last cell saves `heart_disease_bundle.pkl`
    4. Move it next to `app.py` and restart Streamlit
    """)
    st.stop()

# ─── Unpack bundle ─────────────────────────────────────────────────────────────
models       = bundle["all_models"]          # dict of trained model/pipelines
cv_scores    = bundle["cv_results"]          # {name: {mean, std}}
metrics      = bundle["metrics"]             # {name: {precision, recall, f1, cm}}
feat_imp_raw = bundle["feature_importances"] # dict {feature: importance}
best_name    = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
feat_imp     = pd.Series(feat_imp_raw).sort_values(ascending=False)
sklearn_ver  = bundle.get("sklearn_version", "unknown")

# test_scores: use CV mean as the primary score (more reliable than single test split)
test_scores = {name: cv_scores[name]["mean"] for name in models}

# Load raw data for the Dataset tab (optional, graceful fallback)
@st.cache_data
def load_data():
    for path in ["heart-disease.csv", "heart_disease.csv", "data/heart-disease.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

df = load_data()


# ─── Sidebar ────────────────────────────────────────────────────────────────────
cp_map      = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
restecg_map = {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}
slope_map   = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
thal_map    = {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect", 3: "Unknown"}

with st.sidebar:
    st.markdown('<p class="section-label">Patient Parameters</p>', unsafe_allow_html=True)

    age      = st.slider("Age", 20, 85, 52)
    sex      = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp       = st.selectbox("Chest Pain Type", list(cp_map.keys()), format_func=lambda x: cp_map[x])
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, 130)
    chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg  = st.selectbox("Resting ECG Results", list(restecg_map.keys()), format_func=lambda x: restecg_map[x])
    thalach  = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope    = st.selectbox("Slope of Peak Exercise ST", list(slope_map.keys()), format_func=lambda x: slope_map[x])
    ca       = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", list(thal_map.keys()), format_func=lambda x: thal_map[x])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Model Selection</p>', unsafe_allow_html=True)

    default_idx = list(models.keys()).index(best_name) if best_name in models else 0
    selected_model = st.selectbox("Algorithm", list(models.keys()), index=default_idx)
    predict_btn = st.button("🫀 Run Prediction")


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 class="hero-title">Heart Disease<br><span>Risk Predictor</span></h1>
<p class="hero-sub">Trained in notebook · Loaded from bundle · No retraining on startup</p>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="bundle-info">
    ✅ Loaded <strong>{BUNDLE_PATH}</strong> &nbsp;·&nbsp;
    {len(models)} models &nbsp;·&nbsp;
    sklearn {sklearn_ver} &nbsp;·&nbsp;
    Best model: <strong>{best_name}</strong> ({cv_scores[best_name]['mean']}% CV)
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_data = st.tabs(["🫀 Prediction", "📈 Model Insights", "🗂 Dataset"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
with tab_predict:

    # ── Model Cards (via components.html to bypass Streamlit's HTML sanitizer) ──
    st.markdown('<p class="section-label">5-Fold CV Accuracy (from notebook)</p>', unsafe_allow_html=True)
    _cards = '<div style="display:flex;gap:0.75rem;margin-bottom:0.5rem;">'
    for _name, _cv in cv_scores.items():
        _is_best = (_name == best_name)
        _short   = (_name
                    .replace('Gradient Boosting', 'Grad. Boost')
                    .replace('Logistic Regression', 'Log. Reg.')
                    .replace('Voting Ensemble', 'Ensemble')
                    .replace('Tuned Ensemble', 'Tuned Ensemble'))
        _border  = '#e05c5c' if _is_best else '#2a2a38'
        _badge   = ('<div style="position:absolute;top:0.6rem;right:0.6rem;'
                    'background:#e05c5c;color:white;font-size:0.58rem;font-weight:700;'
                    'letter-spacing:0.07em;text-transform:uppercase;'
                    'padding:0.15rem 0.4rem;border-radius:99px;">BEST</div>') if _is_best else ''
        _cards  += (f'<div style="flex:1;background:#16161f;border:1px solid {_border};'
                    f'border-radius:14px;padding:1.1rem 1.3rem;position:relative;min-width:0;">'
                    f'{_badge}'
                    f'<div style="font-size:1.85rem;font-weight:700;color:#f5f0ea;'
                    f'margin:0;line-height:1.1;">{_cv["mean"]}%</div>'
                    f'<div style="font-size:0.76rem;color:#3db87a;margin:0.18rem 0 0;">'
                    f'&plusmn;{_cv["std"]}% std</div>'
                    f'<div style="color:#7a7585;font-size:0.7rem;margin:0.22rem 0 0;'
                    f'text-transform:uppercase;letter-spacing:0.06em;">{_short}</div>'
                    f'</div>')
    _cards += '</div>'
    components.html(
        f'<html><head><style>*{{margin:0;padding:0;box-sizing:border-box;'
        f'font-family:"DM Sans",system-ui,sans-serif;}}</style></head>'
        f'<body style="background:transparent;overflow:hidden;">{_cards}</body></html>',
        height=115, scrolling=False,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_result, col_info = st.columns([1, 1])

    with col_result:
        st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            # Build input dataframe with same column names as training
            raw_input = pd.DataFrame(
                [[age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]],
                columns=["age","sex","cp","trestbps","chol","fbs",
                         "restecg","thalach","exang","oldpeak","slope","ca","thal"]
            )
            input_fe = engineer_features(raw_input)

            # Align columns to what the model was trained on
            expected_cols = bundle.get("feature_names", list(input_fe.columns))
            input_fe = input_fe.reindex(columns=expected_cols, fill_value=0)

            model      = models[selected_model]
            prediction = model.predict(input_fe)[0]
            proba      = model.predict_proba(input_fe)[0]
            confidence = round(proba[prediction] * 100, 1)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <p class="result-title">⚠️ Heart Disease Detected</p>
                    <p class="result-sub">Confidence: <strong>{confidence}%</strong> · {selected_model}</p>
                    <div class="conf-bar-outer">
                        <div class="conf-bar-inner-pos" style="width:{confidence}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <p class="result-title">✅ No Heart Disease</p>
                    <p class="result-sub">Confidence: <strong>{confidence}%</strong> · {selected_model}</p>
                    <div class="conf-bar-outer">
                        <div class="conf-bar-inner-neg" style="width:{confidence}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">All Models Comparison</p>', unsafe_allow_html=True)
            for name, m in models.items():
                p   = m.predict(input_fe)[0]
                pr  = round(m.predict_proba(input_fe)[0][p] * 100, 1)
                lbl = "Disease" if p == 1 else "No Disease"
                clr = "#e05c5c" if p == 1 else "#3db87a"
                sel_style = "border-color:#e05c5c44;" if name == selected_model else ""
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.65rem 1rem;background:#16161f;border-radius:10px;
                            margin-bottom:0.45rem;border:1px solid #2a2a38;{sel_style}">
                    <span style="color:#a09898;font-size:0.82rem;">{name}</span>
                    <span style="color:{clr};font-weight:600;font-size:0.82rem;">{lbl} ({pr}%)</span>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#16161f;border:1px dashed #2a2a38;border-radius:16px;
                        padding:3rem 2rem;text-align:center;">
                <p style="font-size:2.5rem;margin:0;">🫀</p>
                <p style="color:#7a7585;font-size:0.9rem;margin-top:1rem;">
                    Adjust parameters in the sidebar<br>
                    and click <strong style="color:#e05c5c;">Run Prediction</strong>
                </p>
            </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown('<p class="section-label">Patient Summary</p>', unsafe_allow_html=True)
        items = [
            ("Age", f"{age} yrs"),
            ("Sex", "Male" if sex else "Female"),
            ("Chest Pain", cp_map[cp]),
            ("Blood Pressure", f"{trestbps} mmHg"),
            ("Cholesterol", f"{chol} mg/dl"),
            ("Max Heart Rate", f"{thalach} bpm"),
            ("ST Depression", f"{oldpeak}"),
            ("Fasting Blood Sugar", "High" if fbs else "Normal"),
            ("Exercise Angina", "Yes" if exang else "No"),
            ("Thalassemia", thal_map[thal]),
        ]
        for label, value in items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.6rem 0;
                        border-bottom:1px solid #1e1e2a;">
                <span style="color:#7a7585;font-size:0.8rem;text-transform:uppercase;
                             letter-spacing:0.06em;">{label}</span>
                <span style="color:#e8e4df;font-size:0.88rem;font-weight:500;">{value}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Auto Risk Flags</p>', unsafe_allow_html=True)
        flags = []
        if age > 55:        flags.append(("🔴", "Age > 55"))
        if trestbps > 140:  flags.append(("🔴", f"High BP ({trestbps} mmHg)"))
        if chol > 240:      flags.append(("🟠", f"High Cholesterol ({chol})"))
        if thalach < 100:   flags.append(("🟠", "Low Max Heart Rate"))
        if exang == 1:      flags.append(("🔴", "Exercise-Induced Angina"))
        if oldpeak > 2.0:   flags.append(("🟠", f"High ST Depression ({oldpeak})"))
        if ca > 0:          flags.append(("🔴", f"{ca} Major Vessel(s) Blocked"))

        if flags:
            for icon, msg in flags:
                st.markdown(f"""
                <div style="padding:0.45rem 0.8rem;background:#1e1a1a;border-radius:8px;
                            margin-bottom:0.35rem;font-size:0.83rem;color:#c9b8b8;">
                    {icon} {msg}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#3db87a;font-size:0.85rem;">✅ No major risk flags detected</p>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════
with tab_insights:
    c1, c2 = st.columns(2)

    # ── CV vs Test Accuracy ───────────────────────────────────────────
    with c1:
        st.markdown('<p class="section-label">5-Fold CV Accuracy by Model</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#16161f")
        ax.set_facecolor("#16161f")

        names = list(cv_scores.keys())
        short_names = [
            n.replace("Gradient Boosting","Grad.\nBoost")
             .replace("Logistic Regression","Log.\nReg.")
             .replace("Voting Ensemble","Ensemble")
             .replace("Tuned Ensemble","Tuned\nEnsemble")
             .replace("Random Forest","Rand.\nForest")
            for n in names
        ]
        cv_means = [cv_scores[n]["mean"] for n in names]
        cv_stds  = [cv_scores[n]["std"]  for n in names]
        bar_colors = ["#e05c5c" if n == best_name else "#5b8dee" for n in names]

        bars = ax.bar(
            range(len(names)), cv_means, yerr=cv_stds,
            color=bar_colors, alpha=0.85, capsize=5,
            error_kw={"color": "#aaaaaa", "linewidth": 1.5},
            edgecolor="white", linewidth=0.4,
        )
        for bar, mean in zip(bars, cv_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                    f"{mean}%", ha="center", va="bottom", color="#e8e4df", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, color="#a09898", fontsize=8)
        ax.set_ylim(50, 108)
        ax.set_ylabel("CV Accuracy (%)", color="#7a7585", fontsize=9)
        ax.tick_params(colors="#7a7585")
        ax.spines[:].set_color("#2a2a38")
        st.pyplot(fig)
        plt.close()

    # ── Feature Importances ───────────────────────────────────────────
    with c2:
        st.markdown('<p class="section-label">Top Feature Importances (Random Forest)</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#16161f")
        ax2.set_facecolor("#16161f")

        top_n  = feat_imp.head(10)
        colors = ["#e05c5c" if i < 3 else "#7a90c4" for i in range(len(top_n))]
        ax2.barh(top_n.index[::-1], top_n.values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.3)
        ax2.set_xlabel("Importance", color="#7a7585", fontsize=9)
        ax2.tick_params(colors="#a09898", labelsize=8)
        ax2.spines[:].set_color("#2a2a38")
        st.pyplot(fig2)
        plt.close()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    # ── Precision / Recall / F1 ───────────────────────────────────────
    with c3:
        st.markdown('<p class="section-label">Precision · Recall · F1 by Model</p>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        fig3.patch.set_facecolor("#16161f")
        ax3.set_facecolor("#16161f")

        metric_keys = ["precision", "recall", "f1"]
        pal = ["#e05c5c", "#f4a261", "#3db87a"]
        x3  = np.arange(len(models))
        w3  = 0.25

        for i, (met, col) in enumerate(zip(metric_keys, pal)):
            vals = [metrics[n][met] for n in models]
            ax3.bar(x3 + (i - 1) * w3, vals, w3, label=met.capitalize(),
                    color=col, alpha=0.85, edgecolor="white", linewidth=0.3)

        ax3.set_xticks(x3)
        ax3.set_xticklabels(short_names, color="#a09898", fontsize=8)
        ax3.set_ylim(40, 115)
        ax3.set_ylabel("%", color="#7a7585", fontsize=9)
        ax3.tick_params(colors="#7a7585")
        ax3.spines[:].set_color("#2a2a38")
        ax3.legend(fontsize=8, labelcolor="#a09898", facecolor="#1e1e2e", edgecolor="#2a2a38")
        st.pyplot(fig3)
        plt.close()

    # ── Confusion Matrix ──────────────────────────────────────────────
    with c4:
        st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
        sel = st.selectbox(
            "Select model", list(models.keys()), key="cm_select",
            index=list(models.keys()).index(best_name),
        )
        cm = np.array(metrics[sel]["cm"])
        fig4, ax4 = plt.subplots(figsize=(4, 3.5))
        fig4.patch.set_facecolor("#16161f")
        ax4.set_facecolor("#16161f")

        ax4.imshow(cm, cmap="RdYlGn", alpha=0.75)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                         ha="center", va="center",
                         color="white", fontsize=13, fontweight="bold")
        ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["Pred: No Disease", "Pred: Disease"], color="#a09898", fontsize=8)
        ax4.set_yticklabels(["True: No Disease", "True: Disease"], color="#a09898", fontsize=8)
        ax4.spines[:].set_color("#2a2a38")
        st.pyplot(fig4)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ══════════════════════════════════════════════════════════════════════
with tab_data:
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-label">Sample Data</p>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown('<p class="section-label">Statistics</p>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(2), use_container_width=True)

        vc   = df["target"].value_counts()
        pct0 = round(vc.get(0, 0) / len(df) * 100, 1)
        pct1 = round(vc.get(1, 0) / len(df) * 100, 1)
        st.markdown('<p class="section-label" style="margin-top:1rem;">Class Distribution</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;gap:1rem;">
            <div style="background:#16161f;border:1px solid #2a2a38;border-radius:14px;padding:1.2rem 1.4rem;flex:1;text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:2rem;margin:0;color:#3db87a">{vc.get(0,0)}</div>
                <div style="color:#7a7585;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:0.1rem 0 0;">No Disease ({pct0}%)</div>
            </div>
            <div style="background:#16161f;border:1px solid #2a2a38;border-radius:14px;padding:1.2rem 1.4rem;flex:1;text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:2rem;margin:0;color:#e05c5c">{vc.get(1,0)}</div>
                <div style="color:#7a7585;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:0.1rem 0 0;">Disease ({pct1}%)</div>
            </div>
            <div style="background:#16161f;border:1px solid #2a2a38;border-radius:14px;padding:1.2rem 1.4rem;flex:1;text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:2rem;margin:0;color:#f5f0ea">{len(df)}</div>
                <div style="color:#7a7585;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:0.1rem 0 0;">Total Samples</div>
            </div>
            <div style="background:#16161f;border:1px solid #2a2a38;border-radius:14px;padding:1.2rem 1.4rem;flex:1;text-align:center;">
                <div style="font-family:'DM Serif Display',serif;font-size:2rem;margin:0;color:#f5f0ea">{df.shape[1]-1}</div>
                <div style="color:#7a7585;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em;margin:0.1rem 0 0;">Features</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info(
            "`heart-disease.csv` not found in the app directory. "
            "The Dataset tab requires the raw CSV alongside `app.py`.",
            icon="ℹ️"
        )


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#3a3a4a;font-size:0.78rem;padding:2rem 0 1rem;">
    Built with Streamlit · Cleveland UCI Heart Disease Dataset · For educational purposes only<br>
    <strong style="color:#4a4a5a;">Not a substitute for professional medical advice.</strong>
</div>
""", unsafe_allow_html=True)