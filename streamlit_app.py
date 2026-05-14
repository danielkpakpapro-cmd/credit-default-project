"""
Interface Streamlit - Prédiction Défaut de Prêt Bancaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# ─── Configuration de la page ───────────────────────────────
st.set_page_config(
    page_title=" Prédiction Défaut de Prêt",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a237e;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e, #283593);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.3rem;
    }
    .risk-low    { background: linear-gradient(135deg, #2e7d32, #388e3c); color:white; padding:1.5rem; border-radius:16px; text-align:center; font-size:1.3rem; font-weight:bold; }
    .risk-medium { background: linear-gradient(135deg, #f57f17, #f9a825); color:white; padding:1.5rem; border-radius:16px; text-align:center; font-size:1.3rem; font-weight:bold; }
    .risk-high   { background: linear-gradient(135deg, #e65100, #ef6c00); color:white; padding:1.5rem; border-radius:16px; text-align:center; font-size:1.3rem; font-weight:bold; }
    .risk-critical { background: linear-gradient(135deg, #b71c1c, #c62828); color:white; padding:1.5rem; border-radius:16px; text-align:center; font-size:1.3rem; font-weight:bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# ─── Chargement des modèles ──────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    model         = joblib.load(BASE_DIR / "models" / "best_model.pkl")
    scaler        = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")
    model_name    = joblib.load(BASE_DIR / "models" / "best_model_name.pkl")
    return model, scaler, feature_names, model_name

@st.cache_data
def load_data():
    df = pd.read_excel(BASE_DIR / "data" / "Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data")
    df = df.drop(columns=["ID", "ZIP Code"])
    df["Experience"] = df["Experience"].clip(lower=0)
    return df

model, scaler, feature_names, model_name = load_models()
df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=80)
    st.markdown("##  Navigation")
    page = st.radio("", [" Prédiction", " Analyse du Dataset", " Comparaison Modèles", " À propos"])
    st.markdown("---")
    st.markdown(f"**Modèle actif :** `{model_name}`")
    st.markdown(f"**Dataset :** `{len(df)} clients`")
    st.markdown(f"**Features :** `{len(feature_names)}`")

# ═══════════════════════════════════════════════════════════════
# PAGE 1 : PRÉDICTION
# ═══════════════════════════════════════════════════════════════
if page == " Prédiction":
    st.markdown('<div class="main-title"> Prédiction de Défaut de Prêt</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Entrez les informations du client pour évaluer son risque de défaut</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("####  Profil Client")
        age        = st.slider("Âge", 18, 70, 35)
        experience = st.slider("Expérience (années)", 0, 50, 10)
        education  = st.selectbox("Niveau d'éducation", [1, 2, 3],
                                   format_func=lambda x: {1: "🎓 Lycée", 2: "🎓 Licence", 3: "🎓 Master+"}[x])
        family     = st.selectbox("Taille de la famille", [1, 2, 3, 4],
                                   format_func=lambda x: f" {x} personne(s)")

    with col2:
        st.markdown("####  Situation Financière")
        income   = st.number_input("Revenu annuel (milliers $)", 0, 500, 80, step=5)
        ccavg    = st.number_input("Dépenses CB/mois (milliers $)", 0.0, 20.0, 2.5, step=0.1)
        mortgage = st.number_input("Prêt immobilier (milliers $)", 0, 700, 0, step=10)

    with col3:
        st.markdown("####  Produits Bancaires")
        securities = st.toggle(" Compte Titres")
        cd_account = st.toggle(" Compte Dépôt à Terme")
        online     = st.toggle(" Banque en Ligne", value=True)
        creditcard = st.toggle(" Carte de Crédit Banque")

    st.markdown("---")

    if st.button(" Prédire le Risque", use_container_width=True, type="primary"):
        input_dict = {
            "Age": age, "Experience": experience, "Income": income,
            "Family": family, "CCAvg": ccavg, "Education": education,
            "Mortgage": mortgage, "Securities Account": int(securities),
            "CD Account": int(cd_account), "Online": int(online),
            "CreditCard": int(creditcard),
        }
        features        = np.array([[input_dict[f] for f in feature_names]])
        features_scaled = scaler.transform(features)
        pred            = int(model.predict(features_scaled)[0])
        proba           = model.predict_proba(features_scaled)[0]
        prob_default    = float(proba[1])

        st.markdown("### 📋 Résultat de l'analyse")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Décision", "❌ Défaut probable" if pred == 1 else "✅ Pas de défaut")
        r2.metric("Probabilité de défaut", f"{prob_default*100:.1f}%")
        r3.metric("Probabilité de remboursement", f"{(1-prob_default)*100:.1f}%")
        r4.metric("Modèle utilisé", model_name)

        # Niveau de risque
        if prob_default < 0.2:
            st.markdown('<div class="risk-low"> RISQUE FAIBLE — Prêt recommandé</div>', unsafe_allow_html=True)
        elif prob_default < 0.5:
            st.markdown('<div class="risk-medium"> RISQUE MODÉRÉ — Analyse approfondie conseillée</div>', unsafe_allow_html=True)
        elif prob_default < 0.75:
            st.markdown('<div class="risk-high"> RISQUE ÉLEVÉ — Garanties supplémentaires requises</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-critical"> RISQUE TRÈS ÉLEVÉ — Prêt déconseillé</div>', unsafe_allow_html=True)

        # Jauge de probabilité
        st.markdown("#### Probabilité de défaut")
        fig, ax = plt.subplots(figsize=(8, 1.2))
        color = "#2e7d32" if prob_default < 0.2 else "#f57f17" if prob_default < 0.5 else "#e65100" if prob_default < 0.75 else "#b71c1c"
        ax.barh(0, prob_default, color=color, height=0.5)
        ax.barh(0, 1 - prob_default, left=prob_default, color="#e0e0e0", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(f"Probabilité de défaut : {prob_default*100:.1f}%")
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE 2 : ANALYSE DU DATASET
# ═══════════════════════════════════════════════════════════════
elif page == " Analyse du Dataset":
    st.markdown('<div class="main-title"> Analyse Exploratoire du Dataset</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(" Total clients", f"{len(df):,}")
    k2.metric(" Sans défaut", f"{(df['Personal Loan']==0).sum():,}")
    k3.metric(" Avec défaut", f"{(df['Personal Loan']==1).sum():,}")
    k4.metric(" Taux de défaut", f"{df['Personal Loan'].mean()*100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([" Distributions", " Corrélations", " Données brutes"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            df['Personal Loan'].value_counts().plot(kind='bar', color=['#2196F3', '#F44336'], ax=ax)
            ax.set_xticklabels(['Pas de défaut', 'Défaut'], rotation=0)
            ax.set_title("Distribution de la cible")
            ax.set_ylabel("Nombre de clients")
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[df['Personal Loan']==0]['Income'], bins=30, alpha=0.6, label='Pas de défaut', color='#2196F3')
            ax.hist(df[df['Personal Loan']==1]['Income'], bins=30, alpha=0.6, label='Défaut', color='#F44336')
            ax.set_title("Distribution du revenu par classe")
            ax.set_xlabel("Revenu (milliers $)")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(['Education', 'Personal Loan']).size().unstack().plot(
                kind='bar', ax=ax, color=['#2196F3', '#F44336'])
            ax.set_xticklabels(['Lycée', 'Licence', 'Master+'], rotation=0)
            ax.set_title("Défaut par niveau d'éducation")
            ax.legend(['Pas de défaut', 'Défaut'])
            st.pyplot(fig)
            plt.close()

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            df.groupby(['Family', 'Personal Loan']).size().unstack().plot(
                kind='bar', ax=ax, color=['#2196F3', '#F44336'])
            ax.set_title("Défaut par taille de famille")
            ax.legend(['Pas de défaut', 'Défaut'])
            ax.set_xlabel("Taille de la famille")
            st.pyplot(fig)
            plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 7))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax, center=0, square=True, linewidths=0.5)
        ax.set_title("Matrice de corrélation", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Affichage des 100 premières lignes")

# ═══════════════════════════════════════════════════════════════
# PAGE 3 : COMPARAISON MODÈLES
# ═══════════════════════════════════════════════════════════════
elif page == " Comparaison Modèles":
    st.markdown('<div class="main-title"> Comparaison des Modèles</div>', unsafe_allow_html=True)

    results = pd.DataFrame({
        "Modèle": ["XGBoost", "LightGBM", "Random Forest", "Gradient Boosting", "SVM", "KNN", "Logistic Regression", "Naive Bayes"],
        "Accuracy": [0.991, 0.993, 0.987, 0.979, 0.971, 0.954, 0.908, 0.873],
        "Precision": [0.9485, 0.9588, 0.9029, 0.8505, 0.7815, 0.6923, 0.5116, 0.4197],
        "Recall": [0.9583, 0.9688, 0.9688, 0.9479, 0.9688, 0.9375, 0.9167, 0.8438],
        "F1-Score": [0.9534, 0.9637, 0.9347, 0.8966, 0.8651, 0.7965, 0.6567, 0.5606],
        "AUC-ROC": [0.9990, 0.9988, 0.9986, 0.9985, 0.9936, 0.9655, 0.9655, 0.9262],
    })

    st.dataframe(
        results.style.highlight_max(subset=["Accuracy", "F1-Score", "AUC-ROC"], color="#c8e6c9")
                      .highlight_min(subset=["Accuracy", "F1-Score", "AUC-ROC"], color="#ffcdd2")
                      .format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}",
                               "F1-Score": "{:.4f}", "AUC-ROC": "{:.4f}"}),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1a237e' if m == model_name else '#90caf9' for m in results['Modèle']]
        bars = ax.barh(results['Modèle'][::-1], results['AUC-ROC'][::-1], color=colors[::-1])
        ax.set_xlim(0.88, 1.01)
        ax.set_title("AUC-ROC par Modèle", fontweight='bold')
        ax.set_xlabel("AUC-ROC")
        for bar, val in zip(bars, results['AUC-ROC'][::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors2 = ['#1a237e' if m == model_name else '#90caf9' for m in results['Modèle']]
        bars2 = ax.barh(results['Modèle'][::-1], results['F1-Score'][::-1], color=colors2[::-1])
        ax.set_xlim(0.4, 1.05)
        ax.set_title("F1-Score par Modèle", fontweight='bold')
        ax.set_xlabel("F1-Score")
        for bar, val in zip(bars2, results['F1-Score'][::-1]):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        st.pyplot(fig)
        plt.close()

    # Afficher les images générées
    st.markdown("---")
    st.markdown("###  Visualisations générées lors de l'entraînement")
    img_col1, img_col2 = st.columns(2)
    outputs = BASE_DIR / "outputs"
    if (outputs / "roc_curves.png").exists():
        with img_col1:
            st.image(str(outputs / "roc_curves.png"), caption="Courbes ROC", use_container_width=True)
    if (outputs / "confusion_matrix.png").exists():
        with img_col2:
            st.image(str(outputs / "confusion_matrix.png"), caption="Matrice de Confusion", use_container_width=True)
    if (outputs / "feature_importance.png").exists():
        st.image(str(outputs / "feature_importance.png"), caption="Importance des Features", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 : À PROPOS
# ═══════════════════════════════════════════════════════════════
elif page == " À propos":
    st.markdown('<div class="main-title"> À propos du Projet</div>', unsafe_allow_html=True)

    st.markdown("""
    ##  Objectif
    Ce projet prédit la probabilité qu'un client fasse **défaut sur son prêt bancaire**, en comparant 8 modèles de classification et en déployant le meilleur via une API FastAPI et une interface Streamlit.

    ##  Dataset
    - **Source** : [Kaggle - Bank Personal Loan Modelling](https://www.kaggle.com/datasets/itsmesunil/bank-loanmodelling)
    - **5 000 clients**, 14 variables
    - **9.6% de défauts** (dataset déséquilibré → SMOTE appliqué)

    ##  Pipeline
    1. **Preprocessing** : nettoyage, SMOTE, StandardScaler
    2. **Entraînement** : 8 modèles comparés avec cross-validation 5 folds
    3. **Évaluation** : Accuracy, Precision, Recall, F1-Score, AUC-ROC
    4. **Déploiement** : API FastAPI + Interface Streamlit + Docker

    ##  Résultats
    | Modèle | AUC-ROC |
    |--------|---------|
    | **XGBoost**  | **0.999** |
    | LightGBM | 0.999 |
    | Random Forest | 0.999 |

    ##  Technologies
    `Python` `Scikit-learn` `XGBoost` `LightGBM` `FastAPI` `Streamlit` `Docker` `SMOTE`
    """)
