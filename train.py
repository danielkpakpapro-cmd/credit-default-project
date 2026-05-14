"""
Projet 5 - Modélisation Prédictive : Défaut de Prêt Bancaire
Comparaison de 7 modèles de classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────
# 1. CHARGEMENT & EXPLORATION
# ─────────────────────────────────────────
print("=" * 60)
print("  PROJET 5 - DÉFAUT DE PRÊT BANCAIRE")
print("=" * 60)

DATA_PATH = "data/Bank_Personal_Loan_Modelling.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Data")

print(f"\n Dataset shape : {df.shape}")
print(f" Colonnes : {list(df.columns)}")
print(f"\nTarget distribution:\n{df['Personal Loan'].value_counts()}")
print(f"\nProportion défaut : {df['Personal Loan'].mean()*100:.1f}%")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  PREPROCESSING")
print("="*60)

# Supprimer colonnes inutiles
df = df.drop(columns=["ID", "ZIP Code"])

# Gérer les valeurs négatives dans Experience
print(f"Experience < 0 : {(df['Experience'] < 0).sum()} lignes → remplacées par 0")
df['Experience'] = df['Experience'].clip(lower=0)

# Features et target
X = df.drop(columns=["Personal Loan"])
y = df["Personal Loan"]

print(f"\nFeatures utilisées : {list(X.columns)}")
print(f"Taille X : {X.shape}, y : {y.shape}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE pour rééquilibrer les classes
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAprès SMOTE → Class 0: {sum(y_train_sm==0)}, Class 1: {sum(y_train_sm==1)}")

# Sauvegarder le scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

# ─────────────────────────────────────────
# 3. DÉFINITION DES MODÈLES
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric='logloss', verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

# ─────────────────────────────────────────
# 4. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────
print("\n" + "="*60)
print("  ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("="*60)

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n Entraînement : {name}...")
    model.fit(X_train_sm, y_train_sm)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Cross-validation AUC
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm,
                                 cv=cv, scoring='roc_auc')
    
    results.append({
        "Modèle": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
        "CV-AUC (mean)": round(cv_scores.mean(), 4),
        "CV-AUC (std)": round(cv_scores.std(), 4),
    })
    
    print(f"    Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ─────────────────────────────────────────
# 5. TABLEAU DE COMPARAISON
# ─────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
print("\n" + "="*60)
print("  COMPARAISON DES MODÈLES (trié par AUC-ROC)")
print("="*60)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────
# 6. MEILLEUR MODÈLE
# ─────────────────────────────────────────
best_model_name = results_df.iloc[0]["Modèle"]
best_model = models[best_model_name]
print(f"\n Meilleur modèle : {best_model_name}")
print(f"   AUC-ROC = {results_df.iloc[0]['AUC-ROC']}")

# Sauvegarder le meilleur modèle
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(best_model_name, "models/best_model_name.pkl")
print(f"    Modèle sauvegardé dans models/best_model.pkl")

# ─────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
plt.style.use('seaborn-v0_8')
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']

# --- Figure 1 : Comparaison des métriques ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Comparaison des Modèles de Classification - Défaut de Prêt", fontsize=14, fontweight='bold')

metrics = ["Accuracy", "F1-Score", "AUC-ROC"]
x = np.arange(len(results_df))
width = 0.25

for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, results_df[metric], width, label=metric, color=colors[i], alpha=0.85)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(results_df["Modèle"], rotation=30, ha='right', fontsize=9)
axes[0].set_ylim(0.7, 1.02)
axes[0].set_title("Métriques par Modèle")
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# AUC-ROC bar chart
bars = axes[1].barh(results_df["Modèle"][::-1], results_df["AUC-ROC"][::-1], color=colors[:len(results_df)], alpha=0.85)
axes[1].set_xlim(0.85, 1.0)
axes[1].set_title("AUC-ROC par Modèle (meilleur en haut)")
axes[1].grid(axis='x', alpha=0.3)
for bar, val in zip(bars, results_df["AUC-ROC"][::-1]):
    axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("outputs/comparaison_modeles.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n Figure 1 sauvegardée : outputs/comparaison_modeles.png")

# --- Figure 2 : ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 7))
for i, (name, model) in enumerate(models.items()):
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
            label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Courbes ROC - Tous les Modèles")
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print(" Figure 2 sauvegardée : outputs/roc_curves.png")

# --- Figure 3 : Confusion Matrix du meilleur modèle ---
y_pred_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pas de Défaut', 'Défaut'],
            yticklabels=['Pas de Défaut', 'Défaut'])
ax.set_title(f"Matrice de Confusion - {best_model_name}", fontsize=13, fontweight='bold')
ax.set_ylabel("Valeur Réelle")
ax.set_xlabel("Valeur Prédite")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(" Figure 3 sauvegardée : outputs/confusion_matrix.png")

# --- Figure 4 : Feature Importance (si Random Forest ou XGBoost) ---
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot(kind='barh', color='#2196F3', ax=ax, alpha=0.85)
    ax.set_title(f"Importance des Features - {best_model_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 Figure 4 sauvegardée : outputs/feature_importance.png")

# --- Sauvegarder les résultats ---
results_df.to_csv("outputs/resultats_modeles.csv", index=False)
print("\n Résultats sauvegardés dans outputs/resultats_modeles.csv")

print("\n" + "="*60)
print("  ENTRAÎNEMENT TERMINÉ !")
print(f"   Meilleur modèle : {best_model_name}")
print(f"   Modèle sauvegardé : models/best_model.pkl")
print("="*60)
