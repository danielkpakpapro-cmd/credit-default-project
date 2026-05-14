# 🏦 Projet 5 - Prédiction de Défaut de Prêt Bancaire

## 📋 Description

Ce projet prédit la probabilité qu'un client fasse défaut sur son prêt bancaire, en comparant **8 modèles de classification** et en déployant le meilleur via une API FastAPI containerisée avec Docker.

---

## 📁 Structure du projet

```
credit-default-project/
├── data/
│   └── Bank_Personal_Loan_Modelling.xlsx   # Dataset Kaggle
├── models/
│   ├── best_model.pkl                       # Meilleur modèle entraîné
│   ├── scaler.pkl                           # Normalisation
│   └── feature_names.pkl                   # Noms des features
├── app/
│   └── main.py                             # API FastAPI
├── outputs/
│   ├── comparaison_modeles.png             # Graphique comparaison
│   ├── roc_curves.png                      # Courbes ROC
│   ├── confusion_matrix.png                # Matrice de confusion
│   ├── feature_importance.png              # Importance des features
│   └── resultats_modeles.csv               # Tableau des résultats
├── train.py                                # Script d'entraînement
├── Dockerfile                              # Image Docker
├── docker-compose.yml                      # Orchestration
├── requirements.txt                        # Dépendances Python
└── README.md
```

---

## 🚀 Lancement rapide

### 1. Entraîner les modèles

```bash
pip install -r requirements.txt
python train.py
```

### 2. Lancer l'API en local

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Avec Docker

```bash
# Build + run
docker-compose up --build

# En arrière-plan
docker-compose up -d --build
```

L'API est disponible sur : http://localhost:8000

---

## 📊 Modèles comparés

| Modèle | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| LightGBM | 99.3% | 0.9637 | 0.9988 |
| XGBoost | 99.1% | 0.9534 | 0.9990 |
| Random Forest | 98.7% | 0.9347 | 0.9986 |
| Gradient Boosting | 97.9% | 0.8966 | 0.9985 |
| SVM | 97.1% | 0.8651 | 0.9936 |
| KNN | 95.4% | 0.7965 | 0.9655 |
| Logistic Regression | 90.8% | 0.6567 | 0.9655 |
| Naive Bayes | 87.3% | 0.5606 | 0.9262 |

**🏆 Meilleur modèle : XGBoost (AUC-ROC = 0.999)**

---

## 🧩 Utilité et limites de chaque modèle

### 1. Régression Logistique
**Utilité** : Modèle de référence simple et rapide. Très utilisé en banque car il produit des résultats facilement interprétables et explicables aux régulateurs. Idéal pour comprendre l'impact de chaque variable sur le risque de défaut.

**Limites** : Suppose une relation linéaire entre les features et la cible. Peu performant quand les données ont des interactions complexes ou non-linéaires. Dans notre cas, AUC = 0.966, correct mais loin des meilleurs modèles.

---

### 2. Support Vector Machine (SVM)
**Utilité** : Efficace pour trouver la frontière optimale entre les classes, même dans des espaces de haute dimension. Bon choix quand les données sont bien séparables.

**Limites** : Très lent à entraîner sur de grands datasets. Difficile à interpréter (boîte noire). Sensible au choix du kernel et des hyperparamètres. Peu adapté à la production à grande échelle.

---

### 3. K-Nearest Neighbors (KNN)
**Utilité** : Algorithme intuitif basé sur la similarité entre clients. Utile pour comprendre les patterns locaux dans les données. Aucun entraînement nécessaire (lazy learning).

**Limites** : Très lent à la prédiction sur de grands datasets car il compare chaque nouveau client à tous les clients existants. Sensible aux données bruitées et à l'échelle des features. Performances moyennes (AUC = 0.966).

---

### 4. Naive Bayes
**Utilité** : Modèle probabiliste ultra-rapide, idéal pour les systèmes en temps réel avec peu de ressources. Fonctionne bien avec peu de données.

**Limites** : Suppose que toutes les features sont indépendantes entre elles, ce qui est rarement vrai en pratique. C'est son point faible principal. Dans notre projet, c'est le modèle le moins performant (AUC = 0.926) car les variables bancaires sont corrélées.

---

### 5. Random Forest
**Utilité** : Ensemble de centaines d'arbres de décision. Très robuste, résistant au surapprentissage, et fournit une mesure d'importance des features très utile pour comprendre quelles variables influencent le plus le défaut.

**Limites** : Modèle lourd en mémoire et plus lent que XGBoost/LightGBM. Moins performant sur des données très déséquilibrées sans rééchantillonnage. Difficile à interpréter individuellement malgré sa robustesse.

---

### 6. Gradient Boosting (GBM)
**Utilité** : Construit les arbres de décision séquentiellement, chaque arbre corrigeant les erreurs du précédent. Très précis sur des données tabulaires complexes comme les données bancaires.

**Limites** : Entraînement lent car les arbres sont construits un par un. Très sensible aux hyperparamètres (learning rate, profondeur). Risque de surapprentissage si mal configuré.

---

### 7. XGBoost 🏆
**Utilité** : Version optimisée du Gradient Boosting, beaucoup plus rapide grâce à la parallélisation. Standard de l'industrie pour les compétitions ML et la production. Gère nativement les valeurs manquantes et les données déséquilibrées. **Meilleur modèle de ce projet (AUC = 0.999).**

**Limites** : Nombreux hyperparamètres à tuner. Peut être une boîte noire difficile à expliquer aux non-techniciens. Nécessite plus de ressources que les modèles simples.

---

### 8. LightGBM
**Utilité** : Variante de XGBoost encore plus rapide, optimisée pour les très grands datasets. Consomme moins de mémoire grâce à sa technique de construction par feuilles (leaf-wise). Excellentes performances (AUC = 0.999).

**Limites** : Peut surapprendre sur les petits datasets à cause de la croissance leaf-wise. Moins stable que XGBoost sur certains types de données. Les hyperparamètres sont encore plus nombreux et complexes à optimiser.

---

## 🔌 Utilisation de l'API

### Prédiction simple

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "Experience": 10,
    "Income": 80,
    "Family": 3,
    "CCAvg": 2.5,
    "Education": 2,
    "Mortgage": 0,
    "Securities_Account": 0,
    "CD_Account": 0,
    "Online": 1,
    "CreditCard": 0
  }'
```

### Réponse

```json
{
  "prediction": 0,
  "prediction_label": "Pas de défaut",
  "probability_default": 0.023,
  "probability_no_default": 0.977,
  "risk_level": "🟢 Faible",
  "model_used": "XGBoost"
}
```

### Documentation interactive

Accéder à : **http://localhost:8000/docs** (Swagger UI)

---

## 🧠 Features utilisées

| Feature | Description |
|---------|-------------|
| Age | Âge du client |
| Experience | Années d'expérience professionnelle |
| Income | Revenu annuel (en milliers $) |
| Family | Taille de la famille |
| CCAvg | Dépenses mensuelles carte de crédit |
| Education | Niveau d'éducation (1=Lycée, 2=Licence, 3=Master+) |
| Mortgage | Montant du prêt immobilier |
| Securities Account | Possède un compte titres |
| CD Account | Possède un compte dépôt à terme |
| Online | Utilise la banque en ligne |
| CreditCard | Possède une carte de crédit de la banque |

---

## ⚙️ Techniques appliquées

- **SMOTE** : Sur-échantillonnage pour gérer le déséquilibre des classes (9.6% de défauts)
- **StandardScaler** : Normalisation des features
- **Stratified K-Fold (5 folds)** : Validation croisée robuste
- **Métriques** : Accuracy, Precision, Recall, F1-Score, AUC-ROC

---

## 🐳 Docker

```bash
# Build
docker build -t credit-default-api .

# Run
docker run -p 8000:8000 credit-default-api

# Avec docker-compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```
