# Salma Fadli G2 Finance
#  Breast Cancer Wisconsin (Diagnostic) Data Set
![photo de salma fadli.jpeg](https://github.com/fadlisalmaencg-a11y/DS-2025/blob/main/photo%20de%20salma%20fadli.jpeg?raw=true)
# Compte Rendu - Projet de Classification : Breast Cancer Wisconsin (Diagnostic)

## 1. Contexte Métier

### 1.1 Problématique

Le cancer du sein est l'un des cancers les plus répandus chez les femmes dans le monde. Un diagnostic précoce et précis est crucial pour améliorer les chances de survie des patientes. Ce projet vise à développer un modèle de machine learning capable de classifier automatiquement des tumeurs mammaires comme **bénignes (B)** ou **malignes (M)** à partir de caractéristiques cellulaires extraites d'images de biopsies.

### 1.2 Objectif du Projet

Créer un système d'aide à la décision pour les professionnels de santé permettant de :
- Accélérer le processus de diagnostic
- Réduire les erreurs humaines
- Améliorer la précision de détection des tumeurs malignes
- Optimiser la prise en charge des patientes

### 1.3 Dataset

Le **Breast Cancer Wisconsin (Diagnostic) Dataset** contient des mesures numériques calculées à partir d'images numériques de biopsies par aspiration à l'aiguille fine (FNA). Chaque échantillon est caractérisé par 30 variables décrivant les propriétés des noyaux cellulaires présents dans l'image.

## 2. Code et Librairies Utilisées

### 2.1 Bibliothèques Principales

```python
import pandas as pd              # Manipulation de données
import matplotlib.pyplot as plt  # Visualisations de base
import seaborn as sns           # Visualisations statistiques
from sklearn.model_selection import train_test_split  # Division des données
from sklearn.ensemble import RandomForestClassifier   # Algorithme de classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### 2.2 Chargement des Données

```python
# Importation via Google Colab
from google.colab import files
uploaded = files.upload()

# Lecture du fichier CSV
df = pd.read_csv("data.csv")
```

### 2.3 Structure du Code

Le code est organisé en plusieurs étapes séquentielles :
1. Import des données et exploration initiale
2. Nettoyage et préparation (Data Wrangling)
3. Analyse exploratoire avec visualisations (EDA)
4. Division du dataset (Train/Test Split)
5. Entraînement du modèle Random Forest
6. Évaluation des performances

## 3. Data Wrangling (Préparation des Données)

### 3.1 Inspection Initiale

```python
# Aperçu des données
print(df.head())
print(df.info())
print(df.describe())
```

**Résultats attendus** :
- Nombre de lignes et colonnes
- Types de données (numériques vs catégorielles)
- Identification des valeurs manquantes

### 3.2 Vérification des Valeurs Manquantes

```python
# Comptage des valeurs manquantes par colonne
for col in df.columns:
    print(f"{col} : {df[col].isnull().sum()} valeurs manquantes")
```

### 3.3 Nettoyage

**Actions à effectuer** :
- Suppression de la colonne `Unnamed: 32` (colonne vide)
- Suppression de la colonne `id` (identifiant non pertinent pour la prédiction)
- Vérification qu'il n'y a pas de doublons

```python
# Suppression des colonnes non pertinentes
df = df.drop(columns=['id', 'Unnamed: 32'])

# Vérification des doublons
print(f"Nombre de doublons : {df.duplicated().sum()}")
```

### 3.4 Encodage de la Variable Cible

La variable `diagnosis` doit être encodée en valeurs numériques :

```python
# Encodage : M (Malin) = 1, B (Bénin) = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
```

### 3.5 Vérification des Valeurs Uniques

```python
# Distribution de la variable cible
print(df['diagnosis'].value_counts())
```

## 4. EDA (Exploratory Data Analysis)

### 4.1 Analyse Univariée

**Histogrammes de toutes les variables numériques** :

```python
df.hist(figsize=(12,8), bins=15, color='skyblue')
plt.suptitle("Distribution des colonnes numériques")
plt.show()
```

**Interprétation** :
- Identifier la forme des distributions (normale, asymétrique)
- Détecter les variables potentiellement bimodales
- Repérer les échelles de valeurs différentes

### 4.2 Détection des Outliers

**Boxplots horizontaux** :

```python
df.plot(kind='box', figsize=(12,6), vert=False, color='orange')
plt.title("Boxplots des colonnes numériques")
plt.show()
```

**Objectif** :
- Visualiser les valeurs extrêmes
- Identifier les variables nécessitant une normalisation
- Décider du traitement des outliers (conservation ou suppression)

### 4.3 Analyse Bivariée - Matrice de Corrélation

```python
# Sélection des colonnes numériques uniquement
df_numeric = df.drop(columns=['diagnosis'])

# Calcul des corrélations
corr = df_numeric.corr()

# Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()
```

**Insights** :
- Variables fortement corrélées (>0.8) : risque de multicolinéarité
- Identification des features redondantes
- Sélection des variables les plus informatives

### 4.4 Analyse de la Variable Cible

```python
# Répartition Bénin vs Malin
sns.countplot(x='diagnosis', data=df, palette='pastel')
plt.title("Répartition des diagnostics")
plt.xlabel("Diagnostic (0=Bénin, 1=Malin)")
plt.show()
```

**Vérification** : Le dataset est-il équilibré ou déséquilibré ?

## 5. Split (Division des Données)

### 5.1 Séparation Features/Target

```python
# Variables explicatives (X)
X = df.drop(columns=['diagnosis'])

# Variable cible (y)
y = df['diagnosis']
```

### 5.2 Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Division 80% train / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintient la proportion de classes
)

print(f"Taille du jeu d'entraînement : {X_train.shape}")
print(f"Taille du jeu de test : {X_test.shape}")
```

### 5.3 Normalisation (Optionnelle mais Recommandée)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Justification** : Les variables ont des échelles très différentes, la normalisation améliore les performances de nombreux algorithmes.

## 6. Algorithme Random Forest

### 6.1 Présentation de l'Algorithme

**Random Forest** est un algorithme d'ensemble qui :
- Construit plusieurs arbres de décision
- Combine leurs prédictions par vote majoritaire
- Réduit le surapprentissage grâce au bagging
- Gère bien les données non linéaires
- Résiste aux outliers

### 6.2 Entraînement du Modèle

```python
from sklearn.ensemble import RandomForestClassifier

# Instanciation du modèle
rf_model = RandomForestClassifier(
    n_estimators=100,      # Nombre d'arbres
    max_depth=10,          # Profondeur maximale
    random_state=42,
    n_jobs=-1              # Utilise tous les processeurs
)

# Entraînement
rf_model.fit(X_train, y_train)
```

### 6.3 Importance des Features

```python
# Calcul de l'importance des variables
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualisation
plt.figure(figsize=(10,8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title("Top 15 des variables les plus importantes")
plt.show()
```

**Interprétation** : Identifier quelles caractéristiques cellulaires sont les plus discriminantes pour le diagnostic.

## 7. Évaluation du Modèle

### 7.1 Prédictions

```python
# Prédictions sur le jeu de test
y_pred = rf_model.predict(X_test)
```

### 7.2 Métriques de Performance

#### 7.2.1 Accuracy (Précision Globale)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
```

**Interprétation** : Pourcentage de prédictions correctes sur l'ensemble du test.

#### 7.2.2 Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, 
                           target_names=['Bénin', 'Malin']))
```

**Métriques clés** :
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall (Sensibilité)** : Proportion de vrais positifs détectés
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **Support** : Nombre d'échantillons réels par classe

#### 7.2.3 Matrice de Confusion

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Visualisation
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bénin', 'Malin'],
            yticklabels=['Bénin', 'Malin'])
plt.title("Matrice de Confusion")
plt.ylabel("Valeur Réelle")
plt.xlabel("Valeur Prédite")
plt.show()
```

**Analyse** :
- **Vrais Négatifs (TN)** : Tumeurs bénignes correctement identifiées
- **Vrais Positifs (TP)** : Tumeurs malignes correctement identifiées
- **Faux Positifs (FP)** : Fausses alarmes (bénin prédit malin)
- **Faux Négatifs (FN)** : Cas graves manqués (malin prédit bénin) ⚠️ **Critique en médecine**

### 7.3 Courbe ROC et AUC

```python
from sklearn.metrics import roc_curve, auc

# Prédiction des probabilités
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualisation
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.show()
```

**Interprétation de l'AUC** :
- AUC = 0.5 : Modèle aléatoire (aucune capacité prédictive)
- AUC = 0.7-0.8 : Performance acceptable
- AUC = 0.8-0.9 : Bonne performance
- AUC > 0.9 : Excellente performance

### 7.4 Validation Croisée

```python
from sklearn.model_selection import cross_val_score

# Cross-validation avec 5 folds
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

print(f"Scores CV : {cv_scores}")
print(f"Moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Objectif** : Vérifier que le modèle généralise bien et n'est pas surappris.

## Conclusions et Recommandations

### Points Forts du Projet

1. **Dataset de qualité** : Peu de valeurs manquantes, bien structuré
2. **Random Forest** : Algorithme robuste et performant pour ce type de problème
3. **Méthodologie complète** : De l'EDA à l'évaluation détaillée

### Améliorations Possibles

1. **Optimisation des hyperparamètres** : Utiliser GridSearchCV ou RandomizedSearchCV
2. **Feature Engineering** : Créer de nouvelles variables combinées
3. **Essayer d'autres algorithmes** : SVM, XGBoost, Réseaux de neurones
4. **Gestion du déséquilibre** : Si les classes sont déséquilibrées, utiliser SMOTE
5. **Mise en production** : Développer une API Flask/FastAPI pour l'utilisation clinique

### Impact Métier

Ce modèle peut servir d'**outil d'aide à la décision** pour les radiologues et oncologues, en :
- Réduisant le temps de diagnostic
- Fournissant une seconde opinion automatisée
- Priorisant les cas urgents (tumeurs malignes détectées)
- Améliorant la reproductibilité des diagnostics

---

**Note** : Ce projet est à but éducatif. Toute utilisation clinique nécessiterait une validation médicale rigoureuse et des certifications réglementaires appropriées.
