# Salma Fadli G2 Finance
# Assurance Santé
![photo de salma fadli.jpeg](https://github.com/fadlisalmaencg-a11y/DS-2025/blob/main/photo%20de%20salma%20fadli.jpeg?raw=true)
# 📊 Compte Rendu d’Analyse & Clustering du Dataset *Insurance Charges*

## 📝 1. Introduction
Ce rapport présente une analyse exploratoire et une application de techniques de **clustering** sur le dataset *Insurance Charges*.  
L’objectif est de :

- comprendre la structure des données,  
- préparer un pré-traitement adapté,  
- appliquer différents algorithmes de clustering,  
- comparer les performances à l’aide d’indicateurs (ex : silhouette),  
- visualiser les résultats via PCA.  

Le dataset est chargé directement depuis une source publique, sans besoin de fichier ZIP.

---

## 📂 2. Chargement du Dataset
Le dataset provient de :


Il contient les colonnes suivantes :

- **age** : âge de l’assuré  
- **sex** : homme / femme  
- **bmi** : indice de masse corporelle  
- **children** : nombre d’enfants  
- **smoker** : fumeur ou non  
- **region** : zone géographique  
- **charges** : montant annuel des frais médicaux  

---

## 🔍 3. Analyse exploratoire (EDA)

### ✔ Aperçu général
- Aucune valeur manquante.  
- Colonnes numériques : `age`, `bmi`, `children`, `charges`  
- Colonnes catégorielles : `sex`, `smoker`, `region`

### ✔ Observations principales
- Les fumeurs ont des charges beaucoup plus élevées.  
- Le BMI influence fortement les dépenses médicales.  
- Des groupes naturels semblent exister (fumeurs/non-fumeurs avec BMI élevé).  

---

## ⚙️ 4. Pré-traitement

### Pipeline :
- **StandardScaler** pour les variables numériques  
- **OneHotEncoding** pour les variables catégorielles  
- Construction d’une matrice prête pour le clustering  
- **PCA (2 composantes)** pour visualisation simplifiée  

---

## 🤖 5. Méthodes de Clustering Appliquées

### ### ⭐ 5.1 K-Means
- Test des valeurs de k entre 2 et 8  
- Analyse via :
  - **méthode du coude (inertia)**
  - **score silhouette**

✔ **k optimal** ≈ 3  
✔ **Silhouette** ≈ 0.40 (indicatif)  

### ### ⭐ 5.2 Agglomerative Clustering
- Utilisation du même k optimal  
- Score silhouette légèrement inférieur à K-Means  

### ### ⭐ 5.3 DBSCAN
- Permet de détecter :
  - clusters de forme irrégulière  
  - points bruit  
- Résultats sensibles au choix de `eps`  

---

## 📉 6. Visualisations

### PCA 2D avec clusters :
- Les clusters sont bien séparés surtout selon :
  - statut fumeur  
  - BMI élevé  
  - charges médicales importantes  

Les zones sont visuellement cohérentes :  
fumeurs à BMI élevé forment un cluster très distinct.

---

## 🏁 7. Conclusion

- **K-Means (k = 3)** offre la meilleure segmentation globale.  
- Les clusters identifiés correspondent à des profils clairs :
  1. Fumeurs → charges très élevées  
  2. Non-fumeurs, BMI modéré  
  3. Jeunes assurés, charges faibles  

- **Agglomerative** : résultats acceptables mais moins performants.  
- **DBSCAN** : utile pour détecter anomalies / bruit.

---

## 💻 8. Code Python utilisé (Google Colab)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# Charger dataset sans ZIP
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)
df.head()

# Colonnes
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Préprocessing
numeric = Pipeline([('scaler', StandardScaler())])
categorical = Pipeline([('ohe', OneHotEncoder(sparse=False))])

preprocess = ColumnTransformer([
    ('num', numeric, num_cols),
    ('cat', categorical, cat_cols)
])

X = preprocess.fit_transform(df)

# PCA
pca = PCA(2)
X_pca = pca.fit_transform(X)

# K-Means
scores = []
for k in range(2,8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))

best_k = 2 + np.argmax(scores)
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels_k = kmeans.fit_predict(X)

# Visualisation
plt.figure(figsize=(8,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels_k, palette="tab10")
plt.title(f"Clustering K-Means (k={best_k}) — PCA")
plt.show()



