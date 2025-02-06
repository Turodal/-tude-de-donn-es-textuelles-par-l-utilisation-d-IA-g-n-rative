# README - Etude-de-donnees-textuelles-par-l-utilisation-d-IA-generative

## Introduction

Ce projet vise à analyser et regrouper des commentaires sur la beauté chez la femme en utilisant diverses techniques de traitement du langage naturel (NLP) et de clustering.

## Prérequis

Avant d'exécuter le script, assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install pandas numpy matplotlib scipy scikit-learn sentence-transformers ollama plotly gensim
```

## Description du Code

### 1. Chargement des Données

Le script charge un fichier Excel contenant des commentaires et sélectionne les colonnes d'intérêt pour l'analyse.

- **Fichier source** : `Donnees_completes_CWays_2024.xlsx`
- **Colonnes utilisées** : `['BEA1', 'REC_DIPLOME', 'Rec_sexe', 'Rec_age', 'Q0_4', 'CC', 'Rec_CSP', 'Rec_CSP2']`
- Attention le programme ne marche qu'avec des colonnes qualitatives et non quantitative

### 2. Définition des Paramètres

Le script définit plusieurs paramètres pour le traitement et le clustering des commentaires :

- **Nombre d'échantillons** : `n_sample = 2`
- **Filtrage des commentaires** :
  - Minimum de mots : `n_mots_min = 10`
  - Maximum de mots : `n_mots_max = 20`
  - Nombre de lignes à garder : `n_lignes = 300`
- **Nombre de clusters** : `[3, 5, 10]`
- **Nom de la colonne textuelle** : `BEA1`
- **Modèle de traitement du langage** : `Word2Vec` ou `Sbert`

### 3. Prompts pour le Modèle de Langage

Le script utilise un LLM pour analyser et résumer les thèmes des clusters :

- **Identification des thèmes principaux** :
  - Prompt : `prompt`
- **Résumé des thèmes principaux :**
  - Prompt : `prompt2`
- **Commentaire le plus représentatif** :
  - Prompt : `prompt3`

### 4. Clustering des Commentaires

Le script utilise **l'algorithme de classification ascendante hiérarchique** pour regrouper les commentaires en différents clusters.

- **Colonnes générées** : `Cluster_3`, `Cluster_5`, `Cluster_10`
- **Thèmes associés** : `Thème_Cluster_3`, `Thème_Cluster_5`, `Thème_Cluster_10`
- Il est impératif de garder le nom des clusters et des thèmes comme présenté dans l'exemple (on peut alors que changer le numéro de cluster)

## Exécution du Script

Pour exécuter le script, lancez simplement la commande :

```bash
python script.py
```

Assurez-vous que le fichier de données est bien placé dans le dossier `Données` avant l'exécution.

## Résultats et Interprétation

Après exécution du programme, une page devrait s'ouvrir avec un graphique intéractif semblable à dendogramme.
Pour les classes les plus précis affichées, il y a aussi un exemple de commentaire associé. Celui ci peut être caculé (le commentaire le plus proche du centroïde de la classe) ou peut être extrait grâce au LLM 


# README: Analyse des réponses de consommateurs sur la beauté

## Description
Ce script analyse un jeu de données provenant de consommateurs sur leurs perceptions de la beauté chez les hommes et les femmes. Il évalue des réponses à plusieurs questions en utilisant l'API **Ollama** pour obtenir des prédictions sur la complexité, les clichés, et la tonalité des réponses. Les résultats sont ensuite utilisés pour effectuer un calcul de l'alpha de Krippendorff, une mesure de la fiabilité entre les juges.

Les étapes principales comprennent :
1. Chargement des données depuis un fichier Excel.
2. Création de prompts pour évaluer différentes dimensions des réponses : beauté intérieure/extérieur, complexité, clichés, tonalité positive/négative, et niveau de langue.
3. Application de ces prompts sur les réponses des consommateurs.
4. Calcul de la fiabilité des juges à l'aide de l'alpha de Krippendorff.
5. Visualisation des résultats dans un graphique.

## Prérequis
Avant d'exécuter ce script, vous devez installer les dépendances suivantes :
- **Ollama** : Pour l'API de chat (utilisé pour générer les réponses aux prompts).
- **Pandas** : Pour le traitement des données.
- **Numpy** : Pour les calculs numériques.
- **Gensim** : Pour le modèle Word2Vec (bien que ce module ne semble pas être utilisé directement dans ce script, il est importé).
- **Seaborn & Matplotlib** : Pour la visualisation des résultats.
- **Scikit-learn** : Pour la normalisation des données.
- **Scipy** : Pour l'estimation de densité gaussienne.
- **Krippendorff** : Pour le calcul de l'alpha de Krippendorff.

Installez les modules nécessaires avec la commande :
```bash
pip install pandas numpy gensim seaborn matplotlib scikit-learn scipy krippendorff
```

## Structure du code
1. **Chargement des données** :
   Le fichier Excel est chargé et un sous-ensemble des questions est extrait pour les analyses.

2. **Création des prompts** :
   Plusieurs prompts sont définis pour obtenir des informations sur les réponses des consommateurs. Les dimensions analysées sont :
   - Beauté intérieure/extérieur (homme/femme)
   - Complexité des réponses
   - Clichés
   - Tonalité positive
   - Niveau de langage

3. **Fonction `juges`** :
   Cette fonction applique les prompts sur les réponses des consommateurs pour évaluer chaque dimension et stocke les résultats dans de nouvelles colonnes du DataFrame.

4. **Calcul de l'alpha de Krippendorff** :
   L'alpha de Krippendorff est calculé pour évaluer la fiabilité des juges qui ont évalué les réponses.

5. **Visualisation des résultats** :
   Un graphique est généré pour visualiser les résultats de l'alpha de Krippendorff pour chaque dimension analysée.

6. **Fonction `ajouter_colonne`** :
   Cette fonction permet d'ajouter une colonne de résultats à partir d'un prompt. Elle est utilisée pour appliquer plusieurs prompts sur les réponses et ajouter des colonnes supplémentaires à notre DataFrame.

## Fonctionnalités principales

- **`juges` et `juges2`** : Applique un prompt à une réponse et calcule des résultats à travers plusieurs itérations.
- **`calculer_krippendorff`** : Calcule l'alpha de Krippendorff pour évaluer la cohérence entre les juges.
- **`ajouter_colonne`** : Ajoute une nouvelle colonne de résultats à un DataFrame en appliquant un prompt.

## Utilisation
1. Placez vos données sous le format Excel dans le dossier "Données" et assurez-vous que le fichier est au bon emplacement.
2. Exécutez le script Python pour appliquer les analyses et obtenir les résultats.
3. Le script générera un graphique comparant les résultats de fiabilité entre les juges pour chaque dimension analysée.

## Exemple d'exécution
Le script commence par charger un fichier Excel de données, puis il applique plusieurs prompts pour évaluer la beauté intérieure/extérieur, la complexité des réponses, les clichés, la tonalité positive et le niveau de langage des réponses. À la fin, un graphique montre les résultats des coefficients de Krippendorff pour chaque variable.

## Résultats attendus
Le script génère un graphique de points représentant l'alpha de Krippendorff pour différentes dimensions du jeu de données (beauté intérieure, complexité, clichés, tonalité positive, etc.). Ces résultats permettent de visualiser la fiabilité des juges dans l'évaluation des réponses.

## Remarques
- Les résultats dépendent fortement de la qualité des données initiales et des prompts utilisés.
- Vous pouvez personnaliser les prompts ou adapter les fonctions pour d'autres types d'analyse.


## Auteur

Projet réalisé par DIOP Massamba et HERAUD Jean

