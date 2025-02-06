

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

### 2. Définition des Paramètres

Le script définit plusieurs paramètres pour le traitement et le clustering des commentaires :

- **Nombre d'échantillons** : `n_sample = 2`
- **Filtrage des commentaires** :
  - Minimum de mots : `n_mots_min = 10`
  - Maximum de mots : `n_mots_max = 20`
  - Nombre de lignes à garder : `n_lignes = 300`
- **Nombre de clusters** : `[3, 5, 10]`
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

Après exécution du programme, une page devrait s'ouvrir avec un graphique intéractif semblable à dendogramme 

## Auteur

Projet réalisé par DIOP Massamba et HERAUD Jean

