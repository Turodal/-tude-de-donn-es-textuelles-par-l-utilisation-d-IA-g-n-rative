#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:05:14 2024

@author: jeanheraud
"""

from ollama import chat
from ollama import ChatResponse

from ollama import chat
import ollama

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import random as rd
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
rd.seed(a=63, version=2)


#chemin vers le fichier des données
# Définir le chemin du fichier actuel (ou du script principal)
# Pour un notebook, utilisez le chemin du notebook
__file__ = os.getcwd()  # Simule __file__ comme le répertoire courant

# BASE_DIR peut maintenant être défini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR :", BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "Données")

#chemin vers le fichier des données
file_path = os.path.join(BASE_DIR, "Données/Donnees_completes_CWays_2024.xlsx")

data_complete = pd.read_excel(file_path)

# Charger les données Excel en spécifiant le bon sheet et en enlevant la première ligne (header)
data_complete = pd.read_excel(file_path, sheet_name="Labels", header=1)

data_questions = data_complete[["Rec_sexe", "BEA1", "BEA2"]]

data_questions = data_questions.iloc[1:10]


import re
def nombre_trouve(reponse):
    nombre_trouve = re.search(r'\d+', reponse)
    # Si un nombre est trouvé, le convertir en entier
    if nombre_trouve:
        nombre = int(nombre_trouve.group())
        return(nombre)
    else:
        return(reponse)

prompt_reflexion1 = "Le consommateur répond à une question sur la beauté chez la femme.  Indique si le consommateur a une réflexion simple ou complexe  - Réponds **uniquement** par **1** si le consommateur a une **réflexion complexe**.  - Réponds **uniquement** par **0** si le consommateur a une **réflexion simple**. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."
prompt_fin = "Ne donne aucune explication, mets le toujours en écriture numérique, sans aucune ponctuation."


print(data_questions)


def repetition(liste):
    for k in range(len(liste)):
        if liste[k] > 0.5:
            liste[k] = 1
        else:
            liste[k] = 0
    return(liste)
            

#création des prompts
prompt_beauté_femme = "Le consommateur répond à une question sur la beauté chez la femme.  Indique si le consommateur met en avant la beauté **intérieure** (comme la personnalité, les qualités morales, le coeur, etc.) ou la beauté **extérieure** (comme l'apparence physique).  - Réponds **uniquement** par **1** si le consommateur parle de beauté intérieure.  - Réponds **uniquement** par **0** si le consommateur parle de beauté extérieure. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_beauté_homme = "Le consommateur répond à une question sur la beauté chez l'homme.  Indique si le consommateur met en avant la beauté **intérieure** (comme la personnalité, les qualités morales comme la gentillesse, etc.) ou la beauté **extérieure** (comme l'apparence physique).  - Réponds **uniquement** par **1** si le consommateur parle de beauté intérieure.  - Réponds **uniquement** par **0** si le consommateur parle de beauté extérieure. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_complexite_homme = "Le consommateur répond à une question sur la beauté chez l'homme. Indique si le consommateur a une réflexion complexe.  - Réponds **uniquement** par **1** si le consommateur a une réflexion complexe.  - Réponds **uniquement** par **0** si le consommateur n'a pas de réflexion complexe N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_complexite_femme= "Le consommateur répond à une question sur la beauté chez la femme. Indique si le consommateur a une réflexion complexe.  - Réponds **uniquement** par **1** si le consommateur a une réflexion complexe.  - Réponds **uniquement** par **0** si le consommateur n'a pas de réflexion complexe. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_cliche_femme= "Le consommateur répond à une question sur la beauté chez la femme. Indique si le consommateur a une réponse clichée.  - Réponds **uniquement** par **1** si le consommateur a une réponse clichée.  - Réponds **uniquement** par **0** si le consommateur n'a pas de réponse clichée. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_cliche_homme= "Le consommateur répond à une question sur la beauté chez l'homme. Indique si le consommateur a une réponse clichée.  - Réponds **uniquement** par **1** si le consommateur a une réponse clichée.  - Réponds **uniquement** par **0** si le consommateur n'a pas de réponse clichée. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_positive_femme= "Le consommateur répond à une question sur la beauté chez la femme. Indique si le consommateur a une réponse positive.  - Réponds **uniquement** par **1** si le consommateur a une réponse positive.  - Réponds **uniquement** par **0** si le consommateur n'a pas une réponse positive. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_positive_homme= "Le consommateur répond à une question sur la beauté chez l'homme. Indique si le consommateur a une réponse positive.  - Réponds **uniquement** par **1** si le consommateur a une réponse positive.  - Réponds **uniquement** par **0** si le consommateur n'a pas une réponse positive. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_niveau_langue_homme = "Le consommateur répond à une question sur la beauté chez l'homme. Indique si le consommateur utilise un langage soutenu.  - Réponds **uniquement** par **1** si le consommateur utilise un langage soutenu.  - Réponds **uniquement** par **0** si le consommateur n'utilise pas un langage soutenu. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

prompt_niveau_langue_femme = "Le consommateur répond à une question sur la beauté chez la femme. Indique si le consommateur utilise un langage soutenu.  - Réponds **uniquement** par **1** si le consommateur utilise un langage soutenu.  - Réponds **uniquement** par **0** si le consommateur n'utilise pas un langage soutenu. N'ajoute aucun texte supplémentaire, réponds seulement par **1** ou **0**."

import re
def juges(data, prompt, sexe, nom_nouvelle_colonne):
    sexe = sexe.lower()
    if sexe == "homme":
        sexe = "BEA2"
    else:
        sexe = "BEA1"
    
    # Initialiser une liste pour stocker toutes les valeurs des colonnes
    colonnes = [[] for _ in range(5)]
    k = 0
    # Traiter les réponses uniquement pour la plage souhaitée
    for reponse in data[sexe]:
        print(k)
        liste_moyenne = []
        for _ in range(5):
            prompt_complet = f"{prompt} : '{reponse}.'"
            response: ChatResponse = chat(model='llama3.2', messages=[
                {
                    'role': 'user',
                    'content': prompt_complet,
                },
            ])
            nombre = nombre_trouve(response['message']['content'])
            liste_moyenne.append(nombre)
        
        # Ajouter chaque valeur de la liste dans la colonne correspondante
        for i in range(5):
            colonnes[i].append(liste_moyenne[i])
        k = k +1
    
    # Compléter les colonnes si nécessaire pour correspondre à la longueur du DataFrame
    while len(colonnes[0]) < len(data):
        for col in colonnes:
            col.append(None)  # Ajouter des valeurs manquantes (None) pour correspondre à la longueur
    
    # Ajouter les colonnes au DataFrame
    for i in range(5):
        data[f"{nom_nouvelle_colonne}_{i+1}"] = colonnes[i]
    
    return data


def juges2(data, prompt, sexe, nom_nouvelle_colonne):
    sexe = sexe.lower()
    if sexe == "homme":
        sexe = "BEA2"
    else:
        sexe = "BEA1"
    
    # Initialiser une liste pour stocker toutes les valeurs des colonnes
    colonnes = [[] for _ in range(5)]
    k = 0
    # Traiter les réponses uniquement pour la plage souhaitée
    for reponse in data['text']:
        print(k)
        liste_moyenne = []
        for _ in range(5):
            prompt_complet = f"{prompt} : '{reponse}.'"
            response: ChatResponse = chat(model='llama3.2', messages=[
                {
                    'role': 'user',
                    'content': prompt_complet,
                },
            ])
            nombre = nombre_trouve(response['message']['content'])
            liste_moyenne.append(nombre)
        
        # Ajouter chaque valeur de la liste dans la colonne correspondante
        for i in range(5):
            colonnes[i].append(liste_moyenne[i])
        k = k +1
    
    # Compléter les colonnes si nécessaire pour correspondre à la longueur du DataFrame
    while len(colonnes[0]) < len(data):
        for col in colonnes:
            col.append(None)  # Ajouter des valeurs manquantes (None) pour correspondre à la longueur
    
    # Ajouter les colonnes au DataFrame
    for i in range(5):
        data[f"{nom_nouvelle_colonne}_{i+1}"] = colonnes[i]
    
    return data

data_juges = juges(data_questions, prompt_beauté_femme, "femme", "beauté_intérieur_femme")
data_juges = juges(data_juges, prompt_beauté_homme, "homme", "beauté_intérieur_homme")
data_juges = juges(data_juges, prompt_complexite_femme, "femme", "complexite_femme")
data_juges = juges(data_juges, prompt_complexite_homme, "homme", "complexite_homme")
data_juges = juges(data_juges, prompt_cliche_femme, "femme", "cliche_femme")
data_juges = juges(data_juges, prompt_cliche_homme, "homme", "cliche_homme")
data_juges = juges(data_juges, prompt_positive_femme, "femme", "positive_femme")
data_juges = juges(data_juges, prompt_positive_homme, "homme", "positive_homme")
data_juges = juges(data_juges, prompt_niveau_langue_femme, "femme", "langage_femme")
data_juges = juges(data_juges, prompt_niveau_langue_homme, "homme", "langage_homme")


data_juges = data_juges.iloc[:, 3:]

n_colonnes = 5

# Découper en plusieurs sous-DataFrames
sous_dataframes_juges = [data_juges.iloc[:, i:i+n_colonnes] for i in range(0, data_juges.shape[1], n_colonnes)]


import krippendorff

def calculer_krippendorff(group):
    # Convertir les données en une liste de listes (lignes en liste de valeurs pour chaque juge)
    group_values = group.values.T.tolist()
    print(group_values)
    # Calculer l'alpha de Krippendorff
    alpha = krippendorff.alpha(group_values, level_of_measurement='nominal')
    return alpha

alpha_values = []

for idx, sous_df in enumerate(sous_dataframes_juges):
    print(idx)
    alpha = calculer_krippendorff(sous_df)
    alpha_values.append(alpha)
    print(f"Alpha de Krippendorff pour le sous-DataFrame {idx + 1} : {alpha:.2f}")

# Résumé des valeurs d'alpha calculées
print("\nRésumé des alphas de Krippendorff pour chaque sous-DataFrame:")
for idx, alpha in enumerate(alpha_values):
    print(f"Sous-DataFrame {idx + 1}: {alpha:.2f}")


x_labels = ['beauté intérieur', 'complexité de la phrase', 'phrase clichée', 'phrase positive', 'langage soutenu']

grouped_data = [alpha_values[i:i+2] for i in range(0, len(alpha_values), 2)]
mean_alpha_values = [sum(group) / len(group) for group in grouped_data]

# Associer chaque label qualitatif à ses deux points et sa moyenne
labels_and_values = [(label, group, mean) for label, group, mean in zip(x_labels, grouped_data, mean_alpha_values)]

# Trier les variables qualitatives par leur moyenne croissante
sorted_labels_and_values = sorted(labels_and_values, key=lambda x: x[2])
sorted_labels = [item[0] for item in sorted_labels_and_values]
sorted_grouped_data = [item[1] for item in sorted_labels_and_values]
# Créer un graphique

# Affichage du classement
print("Classement des variables qualitatives (du plus petit au plus grand alpha moyen) :")
for label, group, mean in sorted_labels_and_values:
    print(f"{label}: {mean:.2f} (Points : {group[0]:.2f}, {group[1]:.2f})")

# Visualisation des résultats avec 2 points par variable
fig, ax = plt.subplots(figsize=(8, 5))

# Dessiner les points pour chaque variable avec des couleurs spécifiques
for i, (x_label, points) in enumerate(zip(sorted_labels, sorted_grouped_data)):
    x = [i, i]  # Position sur l'axe x (même pour les deux points)
    colors = ['red', 'blue']  # Premier point rouge, deuxième point bleu
    ax.scatter(x, points, c=colors, s=100)

# Ajouter les labels de l'axe x
ax.set_xticks(range(len(sorted_labels)))
ax.set_xticklabels(sorted_labels)

# Ajouter des titres et labels
ax.set_title("Graphique des Coefficients de reproductibilité pour toutes les questions", fontsize=14)
ax.set_xlabel("Questions évaluées", fontsize=12)
ax.set_ylabel("Alpha de Krippendorff", fontsize=12)

# Ajouter une légende avec des points fictifs
ax.scatter([], [], c='red', label='BEA1', s=100)  # Point rouge fictif
ax.scatter([], [], c='blue', label='BEA2', s=100)  # Point bleu fictif
ax.legend(loc='upper left', fontsize=12)

# Afficher la grille
ax.grid(True, linestyle='--', alpha=0.6)

# Afficher le graphique
plt.tight_layout()
plt.show()


#A partir d'un jeu de données et d'un prompt, nous donne une nouvelle colonne aux jeux de données
def ajouter_colonne(data, prompt, sexe, nom_nouvelle_colonne):
    sexe = sexe.lower()
    if sexe == "homme":
        sexe = "BEA2"
    else:
        sexe = "BEA1"
    list_ia = []
    k = 0
    for reponse in data[sexe]:
        print(k)
        liste_moyenne = []
        for boucle in range(5):
            prompt_complet = f"{prompt} : '{reponse}.'"
            response: ChatResponse = chat(model='llama3.2', messages=[
              {
                'role': 'user',
                'content': prompt_complet,
              },
            ])
            nombre = nombre_trouve(response['message']['content'])
            liste_moyenne.append(nombre)
        nombres = [x for x in liste_moyenne if isinstance(x, (int, float))]
        # Calculer la moyenne
        if len(nombres) > 0:
            moyenne = sum(nombres) / len(nombres)        
        list_ia.append(moyenne)
        k = k + 1 
    nouvelle_colonne = repetition(list_ia)
    data[nom_nouvelle_colonne] = nouvelle_colonne
    return(data)
    
data_questions_nouvelles_col = ajouter_colonne(data_questions, prompt_beauté_femme, "femme", "beauté_intérieur_femme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_beauté_homme, "homme", "beauté_intérieur_homme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_complexite_femme, "femme", "complexite_femme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_complexite_homme, "homme", "complexite_homme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_cliche_femme, "femme", "cliche_femme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_cliche_homme, "homme", "cliche_homme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_positive_femme, "femme", "positive_femme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_positive_homme, "homme", "positive_homme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_niveau_langue_femme, "femme", "langage_femme")
data_questions_nouvelles_col = ajouter_colonne(data_questions_nouvelles_col, prompt_niveau_langue_homme, "homme", "langage_homme")






