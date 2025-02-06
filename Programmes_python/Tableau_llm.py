#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:05:14 2024

@author: jeanheraud
"""

from ollama import chat
from ollama import ChatResponse
import pandas as pd
import random as rd
import os
import matplotlib.pyplot as plt

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

def juges(data, prompt, sexe, nom_nouvelle_colonne):
    """
    Fonction qui applique un prompt aux réponses de beauté en fonction du sexe 
    et ajoute de nouvelles colonnes au dataframe avec les résultats.

    :param data: DataFrame contenant les réponses des consommateurs.
    :param prompt: Le prompt utilisé pour interroger le modèle.
    :param sexe: Le sexe du consommateur (homme ou femme).
    :param nom_nouvelle_colonne: Le nom de la nouvelle colonne à ajouter au DataFrame.
    :return: Le DataFrame mis à jour avec les nouvelles colonnes.
    """
    
    # Mapping du sexe à la colonne correspondante
    sexe = "BEA2" if sexe.lower() == "homme" else "BEA1"
    
    # Initialiser une liste de colonnes vides pour stocker les résultats
    colonnes = [[] for _ in range(5)]
    
    # Traiter chaque réponse dans la colonne spécifique (selon le sexe)
    for k, reponse in enumerate(data[sexe]):
        print(f"Traitement de la réponse {k+1}")
        
        # Liste pour stocker les résultats de chaque question (5 résultats par réponse)
        liste_moyenne = []
        
        # Boucle pour obtenir plusieurs réponses du modèle (5 fois)
        for _ in range(5):
            prompt_complet = f"{prompt} : '{reponse}.'"
            response = chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt_complet}])
            nombre = nombre_trouve(response['message']['content'])
            liste_moyenne.append(nombre)
        
        # Ajouter les résultats dans les colonnes correspondantes
        for i in range(5):
            colonnes[i].append(liste_moyenne[i])
    
    # Compléter les colonnes si nécessaire pour correspondre à la longueur du DataFrame
    max_length = len(data)
    for col in colonnes:
        col.extend([None] * (max_length - len(col)))  # Remplir avec des None si nécessaire
    
    # Ajouter les nouvelles colonnes au DataFrame
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


def ajouter_colonne(data, prompt, sexe, nom_nouvelle_colonne):
    """
    Fonction qui applique un prompt aux réponses en fonction du sexe 
    et ajoute une nouvelle colonne avec les moyennes des résultats obtenus.

    :param data: DataFrame contenant les réponses des consommateurs.
    :param prompt: Le prompt utilisé pour interroger le modèle.
    :param sexe: Le sexe du consommateur (homme ou femme).
    :param nom_nouvelle_colonne: Le nom de la nouvelle colonne à ajouter au DataFrame.
    :return: Le DataFrame mis à jour avec la nouvelle colonne.
    """
    # 1. Traiter et normaliser l'argument sexe
    sexe = sexe.lower()
    if sexe == "homme":
        sexe = "BEA2"
    else:
        sexe = "BEA1"
    
    # 2. Initialiser une liste pour stocker les résultats des moyennes
    list_ia = []
    
    # 3. Parcourir les réponses pour le sexe spécifié
    for reponse in data[sexe]:
        # Initialiser une liste pour stocker les valeurs pour chaque itération
        liste_moyenne = []
        
        # 4. Boucle pour effectuer 5 appels à l'IA (par exemple)
        for boucle in range(5):
            # Créer le prompt complet pour chaque appel
            prompt_complet = f"{prompt} : '{reponse}.'"
            
            # Effectuer l'appel à l'IA et récupérer la réponse
            response = chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt_complet}])
            
            # Extraire le nombre de la réponse de l'IA (fonction à définir)
            nombre = nombre_trouve(response['message']['content'])
            liste_moyenne.append(nombre)
        
        # 5. Filtrer les valeurs valides (nombres)
        nombres = [x for x in liste_moyenne if isinstance(x, (int, float))]
        
        # 6. Calculer la moyenne des valeurs valides
        if len(nombres) > 0:
            moyenne = sum(nombres) / len(nombres)
        else:
            moyenne = None  # Ou gérer autrement si aucune valeur valide n'est trouvée
        
        # 7. Ajouter la moyenne calculée à la liste
        list_ia.append(moyenne)
    
    # 8. Appliquer une fonction (comme repetition) pour ajuster la taille de la liste
    nouvelle_colonne = repetition(list_ia)
    
    # 9. Ajouter la nouvelle colonne au DataFrame
    data[nom_nouvelle_colonne] = nouvelle_colonne
    
    return data

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






