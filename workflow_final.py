#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 22:27:50 2025

@author: jeanheraud
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:59:49 2025

@author: massambadiop
"""
test 
##### DATA Processing
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from ollama import chat
from ollama import ChatResponse
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import plotly.io as pio
import re
from scipy.stats import norm
from scipy.stats import chi2_contingency
from gensim.models import Word2Vec
from matplotlib import colors


__file__ = os.getcwd()  # Simule __file__ comme le répertoire courant

# BASE_DIR peut maintenant être défini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR :", BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "Données")
file_path = os.path.join(BASE_DIR, "Données/Donnees_completes_CWays_2024.xlsx")
Data = pd.read_excel(file_path, sheet_name="Labels", header = 1)
# Charger le fichier sans changer le nom de la variable (Data)
#Data = pd.read_excel(file_path)
Data = Data[['BEA1', 'REC_DIPLOME', 'Rec_sexe', 'Rec_age', 'Q0_4', 'CC', 'Rec_CSP', "Rec_CSP2"]]  # Sélectionner les colonnes d'intérêt
text = 'BEA1' # remplacer avec votre colonne contenant les commentaires
model = "W2V"


output_folder = "cluster_outputs"  # Dossier de téléchargement
os.makedirs(output_folder, exist_ok=True)

####################
# AGRUMENTS
####################
n_sample = 2  # Nombre de commentaires à échantillonner
# Filtre des commentaires
n_mots_min = 10  # nombre minimum de mots contenu dans un commentaire
n_mots_max = 20  # nombre maximum de mots contenu dans un commentaire
n_lignes = 300  # (Optionnel) Nombre de premières lignes à garder
n_clusters = [3, 5, 10]  # nombre de cluster que l on veut creer par la clasification ascendante hierarchique 
#prompt a utiliser pour avoir les themes d un cluster
prompt = "Voici un ensemble de commentaires sur la beauté chez la femme. Donne les thèmes principaux en exactement **5 mots**, écrits les uns après les autres, séparés uniquement par un espace. Ta réponse doit contenir **strictement 5 mots**, rien de plus, rien de moins. N'écris pas de phrases ni de ponctuation."
#prompt a utiliser pour faire le resume des themes d un cluster
prompt2 = "Voici un ensemble de commentaires sur la beauté chez la femme. Donne L'unique thème principal de ces phrases. Ta réponse doit contenir **au maximum 5 mots**, rien de plus, rien de moins. N'écris pas de phrases ni de ponctuation."
#prompt a utiliser si on veut utiliser le llm pour trouver le commentaire le plus representatif
prompt3 = "Voici un ensemble de commentaires sur la beauté chez la femme. Tous les commentaires sont dans une liste et sont séparés par une virgule. Donne le commentaire le plus représentatif de l'ensemble de la liste. Ta réponse doit contenir **juste un commentaire**, rien de plus, rien de moins. N'écris pas de phrases ni de ponctuation en plus."
# nom des differents cluster que l on cree (du type Cluster_n_cluster)
Cluster = ['Cluster_3', "Cluster_5", "Cluster_10"]
# nom des colonnes qui contiennent les themes (du type Thème_Cluster)
theme = ['Thème_Cluster_3', 'Thème_Cluster_5', 'Thème_Cluster_10']

Cluster_debut = 'Cluster_3'
Cluster_fin =   "Cluster_10"
### Filtre en fonction du nombre de mots
TEXT = Data[text] 

def filtre(df, TEXT, min_mots, max_mots):
    """
    Filtre les lignes d'un DataFrame en fonction du nombre de mots dans une colonne.

    Args:
        df (pd.DataFrame): Le DataFrame à filtrer.
        colonne (str): Le nom de la colonne contenant le texte.
        min_mots (int): Le nombre minimum de mots requis.
        max_mots (int): Le nombre maximum de mots permis.

    Returns:
        pd.DataFrame: Un DataFrame filtré selon le critère du nombre de mots.
    """
    return df[TEXT.apply(lambda x: min_mots <= len(str(x).split()) <= max_mots)].reset_index(drop=True)


Data = filtre(Data, TEXT, n_mots_min, n_mots_max)

if n_lignes is not None:
    Data = Data.iloc[:n_lignes+1, :]
else:
    Data = Data.iloc[:, :]

TEXT = Data[text] 
# ### Plot distribution univariée
# def Distribution(Data, col_name, item_name):
#     """
#     Construit le graphe de la distribution de variables d'intérêt.
#     Args:
#         Data (pd.DataFrame): Le data frame d'intérêt.
#         col_name (str): Le nom de la colonne contenant la variable.
#         item_name (str): Le nom de la variable.
#     """
#     sns.histplot(Data[col_name], kde=True, bins=30)
#     plt.title(f'Distribution par {item_name}')
#     plt.ylabel("Nombre d'individus")
#     plt.grid()
#     plt.show()


# # Distribution(Dior, 'country_code', 'PAYS')
# # Distribution(Dior, 'year', 'Années')

##############
# EMbeddings

# Charger le modèle RoBERTa
#RoBERTa = SentenceTransformer('stsb-roberta-large')



def embedding_analysis(TEXT, model, metric='cosine', output_results=True):
    """
    Analyze text embeddings, calculate distance matrices, and find most similar/distant pairs.

    Args:
        text (list[str]): List of sentences or texts.
        model (object): Embedding model (e.g., SentenceTransformer) with an `encode` method.
        
    Returns:
        dict: A dictionary containing:
            - embeddings (pd.DataFrame): DataFrame of embeddings.
    """
    # Input validation
    if not isinstance(TEXT, list) or not all(isinstance(t, str) for t in TEXT):
        raise ValueError("`TEXT` must be a list of strings.")
    if not hasattr(model, "encode"):
        raise ValueError("`model` must have an `encode` method.")
    # Generate embeddings
    embeddings = np.array(model.encode(TEXT, batch_size=32, show_progress_bar=True))

    return {
        'embeddings': pd.DataFrame(embeddings),
    }


def get_sentence_embedding(sentence, model):
    """
    Fait de l'embedding sur une phrase

    Parameters
    ----------
    text : liste de mots
    model : Word2Vec modèle ou similaire

    Returns
    -------
    sentence_embedding_np : array, les coordonnées de la phrase après embedding

    """
    word_vectors = [model.wv[word] for word in sentence if word in model.wv]
    if not word_vectors:  # Si aucun mot n'est connu du modèle
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def get_embedding_corpus(corpus, model):
    """
    Fait de l'embedding sur tout un corpus de texte en moyennant les embeddings des mots.
    Parameters
    ----------
    corpus : dataframe (Y x 1 avec Y appartenat à N+*)
        une ligne contient une liste de mot 
    model : modèle Word2Vec ou similaire.

    Returns
    -------
    sentence_embeddings : liste d'array
        Chaque ligne correspond à une phrase du corpus.

    """
    word_vectors = [get_sentence_embedding(sentence, model) for sentence in corpus]
    return word_vectors

#Fait de l'embedding du texte soit méthode Word2Vec soit avec Sbert

def embedding(corpus, model):
    if model == "W2V":
        word2vec_model = Word2Vec(corpus, vector_size=500, window=5, min_count=10, workers=4)
        Embeds = get_embedding_corpus(corpus, word2vec_model)
        Embeds = pd.DataFrame(Embeds)
    else:
        SBERT = SentenceTransformer('all-MiniLM-L6-v2')
        results = embedding_analysis(corpus.tolist(), SBERT, metric='cosine', output_results=True)
        Embeds = results['embeddings']
        Embeds = pd.DataFrame(Embeds)
    return Embeds





# Run analysis

# Save results to CSV (optional)
#results['embeddings'].to_csv('embeddings_Dior_SBERT.csv', index=False)
Embeds = embedding(TEXT, model)

###############
# Dendrogramme
###############
def dendro(embeddings, methode_embedding):
    """
    Trace un dendrogramme pour une classification hiérarchique ascendante (CAH),
    basé sur des embeddings générés par une méthode donnée.

    Parameters:
    -----------
    embeddings : array-like
        Les embeddings à utiliser pour le clustering.
    methode_embedding : str
        Le nom de la méthode utilisée pour générer les embeddings (par ex., 'USE', 'BERT').

    Returns:
    --------
    dict
        Un dictionnaire contenant la matrice de liaison calculée (Z).
    """
    # Calcul de la matrice de liaison avec la méthode 'ward' par défaut
    Z = linkage(embeddings, method='ward', metric='euclidean')
    # Tracer le dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(Z, orientation= 'top')
    plt.title(f"Dendrogramme pour la CAH avec embeddings ({methode_embedding})")
    plt.xlabel("Individus")
    plt.ylabel("Distance de fusion")
    plt.show()
    # Retourner la matrice de liaison sous forme de dictionnaire
    return Z

Dendro = dendro(Embeds, "SBERT")

#################
# Choix du nombre de clusters
#################


def plot_coude(Z):
    """
    Ttrace le graphe de la méthode du coude
    Args:
        Z_dendro (Dict): matrice de liaison issue de la fonction dendro.
   """
    distances = Z
    distances = distances[:, 2]
    # Tracer la méthode du coude pour la CAH
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(distances) + 1), distances[::-1], 'bo-')
    plt.title("Méthode du Coude pour CAH")
    plt.xlabel("Nombre de clusters restants")
    plt.ylabel("Distance de fusion")
    plt.grid()
    plt.show()
    return distances

distances = plot_coude(Dendro)

# 
def zoom(distances):
    """
    

    Parameters
    ----------
    distances : Array
        Zoom sur les premier clusters.

    Returns
    -------
    Plot.

    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(distances) + 1), distances[::-1], 'bo-')
    plt.title("Méthode du Coude pour CAH")
    plt.xlabel("Nombre de clusters restants")
    plt.ylabel("Distance de fusion")
    plt.xticks(range(0, len(distances) + 1, 10))
    plt.xlim(0, 50)  # Limite en fonction du nombre de clusters
    plt.grid()
    plt.show()


zoom(distances)


##################
# Clustering
##################

def clustering(data, embedding, clusters):
    # Loop through each cluster size
    df_clusters = pd.DataFrame()

    for k in n_clusters:
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')  # Use current k
        print(hc)
        df_clusters[f"Cluster_{k}"] = hc.fit_predict(embedding)  # Add cluster labels to DataFrame
    df_clusters['key'] = df_clusters.index
    data_complete = pd.concat([data, df_clusters], axis = 1)
    return(data_complete)
    
    
data_complete = clustering(Data, Embeds, n_clusters)

def call_model(prompt):
    """
    Fonction simulant un appel à un modèle de traitement de langage naturel.
    Remplacez cette fonction par une implémentation réelle pour intégrer un modèle ou une API.
    
    :param prompt: Prompt à envoyer au modèle.
    :return: Réponse générée par le modèle.
    """
    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    reponse = response['message']['content']
    return reponse


def generate_cluster_samples(df, cluster_column, text_column, promp, prompt2, k=0, n=60):
    """
    Génère un dataframe contenant des thèmes résumés pour chaque cluster.
    
    :param df: DataFrame contenant les données.
    :param cluster_column: Nom de la colonne indiquant les clusters.
    :param text_column: Nom de la colonne contenant les textes à concaténager.
    :param prompt: Prompt à utiliser pour les appels au modèle.
    :param k: Nombre d'échantillons à générer par cluster. Si 0, calculé dynamiquement.
    :param n: Nombre de lignes à échantillonner par répétition.
    :return: DataFrame contenant les clusters et leurs thèmes résumés.
    """
    # Obtenir les clusters uniques
    unique_clusters = df[cluster_column].unique()
    valeurs_uniques = unique_clusters.tolist()
    themes = [[] for _ in range(len(valeurs_uniques))]  # Initialisation des thèmes
    # Parcourir chaque cluster
    for cluster_value in unique_clusters:
        # Filtrer les lignes pour ce cluster
        filtered_df = df[df[cluster_column] == cluster_value]
        cluster_index = valeurs_uniques.index(cluster_value)  # Obtenir l'index du cluster
        sample_summaries = []
        # Calculer `k` si nécessaire
        if k == 0:
            k = max(1, len(filtered_df) // n)  # Minimum 1 échantillon

        # Si le cluster contient moins de `n` lignes
        if len(filtered_df) < n:
            for idx in range(k):
                concatenated_text = '. '.join(filtered_df[text_column])
                prompt_complet = f"{prompt} : '{concatenated_text}.'"
                response = call_model(prompt_complet)
                sample_summaries.append(response)

        else:
            # Générer `k` échantillons
            for sample_idx in range(k):
                sample_df = filtered_df.sample(n=n,
                                               replace=False,
                                               random_state=42 + sample_idx)
                concatenated_text = '. '.join(sample_df[text_column])
                prompt_complet = f"{prompt} : '{concatenated_text}.'"
                response = call_model(prompt_complet)
                sample_summaries.append(response)

        # Faire un résumé global à partir des résumés des échantillons
        concatenated_summaries = '. '.join(sample_summaries)
        global_prompt = f"{prompt2} pour l'ensemble des résumés : '{concatenated_summaries}.'"
        global_summary = call_model(global_prompt)

        # Ajouter le thème global au cluster
        themes[cluster_index] = global_summary

    # Convertir les résultats en DataFrame
    result_df = pd.DataFrame({
        f"{cluster_column}": valeurs_uniques,
        f"Thème_{cluster_column}": themes
    })
    data_complete2 = pd.merge(df, result_df, on = cluster_column, how = "left" )
    return (data_complete2, result_df)

def creation_theme(data_complete, cluster, text,prompt,prompt2, k = 1, n = 60):
    for elem in cluster:
        print(elem)
        data_complete, theme_test_cluster6 = generate_cluster_samples(data_complete, elem, text,prompt,prompt2, k = 1, n = 60)
    return data_complete





def chercher_distance_clusters(data, df_embedding, cluster_col, key=False):
    """
    Calcul des distances entre les clusters et hiérarchisation des classes.
    
    Arguments :
    - data : DataFrame avec les informations des points.
    - df_embedding : DataFrame avec les embeddings des points.
    - cluster_col : Colonne qui contient les labels des clusters.
    - key : Clé pour trier les données (par défaut False).
    
    Retourne :
    - Une liste hiérarchisée des clusters, avec les plus proches ensemble.
    """
    # Combinaison des données avec les embeddings
    if not key:
        embedding_et_data = pd.concat([data, df_embedding], axis=1)
    else:
        data_sorted = data.sort_values(by=key)
        embedding_et_data = pd.concat([data_sorted, df_embedding], axis=1)

    # Identifier les clusters uniques
    unique_clusters = embedding_et_data[cluster_col].unique()
    unique_clusters = [x for x in unique_clusters if not np.isnan(x)]    
    unique_clusters.sort()  # Trier les clusters
    # Calcul des points moyens pour chaque cluster
    points_moyens = []
    nrow_data = data.shape[1]
    for k in unique_clusters:
        embedding_et_data2 = embedding_et_data[embedding_et_data[cluster_col] == k]
        embedding_et_data2 = embedding_et_data2.iloc[:, nrow_data:]
        points_moyens.append(embedding_et_data2.mean(axis=0))

    # Calcul de la matrice des distances entre clusters
    distances = cdist(points_moyens, points_moyens, metric='euclidean')
    # Hiérarchisation des clusters en fonction des distances
    cluster_hierarchy = {}
    for i, cluster in enumerate(unique_clusters):
        cluster_hierarchy[cluster] = distances[i]

    # Tri des clusters par proximité (distances)
    sorted_clusters = sorted(cluster_hierarchy, key=lambda x: np.sum(cluster_hierarchy[x]))
    return sorted_clusters

def generate_samples(data, Cluster, numero, commentaire, prompt):
    """
    Génère un dataframe contenant des thèmes résumés pour chaque cluster.
    
    :param data: DataFrame contenant les données.
    :param cluster_column: Nom de la colonne indiquant les clusters.
    :param text_column: Nom de la colonne contenant les textes à concaténager.
    :param prompt: Prompt à utiliser pour les appels au modèle.
    :param k: Nombre d'échantillons à générer par cluster. Si 0, calculé dynamiquement.
    :param n: Nombre de lignes à échantillonner par répétition.
    :return: DataFrame contenant les clusters et leurs thèmes résumés.
    """
    # Obtenir les clusters uniques
    filtered_df = data[data[Cluster] == numero]
    filtered_df[commentaire] = filtered_df[commentaire].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    prompt_total = f"{prompt} : '{filtered_df[commentaire]}.'"
    reponse = call_model(prompt_total)
    return(reponse)

def cherche_commentaire(data, df_emebdding, Cluster, numero, commentaire, key = False):
    if key == False :
        embedding_et_data = pd.concat([data, df_emebdding], axis = 1)
    else :
        data_sort = data_complete.sort_values(by = key)
        embedding_et_data = pd.concat([data_sort, df_emebdding], axis = 1)
    nrow_data = data.shape[1]
    embedding_et_data2 = embedding_et_data[embedding_et_data[Cluster] == numero]
    embedding_et_data2 = embedding_et_data2.iloc[:, nrow_data:]
    point_moyen = embedding_et_data2.mean(axis=0)
    distances = np.sqrt(((embedding_et_data2 - point_moyen) ** 2).sum(axis=1))
    indice_plus_proche = distances.idxmin()
    return embedding_et_data[commentaire].loc[indice_plus_proche]



# Fonction pour dessiner une courbe de Bézier
def draw_bezier_curve(ax, start, end, nom_cluster, cluster, data):
    """Dessine une courbe de Bézier cubique entre deux points."""
    control1 = (start[0] + 1, start[1])  # Premier point de contrôle
    control2 = (end[0] - 1, end[1])      # Deuxième point de contrôle
    vertices = [start, control1, control2, end]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    path = Path(vertices, codes)
    couleur =  data.loc[data[nom_cluster] == cluster, f'{nom_cluster}_Color'].iloc[0]
    patch = PathPatch(path, lw=1.5, edgecolor=couleur, facecolor="none", alpha=0.7)
    ax.add_patch(patch)

def calculate_node_positions(df_sorted, Clusters, theme, x_spacing, y_spacing, embedding, commentaire=False, key=False):
    """Calcule les positions des nœuds pour chaque colonne hiérarchique."""
    positions = {}
    node_positions = {}
    print(Clusters)
    for i in range(len(Clusters) - 1, -1, -1):

        col = Clusters[i]
        theme2 = theme[i]
        x_pos = x_spacing * i
        if i == len(Clusters) - 1:
            # Dernière colonne : répartir uniformément les nœuds
            unique_nodes = df_sorted[col].drop_duplicates()
            y_positions = range(len(unique_nodes))
            for y, node in zip(y_positions, unique_nodes):
                if commentaire:
                    theme_value = f"{df_sorted.loc[df_sorted[col] == node, theme2].iloc[0]} ({i}_{node})"
                    com = cherche_commentaire(df_sorted, embedding, col, node, commentaire)
                    positions[theme_value] = [(x_pos, -y * y_spacing), col, node, com, 0]
                    node_positions.setdefault(col, {})[node] = -y * y_spacing
                else:
                    theme_value = f"{df_sorted.loc[df_sorted[col] == node, theme2].iloc[0]} ({i}_{node})"
                    positions[theme_value] = [(x_pos, -y * y_spacing), col, node, com, 0]
                    node_positions.setdefault(col, {})[node] = -y * y_spacing
        else:
            # Autres colonnes : calculer la position des parents
            child_col = Clusters[i + 1]
            parent_to_children = df_sorted[[col, child_col, theme2]].drop_duplicates().groupby(col)[child_col].apply(list)
            for parent, children in parent_to_children.items():
                child_positions = [node_positions[child_col][child] for child in children]
                theme_value = f"{df_sorted.loc[df_sorted[col] == parent, theme2].iloc[0]} ({i}_{parent})"
                parent_y = sum(child_positions) / len(child_positions)  # Moyenne des enfants
                # Si ce n'est pas la première colonne, chercher le parent du parent
                if i > 0:  # Vérfier qu'on n'est pas à la première colonne
                    grandparent_col = Clusters[i - 1]  # Colonne du parent du parent
                    grandparent = df_sorted.loc[df_sorted[col] == parent, grandparent_col].iloc[0]  # Parent du parent
                    grandparent_children = df_sorted[df_sorted[grandparent_col] == grandparent][col].unique()  # Enfants du parent du parent
                    grandparent_child_count = len(grandparent_children)  # Nombre d'enfants du parent du parent
                else:
                    grandparent = None
                    grandparent_child_count = 0
                positions[theme_value] = [
                    (x_pos, parent_y),  # Position (x, y)
                    col,  # Colonne actuelle
                    parent,  # Nom du parent actuel
                    None,  # Pas de commentaire
                    grandparent_child_count  # Nombre d'enfants du parent du parent
                    ]
                node_positions.setdefault(col, {})[parent] = parent_y

    return positions, node_positions


def draw_nodes_and_connections(ax, positions, df_sorted, Clusters, theme, invisible = True, commentaire = False):
    """Dessine les nœuds et les connexions entre les nœuds."""
    # Dessiner les nœuds
    first_cluster = Clusters[0]
    last_cluster = Clusters[-1]
    if invisible != True:
        for node, coord in positions.items():
            if commentaire != False and coord[1] == Clusters[len(Clusters)-1]:
                ax.text(coord[0][0], coord[0][1], node, ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
                ax.text(coord[0][0] + 20, coord[0][1], coord[3], ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
            elif coord[4] == 1:
                continue
            else:
                ax.text(coord[0][0], coord[0][1], node, ha='center', va='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))  
    else:

        # Dessiner les nœuds
        for node, coord in positions.items():
            if commentaire != False and coord[1] == Clusters[len(Clusters)-1]:
                # Afficher le label uniquement si le cluster est le premier ou le dernier
                if str(coord[1]) == str(first_cluster) or str(coord[1]) == str(last_cluster):
                    ax.text(coord[0][0], coord[0][1], node, ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
                    ax.text(coord[0][0] + 10, coord[0][1], coord[3], ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
                elif coord[4] == 1:
                    continue
                else:
                    # Si ce n'est pas le premier ou dernier cluster, ne pas afficher de texte
                    ax.text(coord[0][0], coord[0][1], "", ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
            else:
                if str(coord[1]) == str(first_cluster) or str(coord[1]) == str(last_cluster):
                    ax.text(coord[0][0], coord[0][1], node, ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
                elif coord[4] == 1:
                    continue
                else:
                    # Si ce n'est pas le premier ou dernier cluster, ne pas afficher de texte
                    ax.text(coord[0][0], coord[0][1], "", ha='center', va='center', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))
    # Dessiner les connexions
    for i in range(len(Clusters) - 1):
        print(i)
        parent_col = Clusters[i]
        child_col = Clusters[i + 1]
        theme_parent = theme[i]
        theme_child = theme[i + 1]

        # Créer des relations basées sur les thèmes
        rel = df_sorted[[parent_col, child_col, theme_parent, theme_child]].drop_duplicates()
        for parent, child, parent_theme, child_theme in rel.values:
            parent_theme_with_id = f"{parent_theme} ({i}_{parent})"
            child_theme_with_id = f"{child_theme} ({i+1}_{child})"
            print(child_theme_with_id)
            print(parent_theme_with_id)
            if parent_theme_with_id in positions and child_theme_with_id in positions:
                start = positions[parent_theme_with_id][0]
                end = positions[child_theme_with_id][0]
                draw_bezier_curve(ax, start, end, child_col, child, df_sorted)


def generate_cluster_graphs(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, invisible = False, commentaire = False, key = False, color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f58231', '#ff0033', 
           '#33ff77', '#ff66cc', '#ffcc00', '#0066cc', '#33ccff', '#6600cc', 
           '#00ffcc', '#ff9966']):
    """Génère des graphiques pour chaque cluster en fonction de la hiérarchie."""
    clusters_6 = data_complete[cluster_début].unique()
    if all(isinstance(item, (int, float)) for item in Clusters):
        Clusters = data_complete.columns[Clusters]
        Clusters = Clusters.tolist()
    elif all(isinstance(item, str) for item in Clusters):
        Clusters = Clusters
    else:
        return(print("il y a un problème dans les classes que vous avez donnés"))
    if all(isinstance(item, (int, float)) for item in theme):
        theme = data_complete.columns[theme]
        theme = theme.tolist()
    elif all(isinstance(item, str) for item in theme):
        
        theme = theme
    else:
        return(print("il y a une erreur dans les thèmes que vous avez données"))
    #créer une copie de cluster car on en a besoin pour faire les couleurs 
    Clusters_pur = Clusters.copy()
    Clusters = couper_liste_a_partir_de_valeur(Clusters, cluster_fin)
    couleur_parents = parent_color(color, data_complete, Clusters_pur)
    data_complete = children_color(couleur_parents, data_complete, Clusters_pur)
    ordre_colonne_1 = chercher_distance_clusters(data_complete, embedding, Clusters[0], key)
    for cluster_ in clusters_6:
        # Filtrer les données pour ce cluster principal
        subset = data_complete[data_complete[cluster_début] == cluster_]

        # Trier les données par les colonnes de classification
        # Filtrer les données pour ce cluster principal

        subset['custom_sort'] = subset.iloc[:, 0].map({val: i for i, val in enumerate(ordre_colonne_1)})
        # Trier les données par la colonne 'custom_sort' et ensuite par les autres colonnes de Clusters
        subset = subset.sort_values(by=['custom_sort'] + Clusters)
        subset = subset.drop(columns=['custom_sort'])

        # Calculer les positions des nœuds
        x_spacing = 20  # Espacement horizontal entre colonnes
        y_spacing = 5  # Espacement vertical entre nœuds
        positions, node_positions = calculate_node_positions(subset, Clusters, theme, x_spacing, y_spacing,embedding, commentaire, key)

        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 8))

        # Dessiner les nœuds et les connexions
        draw_nodes_and_connections(ax, positions, subset, Clusters, theme, invisible, commentaire)

        # Configurer l'affichage
        ax.set_xlim(-1, x_spacing * len(Clusters))
        ax.set_ylim(-len(node_positions[Clusters[-1]]) * y_spacing - 1, 2)
        ax.axis("off")
        plt.title(f"Graphique pour Cluster = {cluster_}", fontsize=14)
        plt.tight_layout()
        plt.show()

def generate_one_graphe(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, invisible = False, commentaire = False, key = False, color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f58231', '#ff0033', 
           '#33ff77', '#ff66cc', '#ffcc00', '#0066cc', '#33ccff', '#6600cc', 
           '#00ffcc', '#ff9966']):
    """Génère des graphiques pour chaque cluster en fonction de la hiérarchie."""

    if all(isinstance(item, (int, float)) for item in Clusters):

        Clusters = data_complete.columns[Clusters]
        Clusters = Clusters.tolist()
    elif all(isinstance(item, str) for item in Clusters):
        Clusters = Clusters
    else:
        return(print("il y a un problème dans les classes que vous avez donnés"))
    if all(isinstance(item, (int, float)) for item in theme):
        theme = data_complete.columns[theme]
        theme = theme.tolist()
    elif all(isinstance(item, str) for item in theme):
        
        theme = theme
    else:
        return(print("il y a une erreur dans les thèmes que vous avez données"))
    Clusters_pur = Clusters.copy()
    Clusters = couper_liste_a_partir_de_valeur(Clusters, cluster_fin)
    couleur_parents = parent_color(color, data_complete, Clusters_pur)
    data_complete = children_color(couleur_parents, data_complete, Clusters_pur)

    # Filtrer les données pour ce cluster principal
    ordre_colonne_1 = chercher_distance_clusters(data_complete, embedding, Clusters[0], key)
    data_complete['custom_sort'] = data_complete.iloc[:, 0].map({val: i for i, val in enumerate(ordre_colonne_1)})
    # Trier les données par la colonne 'custom_sort' et ensuite par les autres colonnes de Clusters
    df_sorted = data_complete.sort_values(by=['custom_sort'] + Clusters)
    df_sorted = df_sorted.drop(columns=['custom_sort'])
    # Calculer les positions des nœuds
    x_spacing = 20  # Espacement horizontal entre colonnes
    y_spacing = 5  # Espacement vertical entre nœuds
    positions, node_positions = calculate_node_positions(df_sorted, Clusters, theme, x_spacing, y_spacing, embedding, commentaire, key)

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))

    # Dessiner les nœuds et les connexions
    draw_nodes_and_connections(ax, positions, df_sorted, Clusters, theme, invisible, commentaire)

    # Configurer l'affichage
    ax.set_xlim(-1, x_spacing * len(Clusters))
    ax.set_ylim(-len(node_positions[Clusters[-1]]) * y_spacing - 1, 2)
    ax.axis("off")
    plt.title("Graphique pour Cluster = Graphe total", fontsize=14)
    plt.tight_layout()
    plt.show()

def generate_graphe(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, graphe = True, invisible = False, commentaire = False, key = False, color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f58231', '#ff0033', 
           '#33ff77', '#ff66cc', '#ffcc00', '#0066cc', '#33ccff', '#6600cc', 
           '#00ffcc', '#ff9966']):
    if graphe == True:
        generate_cluster_graphs(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, invisible, commentaire, key, color )
    else:
        generate_one_graphe(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, invisible, commentaire, key, color )


def couper_liste_a_partir_de_valeur(liste, valeur_cible):
    """
    Coupe une liste à partir de la première occurrence d'une valeur donnée.

    Args:
        liste (list): La liste d'origine.
        valeur_cible (str): La valeur à partir de laquelle couper la liste.

    Returns:
        list: La liste coupée.
    """
    if valeur_cible in liste:
        index_cible = liste.index(valeur_cible)
        return liste[:index_cible+1]
    else:
        return []  # Retourne une liste vide si la valeur n'est pas trouvée
    
    
def couper_liste_a_partir_de_valeur_inverse(liste, valeur_cible):
    """
    Coupe une liste en excluant tous les éléments jusqu'à la première occurrence 
    de la valeur cible (y compris la valeur cible elle-même), et retourne 
    tous les éléments après cette valeur.

    Args:
        liste (list): La liste d'origine.
        valeur_cible (str): La valeur à partir de laquelle garder les éléments.

    Returns:
        list: La liste coupée, contenant les éléments après la valeur cible.
    """
    if valeur_cible in liste:
        index_cible = liste.index(valeur_cible)
        return liste[index_cible+1:]  # Retourne les éléments après la valeur cible
    else:
        return []  # Retourne une liste vide si la valeur n'est pas trouvée



def parent_color(color, data, Clusters):
    len(Clusters)
    nb_color = len(color)
    nb_class = 0
    
    for k in range(1,len(Clusters)):
        nb_class = nb_class + len(data[Clusters[k]].unique())
        dernière_class = k
        if nb_class > nb_color:
            nb_class = nb_class - len(data[Clusters[k]].unique())
            break
    color = couper_liste_a_partir_de_valeur(color, color[nb_class-1])
    dic = {}
    for k in range(dernière_class):
        if k == 0:
            dic[f'{Clusters[k]}'] =  9
        else: 
            color2 = couper_liste_a_partir_de_valeur(color, color[len(data[Clusters[k]].unique())-1])
            color = couper_liste_a_partir_de_valeur_inverse(color,color[len(data[Clusters[k]].unique())-1] )
            dic[f'{Clusters[k]}'] =  color2
    return dic


def children_color(dictionary, data, Clusters):
    longueur_dict = len(dictionary)
    longueur_class = len(Clusters)
    data2 = data.copy()

    # Ajout des colonnes de couleur pour chaque clé dans le dictionnaire
    for k in dictionary:
        liste = [data_complete[k].unique(), dictionary[k]]
        df = pd.DataFrame({f'{k}': liste[0], f'{k}_Color': liste[1]})
        data2 = pd.merge(data2, df, on=k, how = "left")
    if longueur_class != longueur_dict:
        # Parcourir les colonnes restantes
        for i in range(longueur_dict, len(Clusters)):
            liste_couleur = []
            liste_numero = []
            for elem in data2[Clusters[i]].unique():
                liste_numero.append(elem)
                # Vérifier si une ligne correspondante existe
                matching_rows = data2.loc[data2[Clusters[i]] == elem, f'{Clusters[i-1]}_Color']
                if not matching_rows.empty:
                    parent = matching_rows.iloc[0]  # Prendre la première ligne correspondante
                    liste_couleur.append(parent)
                else:
                    liste_couleur.append(None)  # Ajouter une valeur par défaut en cas d'absence de parent
            liste = [data_complete[k].unique(), dictionary[k]]
            df = pd.DataFrame({f'{Clusters[i]}': liste_numero, f'{Clusters[i]}_Color': liste_couleur})
            data2 = pd.merge(data2, df, on=Clusters[i], how = 'left')
    return(data2)


# Utilisation de la fonction principale





# Liste des couleurs hexadécimales
couleur = ['#df187f', '#3f67b6', '#23253e', '#eade84', '#b840e9', '#86466d', 
 '#eb586f', '#3522f9', '#adfb8a', '#4c6bf7', '#aa2fd4']

# Convertir chaque couleur hexadécimale en format RGBA
couleurs_rgba = [colors.to_rgba(couleur) for couleur in couleur]

    
data_complete2 = creation_theme(data_complete, Cluster, text,prompt,prompt2, k = 1, n = 60)
test2 = generate_graphe(data_complete2, Cluster, theme, Cluster_debut, Cluster_fin, Embeds,graphe = True, invisible = False, commentaire = text, key = "key", color = couleurs_rgba)


import plotly.graph_objects as go

def generate_interactive_graph(data_complete, Clusters, theme, cluster_début, cluster_fin, embedding, commentaire=False, key = False,
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                                      '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f58231', '#ff0033', 
                                      '#33ff77', '#ff66cc', '#ffcc00', '#0066cc', '#33ccff', '#6600cc', 
                                      '#00ffcc', '#ff9966'], prob=0.05):
    """
    Génère un graphique interactif des clusters en intégrant l'analyse des variances quantitatives et qualitatives.
    
    Parameters:
        data_complete (pd.DataFrame): Le DataFrame d'origine avec les données complètes.
        Clusters (list): Liste des colonnes représentant les clusters (noms de colonnes).
        theme (list): Liste des colonnes représentant les thèmes associés.
        cluster_début (str): Nom de la colonne où commence l'analyse des clusters.
        cluster_fin (str): Nom de la colonne où se termine l'analyse des clusters.
        commentaire (str, optional): Colonne contenant les commentaires supplémentaires.
        color (list, optional): Liste des couleurs pour la visualisation.
        prob (float, optional): Niveau de probabilité pour la significativité.

    Returns:
        None: Affiche directement le graphique interactif.
    """
    # Vérification que Clusters contient des noms de colonnes (chaînes de caractères)
    if not all(isinstance(item, str) for item in Clusters):
        if all(isinstance(item, int) for item in Clusters):
            Clusters = data_complete.columns[Clusters].tolist()
        else:
            raise ValueError("Clusters doit être une liste de noms de colonnes (strings) ou d'indices.")

    # Vérification que theme contient bien des noms de colonnes (strings)
    if not all(isinstance(item, str) for item in theme):
        if all(isinstance(item, int) for item in theme):
            theme = data_complete.columns[theme].tolist()
        else:
            raise ValueError("Theme doit être une liste de noms de colonnes (strings) ou d'indices.")
    
    # Créer une liste de toutes les variables qui ne sont pas des colonnes de type 'Cluster' ou 'theme' 
    columns_to_drop = Clusters + theme
    if commentaire:
        columns_to_drop.append(commentaire)

    # Sélectionner toutes les colonnes sauf celles dans columns_to_drop
    variable_names = [col for col in data_complete.columns if col not in columns_to_drop and "Cluster" not in col]

    # On fait une copie de Clusters pour l'utilisation future (pur) si nécessaire
    Clusters_pur = Clusters.copy()

    # Élagage de la liste des clusters (couper la liste à partir de la valeur cluster_fin)
    Clusters = couper_liste_a_partir_de_valeur(Clusters, cluster_fin)


    temp_data = data_complete.copy()
    # Étape 1 : Analyse et enrichissement pour chaque variable dans Clusters
    for cluster_var in Clusters:

        # Copie du DataFrame pour chaque itération afin de ne pas affecter les suivantes

        # Étape 1.1 : Analyse des variances quantitatives et qualitatives
        quanti, quali = catdes_du_pauvre(temp_data, cluster_var, prob, Clusters, theme, commentaire)

        # Étape 1.2 : Enrichissement des données avec les modalités significatives
        catdes_df = catedes_colonne(temp_data, quanti, quali, cluster_var)


        # Étape 1.3 : Calculer les colonnes en plus dans catdes_df par rapport à temp_data
        num_temp_data_cols = temp_data.shape[1]
        num_catdes_df_cols = catdes_df.shape[1]

        if num_catdes_df_cols > num_temp_data_cols:
            # On récupère les colonnes en trop dans catdes_df
            new_columns = catdes_df.columns[num_temp_data_cols:]
            jpp = catdes_df[[cluster_var] + list(new_columns)]
            # Fusionner les nouvelles colonnes à data_complete
            data_complete = pd.concat([data_complete, jpp.drop(cluster_var, axis=1)], axis=1)

        # Vérification après la fusion des nouvelles colonnes
    
    # Calcul des couleurs pour les parents et enfants
    couleur_parents = parent_color(color, data_complete, Clusters_pur)
    data_complete = children_color(couleur_parents, data_complete, Clusters_pur)
    
    # Étape 2 : Trier les données
    ordre_colonne_1 = chercher_distance_clusters(data_complete, embedding, Clusters[0], key)
    data_complete['custom_sort'] = data_complete.iloc[:, 0].map({val: i for i, val in enumerate(ordre_colonne_1)})
    # Trier les données par la colonne 'custom_sort' et ensuite par les autres colonnes de Clusters
    df_sorted = data_complete.sort_values(by=['custom_sort'] + Clusters)
    df_sorted = df_sorted.drop(columns=['custom_sort'])
    
    # Calcul des positions des nœuds
    x_spacing = 20
    y_spacing = 10
    positions, node_positions = calculate_node_positions(df_sorted, Clusters, theme, x_spacing, y_spacing, embedding, commentaire, key)

    # Initialisation de la figure
    fig = go.Figure()

    # Ajouter les nœuds au scatter plot
    for node, coord in positions.items():
        x, y = coord[0]
        # Créer le texte de survol (hovertext) pour chaque nœud
        hover_text = f"{node}"

        # Sélectionner les colonnes correspondantes aux Modalités et aux Significatives
        for variable in variable_names:
            for prefix in ['Modalité', 'Significative_sentence']:
                # Créer le nom de la colonne correspondant à ce préfixe
                modality_col = f"{prefix}_{variable}_{coord[1]}"
                if modality_col in data_complete.columns:
                    # Récupérer la valeur dans la colonne
                    modality_value = df_sorted.loc[df_sorted[coord[1]] == coord[2], modality_col].iloc[0]
                    
                    # Si la valeur existe (non None/NaN), l'ajouter au texte de survol
                    if pd.notna(modality_value):
                        # Si c'est une valeur significative, on l'affiche
                        hover_text += f"<br>{prefix} pour {variable}: {modality_value}"

        # Ajouter le nœud avec les informations de survol
        if commentaire and len(coord) > 3:
            hover_text += f"<br>{coord[3]}"
        if coord[4] == 1:
            continue
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            text=[node],
            hovertext=hover_text,  # Ajouter le hovertext avec les infos significatives
            textposition="bottom center",
            marker=dict(size=10, color="blue")
        ))

    # Ajouter les courbes de Bézier (restant inchangé)
    for i in range(len(Clusters) - 1):
        parent_col = Clusters[i]
        child_col = Clusters[i + 1]
        theme_parent = theme[i]
        theme_child = theme[i + 1]

        relations = df_sorted[[parent_col, child_col, theme_parent, theme_child]].drop_duplicates()
        for parent, child, parent_theme, child_theme in relations.values:
            parent_key = f"{parent_theme} ({i}_{parent})"
            child_key = f"{child_theme} ({i+1}_{child})"

            if parent_key in positions and child_key in positions:
                start = positions[parent_key][0]
                end = positions[child_key][0]
                
                # Approximation d'une courbe de Bézier
                control1 = (start[0] + 1, start[1])
                control2 = (end[0] - 1, end[1])
                t = np.linspace(0, 1, 100)
                bezier_x = (1 - t)**3 * start[0] + 3 * (1 - t)**2 * t * control1[0] + 3 * (1 - t) * t**2 * control2[0] + t**3 * end[0]
                bezier_y = (1 - t)**3 * start[1] + 3 * (1 - t)**2 * t * control1[1] + 3 * (1 - t) * t**2 * control2[1] + t**3 * end[1]

                # Ajouter la courbe à la figure
                fig.add_trace(go.Scatter(
                    x=bezier_x,
                    y=bezier_y,
                    mode='lines',
                    line=dict(color=df_sorted.loc[df_sorted[child_col] == child, f'{child_col}_Color'].iloc[0], width=1),
                    hoverinfo='skip'
                ))

    # Configurer la mise en page
    fig.update_layout(
        title="Graphe interactif des clusters",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )

    pio.renderers.default = "browser"
    fig.show()
    return(data_complete)



def variance_category(data, target_var, proba=0.05, na_method="NA", threshold=6):
    """
    Analyse les relations entre une variable cible catégorielle et les autres variables catégorielles
    détectées dans un DataFrame en utilisant un test du chi-carré.

    Parameters:
        data (pd.DataFrame): Le DataFrame à analyser.
        target_var (str): Nom de la variable cible.
        proba (float): Seuil de significativité pour les tests (alpha).
        na_method (str): Méthode de gestion des valeurs manquantes ("NA" ou "na.omit").
        threshold (int): Le seuil pour considérer une variable comme catégorielle.

    Returns:
        dict: Résultats du test du chi-carré et détails des modalités significatives.
    """
    # Identifier les variables catégorielles et continues
    categorical_vars, _ = detect_variable_types(data, threshold=threshold)

    # S'assurer que la variable cible est convertie en type catégoriel
    if target_var not in data.columns:
        raise ValueError(f"La variable cible '{target_var}' n'existe pas dans le DataFrame.")
    data[target_var] = data[target_var].astype("category")

    # Retirer la variable cible de la liste des autres variables catégorielles
    categorical_vars = [col for col in categorical_vars if col != target_var]
    # Conversion des autres variables catégorielles en type catégoriel
    data[categorical_vars] = data[categorical_vars].astype("category")

    # Initialisation des résultats
    results = {}
    target_levels = data[target_var].cat.categories

    for var in categorical_vars:
        results[var] = {}
        contingency_table = pd.crosstab(data[target_var], data[var])

        # Effectuer le test du chi-carré
        chi2, p_value, _, expected = chi2_contingency(contingency_table)
        results[var]['p_value'] = p_value
        results[var]['chi2_stat'] = chi2

        if p_value <= proba:
            results[var]['significant_modalities'] = []
            observed = contingency_table.values

            for i, target_modality in enumerate(target_levels):
                for j, var_modality in enumerate(data[var].cat.categories):
                    obs = observed[i, j]
                    exp = expected[i, j]

                    # Calcul de l'écart standardisé
                    std_diff = (obs - exp) / np.sqrt(exp)
                    
                    # Calcul de la p-valeur pour cette cellule
                    cell_p_value = 2 * norm.sf(abs(std_diff))
                    
                    if cell_p_value <= proba:
                        significance = "Over-represented" if std_diff > 0 else "Under-represented"
                        
                        # Ajouter à la liste des modalités significatives
                        results[var]['significant_modalities'].append({
                            'target_modality': target_modality,  # Cette modalité peut se répéter
                            'var_modality': var_modality,
                            'observed': obs,
                            'expected': exp,
                            'std_diff': std_diff,
                            'p_value': cell_p_value,
                            'significance': significance
                        })

    return results



def variance_quanti(data, variable_quali, continuous_vars, prob):

    results = []
    for modality in data_complete[variable_quali].unique():
        # Sous-échantillon pour la modalité actuelle
        subset = data_complete[data_complete[variable_quali] == modality]
        
        # Parcourir chaque variable continue
        for var in continuous_vars:
            # Moyenne pour la modalité actuelle
            mean_modality = subset[var].mean()
            
            # Moyenne générale
            mean_total = data_complete[var].mean()
            
            # Écart-type général
            std_total = data_complete[var].std()
            
            # Taille des groupes
            n_modality = len(subset)
            n_total = len(data_complete)
            
            # Fréquence relative de la modalité
            freq_modality = n_modality / n_total
            
            # Calcul du V-test
            v_test = (mean_modality - mean_total) / (std_total * np.sqrt(freq_modality))
            
            # Calcul de la p-valeur (bilatérale)
            p_value = 2 * norm.sf(abs(v_test))
            if p_value < prob:
                # Stocker les résultats
                results.append({
                    'Modality': modality,
                    'Variable': var,
                    'V-test': v_test,
                    'P-value': p_value,
                    'Moyenne de la modalité' : mean_total,
                    'Moyenne totale' : mean_total
                })
        return(results)
def detect_variable_types(data, threshold=5):
    """
    Détecte si les variables d'un DataFrame sont continues ou catégorielles et les sépare en deux listes.

    Parameters:
        data (pd.DataFrame): Le DataFrame à analyser.
        threshold (int): Le seuil pour considérer une variable comme catégorielle
                         (si le nombre de valeurs uniques est inférieur ou égal à ce seuil).

    Returns:
        tuple: Deux listes, la première contenant les colonnes catégorielles et la seconde les colonnes continues.
    """
    categorical = []
    continuous = []
    
    for col in data.columns:
        # Si le type est object ou catégorique, ou si le nombre de valeurs uniques est inférieur au seuil
        if data[col].dtype == 'object' or data[col].dtype.name == 'category' or data[col].nunique() <= threshold:
            categorical.append(col)
        else:
            continuous.append(col)
    
    return categorical, continuous

def catdes_du_pauvre(data, variable, prob, Cluster, theme, text):
    # Supprime l'élément 'variable' de la liste Cluster
    Cluster = [item for item in Cluster if item != variable]
    
    # Combine Cluster, theme et text pour les colonnes à supprimer
    columns_to_drop = Cluster + theme
    columns_to_drop.append(text)
    
    # Vérifier si les colonnes existent dans le DataFrame avant de les supprimer
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    # Supprime les colonnes du DataFrame
    data = data.drop(columns=columns_to_drop, axis=1)
    
    # Détecte les variables catégoriques et continues
    categorial_df, continuous_df = detect_variable_types(data)
    
    # Analyse des variances quantitatives et qualitatives
    quanti = variance_quanti(data, variable, continuous_df, prob)
    quali = variance_category(data, variable, prob)
    return quanti, quali





def catedes_colonne(data, quanti, quali, variable):
    """
    Génère un tableau synthétique pour les modalités significatives d'une variable donnée,
    puis fusionne les résultats avec le DataFrame d'origine.

    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les données.
        quanti (dict): Un dictionnaire contenant des informations sur les variables quantitatives (non utilisé ici).
        quali (dict): Un dictionnaire contenant des informations sur les variables qualitatives et leurs modalités significatives.
        variable (str): La variable cible pour laquelle analyser les modalités.

    Returns:
        pd.DataFrame: Le DataFrame enrichi avec les colonnes des modalités significatives.
    """
    # Copie pour conserver l'original intact
    merged_data = data.copy()
    valeurs_uniques = data[variable].unique().tolist()  # Modalités uniques de la variable cible
    
    for k in quali:  # Parcourt chaque variable catégorielle dans quali
        # Vérifie si significant_modalities est présent et non vide
        if "significant_modalities" not in quali[k] or not quali[k]["significant_modalities"]:
            continue
        
        # Initialisation des listes pour les données de sortie
        modalite = [[] for _ in range(len(valeurs_uniques))]
        signif_value = [[] for _ in range(len(valeurs_uniques))]
        signif_sentence = [[] for _ in range(len(valeurs_uniques))]
        
        # Création d'un dictionnaire pour stocker les associations target_modality -> [(var_modality, significance)]
        target_modality_dict = {}

        # Parcourt les modalités significatives
        for l in quali[k]["significant_modalities"]:
            target_modality = l["target_modality"]
            var_modality = l["var_modality"]
            significance = l["significance"]
            std_diff = l["std_diff"]
            # Ajouter aux listes

            
            if target_modality not in target_modality_dict:
                target_modality_dict[target_modality] = []
            target_modality_dict[target_modality].append((var_modality, significance, std_diff))

        # Maintenant on remplit les listes modalite, signif_value, signif_sentence
        for i, target_modality in enumerate(valeurs_uniques):
            if target_modality in target_modality_dict:
                # Récupère toutes les modalités et significations pour cette target_modality
                var_modalities = []
                significances = []
                std_diffs = []

                for var_modality, significance, std_diff in target_modality_dict[target_modality]:
                    var_modalities.append(var_modality)
                    significances.append(significance)
                    std_diffs.append(std_diff)
                # Ajouter aux listes
                modalite[i] = "; ".join(map(str, var_modalities)) if var_modalities else None
                signif_value[i] = "; ".join(map(str, std_diffs)) if std_diffs else None
                
                signif_sentence[i] = "; ".join(map(str, significances)) if significances else None
        
        # Création du DataFrame pour la variable catégorielle actuelle
        df = pd.DataFrame({
            f"{variable}": valeurs_uniques,
            f"Modalité_{k}_{variable}": modalite,
            f"Significative_value_{k}_{variable}": signif_value,
            f"Significative_sentence_{k}_{variable}": signif_sentence
        })
        
        # Fusionne avec le DataFrame d'origine
        merged_data = pd.merge(merged_data, df, on=variable, how="left")
    
    return merged_data

generate_interactive_graph(data_complete2, Cluster, theme, Cluster_debut, Cluster_fin, Embeds, commentaire=text, key = "key")
