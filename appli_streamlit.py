import os
import sys
import csv
import io
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



pd.options.display.precision =3

# Fonction pour détecter le séparateur
def detect_delimiter(uploaded_file):
    file_content = uploaded_file.getvalue().decode("utf-8")

    # Utilisation du Sniffer pour détecter le séparateur
    sample = io.StringIO(file_content)
    dialect = csv.Sniffer().sniff(sample.read(1024))
    return dialect.delimiter


def validate_file(file):
    filename = file.name
    name, ext = os.path.splitext(filename)
    if ext == ".csv":
        return ext
    else:
        return False
    
def validate_columns_df(df):
    # Vérification du nom des colonnes
    cols = [ "diagonal","is_genuine", "height_left", "height_right", "margin_low", "margin_up", "length"]
    if list(df.columns) == cols:
        st.write("coooool")
    else :
        print("Le fichier csv doit contenir les colonnes : {cols}")
        
@st.cache_data
def load_model():
    # Charger le modèle depuis le fichier pickle
    with open('modele.pickle', 'rb') as fichier:
        model = pickle.load(fichier)
        return model
    
@st.cache_data
def load_scaler():
    with open('scaler.pickle', 'rb') as fichier:
        scaler = pickle.load(fichier)
        return scaler

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


def highlight_false_billet(row):
    return ['background-color: orange' if row['classification'] == "faux billet" else '' for _ in row]

############################################


st.image("image/logo_ONCFM.png")
st.title("APPLICATION DE DETECTION DE FAUX BILLETS")

st.write("""L’Organisation nationale de lutte contre le faux-monnayage (ONCFM) est
une organisation publique ayant pour objectif de mettre en place des
méthodes d’identification des contrefaçons des billets en euros. Dans le
cadre de cette lutte, un algorithme qui est capable de différencier
automatiquement les vrais des faux billets a été élaboré.""", )


st.markdown("""
    La classification vrai ou faux billets est basée sur les caractéristiques géométriques des billets de banque :
            
    - Longueur en mm --> **lenght**            
    - Hauteur gauche en mm --> **height_left**            
    - Hauteur droite en mm --> **height_right**
    - Marge du bas en mm --> **margin_low**            
    - Marge du haut en mm --> **margin_up**            
    - diagonale en mm --> **diagonal**
""")
# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Charger le fichier contenant les dimensions des billets à prédire",
                                   accept_multiple_files=False,
                                   type=["csv"])


if uploaded_file is not None:
    sep = detect_delimiter(uploaded_file)
    st.write(sep)
    print(type(uploaded_file))
    validate_file(uploaded_file)
    df = pd.read_csv(uploaded_file, sep=sep)
    validate_columns_df(df)
    df = df.dropna(axis=0)


    st.success(f"Fichier {uploaded_file.name} téléchargé avec succès.")

    # Chargement des données
    st.write("### Données chargées :")
    st.dataframe(df.head())

    # Affichage la dimension du fichier de données
    dim = df.shape
    st.write(f" **Nombre de billets à détecter : {dim[0]}**\n")

    generate_predict = st.button("Générer les prédictions",
                                 use_container_width=True,
                                 type="primary")
    if generate_predict:
        # Select columns in DF
        columns_for_predict = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up',
       'length']
        df = df[columns_for_predict]
        # En arrière plan : chargement du scaler et du model

        scaler = load_scaler()
        model = load_model()
        # Scaling df
        df_scaled = scaler.transform(df)

        # Make prédictions
        predictions = model.predict(df_scaled)
        df["classification"] = predictions
        df['classification'] = df['classification'].apply(lambda x: 'vrai billet' if x == 1 else 'faux billet')

        # Stock predictions proba
        pred_proba = model.predict_proba(df_scaled)
        df['probabilites'] = pred_proba[:,0]
    

        # display dataframe with predictions
        st.write("### Tableau avec prédictions:")
        st.write("Les lignes orange indiquent les faux billets.")
        st.write("*La colonne 'probabilite' indique la probabilité que le billet soit faux\n(si = 0.9 alors 90% de chance que le billet soit faux).*")
        styled_df = df.style.apply(highlight_false_billet, axis=1).format(precision=2) # # Apply style to DataFrame
        st.dataframe(styled_df, use_container_width=True)

        #####################################################################
        # results
        st.write("---")
        st.write("### Résultats :")

        # create bar plot to synthetise the results
        fig, ax = plt.subplots(1,1)
        df["classification"].value_counts().plot.bar(color =["blue", "red"],ax=ax)
        # Ajouter les étiquettes de valeur
        for p in ax.patches:
            ax.annotate(f"{p.get_height()/len(df.classification):.1%}", (p.get_x()+0.17, p.get_height()-30,), color="white", weight="bold")
            ax.annotate(f"({p.get_height()})", (p.get_x()+0.18, p.get_height()-50), color="white", weight="bold", size=10)
        ax.set_title("Détection des faux billets dans les données chargées")
        ax.set_xlabel(None)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

        st.pyplot(fig) # display chart


        # Button - Download the dataframe with predictions
        csv = convert_df(df)
        st.download_button("Télechargez le fichier avec les prédictions  (.csv)",
                        data=csv,
                        file_name="test.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type = "primary",
                        )


else: 
    st.error("Veuillez charger un fichier .csv contenant les caractéristiques des billets")





