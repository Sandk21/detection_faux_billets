![image](https://github.com/user-attachments/assets/62406732-8067-4bc2-9d96-440062cd4ddc)

# Détection automatique de faux billets

[Lien vers l'application finalisée](https://detectionfauxbillets-eerqtpxtxetqwjdsg6l7jy.streamlit.app/)

Des fichiers csv sont mis à disposition dans ce [dossier](detection_faux_billets/tree/main/data).

## Contexte du projet
L'**Organisation Nationale de Lutte contre le Faux-Monnayage (ONCFM)**, organisme public, a pour mission de développer des méthodes avancées d’**identification des contrefaçons de billets** en euros.

Dans ce cadre, l'ONCFM souhaite développer un algorithme pour analyser les **caractéristiques géométriques des billets** 
et déterminer, sur la base de ces paramètres, s’il s’agit de billets authentiques ou falsifiés.

## Objectifs
Développer un **algorithme de machine learning** capable de distinguer les billets authentiques des contrefaçons en s'appuyant sur leurs caractéristiques géométriques.
Mettre cet algorithme à disposition des professionnels bancaires via une **application**, facilitant ainsi la détection des faux billets.

## Données
![{6FEDE686-7D5E-43B2-93CC-BBE781973EF4}](https://github.com/user-attachments/assets/7cdb97e1-1646-4501-a324-b4a123c523bc)

## Raodmap
![{930F8378-D2FC-4F71-802A-5AD4794F3836}](https://github.com/user-attachments/assets/c028a3f1-eaaa-4f3b-94ec-919e3ddbec74)
![{4DC49935-A7F1-4993-A09F-295CCA2A369E}](https://github.com/user-attachments/assets/9e33dfdd-d2be-42c8-8d1f-aae855f9ca36)


## Résultats
![image](https://github.com/user-attachments/assets/7605f6ef-f2c7-4909-9acd-7679cfd5a93d)

Modèle de classification retenu : la régression logistique a permis d'obtenir les meilleurs performance de classification pour les différentes métrics utilisées
(recall, score f1 en particulier)
![{098798EE-EA65-42A8-9FA4-3DFC36DFE992}](https://github.com/user-attachments/assets/813632e1-c8e8-446e-ab72-ef80d90a65e3)

## Livrables
Modèle de régression linéaire stocké au format pickle
Application de détection de faux billets avec streamlit
![image](https://github.com/user-attachments/assets/1771ad1c-b426-41e6-ae36-cdddd42e6781)

![image](https://github.com/user-attachments/assets/1ad8388c-2664-420e-a0e6-203cb9fdcee0)

![image](https://github.com/user-attachments/assets/86e3db4d-9e8d-44ce-915c-555214368fa9)


## 
>Le projet a été réalisé dans le cadre d'une formation de Data Analyst avec OpenClassroom
