
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from  sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer




class PCA_Analyse():
    """Réalise une ACP en utilisant la bibliothèque sklearn et 
    calcule l'ensemble des données issues de l'ACP pour les inidividus,
    les variables et les facteurs
    """

    def __init__(self) -> None:
        """_summary_

        Raises:
            TypeError: _description_
            ValueError: _description_
            TypeError: _description_

        Returns:
            _type_: instancie un objet de classe PCA_Analyse
        """

        self.X = None
        self.n = None  # nb observations
        self.p = None  # nb variables
        self.labels = None # labels
        self.scaler = None  # scaler
        self.X_scaled = None  # X scaled
        self.X_transform = None
        self.eigval = None  # eigenvalues
        self.di = None  # individuals contributions vs total inertia
        self.cos2_ind = None  # individuals cosine²
        self.ctr = None  # individuals contributions for components
        self.corr_var = None  # variable correlation's for compenents
        self.cos2_var = None  # variable cosine²
        self.ctr_var = None  # variable contributions for components
        


    def __prepare__data(self) -> str:
        """Prépare les données en calculant n, p et labels

        Returns:
            str: Taille du dataset
        """
        self.n = self.X.shape[0]  # Nombre d'observations
        self.p = self.X.shape[1]  # Nombre de variables
        self.labels = self.X.index
        return print(f"Description de X : \nn (observations): {self.n} \np (variables): {self.p}")

    def __scale_data(self):
        # Instanciation
        # self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        print(f"Les données ont été mises à l'echelle avec {self.scaler}")

    def instance_fit_PCA(self, df: pd.DataFrame, scaler=StandardScaler()):
        """1 - Prépare les données : X, labels, n et p
        2 - Transforme les données avec le scaler de sklearn.preprocessing (par défaut = StandardScaler) : X_scaled
        3 - Insancie la classe sklearn.decomposition.PCA : X_transform
        4 - Récupère / calcul des différentes données relatives à l'ACP : eigval, d_i, cos2_ind, cos2_var, ...

        Args:
            df (pd.DataFrame): Dataframe sans les labels en index
            scaler (_type_, optional): Choix du scaler de sklearn.preprocessing. Defaults to StandardScaler().

        Raises:
            TypeError: Vérification du type du paramètre du Dafatframe
            ValueError: Vérification de l'absence de données non numériques
            TypeError: Vérfication du nom du scaler

        Returns:
            _type_: Affiche un message présicant la taille du data_set
        """

        try : 
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Le paramètre fourni n'est pas un DataFrame pandas8888888.") 
            if not all(df.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype))):
                raise ValueError("Toutes les colonnes ne sont pas de type numérique.")
            if not isinstance(scaler, (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer)):
                raise TypeError("Le scaler doit être une instance de StandardScaler ou MinMaxScaler.")
            self.scaler = scaler

        except Exception as e:
            print(f"Erreur: {e}")

        
        else:
            self.X = df
            self.__prepare__data()
            self.__scale_data()
            
            # Instanciation
            self.acp = PCA(svd_solver="full")
            print(f"Paramètre de la PCA :")
            print(self.acp.get_params())
            # Calculs
            self.X_transform = self.acp.fit_transform(self.X_scaled)
            if self.acp :
                print(f"L'entrainement des données avec l'ACP a été réalisée")
            else:
                print("L'ACP n'a pas fonctionné !")
            # valeurs propres
            self.eigval = (self.n-1)/self.n*self.acp.explained_variance_
            # contribution des individus dans l'inertie totale
            self.di = np.sum(self.X_scaled**2, axis=1)
            self.n_components = self.acp.n_components_
            # Calcul des cosinus² des individus
            self.__calculate_individuals_COS2()

            self.__calculate_individuals_contributions()
            self.__calculate_variable_representation()
            self.__calculate_variable_contributions_to_axes()
            return print(f"{"-"*100}\nTaille de : {self.X_transform.shape}\nNombre de composantes : {self.acp.n_components_}")
      

    def __calculate_individuals_COS2(self):
        """Calcul les corrélations(cos²) entre les individus et les composantes principales
        """
        self.cos2_ind = self.X_transform**2
        for j in range(self.p):
            self.cos2_ind[:, j] = self.cos2_ind[:, j]/self.di


    def __calculate_individuals_contributions(self):
        """Calcul les contributions des individus aux composantes principales        
        """
        self.ctr = self.X_transform**2
        for j in range(self.p):
            self.ctr[:, j] = self.ctr[:, j]/(self.n*self.eigval[j])

    def __calculate_variable_representation(self):
        """Calcul les corrélations (cos²) entre variables et composantes principales        
        """
        sqrt_eigval = np.sqrt(self.eigval)  # racine carrée des valeurs propres
        # corrélation des variables avec les axes
        self.corr_var = np.zeros((self.p, self.p))
        for k in range(self.p):
            self.corr_var[:, k] = self.acp.components_[k, :] * sqrt_eigval[k]   

    def __calculate_variable_contributions_to_axes(self):
        """Calcul les contributions des variables aux composantes principales        
        """
        self.cos2_var = np.square(self.corr_var)
        self.ctr_var = self.cos2_var
        for k in range(self.p):
            self.ctr_var[:, k] = self.ctr_var[:, k]/self.eigval[k]

    # -------------------------------------------------GET_DATA----------------------------------------------
    # get dataframe with indivuals and variables datas calculed

    def get_eigenvalues_(self
                         ):
        """Affiche les valeurs propres
        """
        print("Valeurs propres associées aux plans factoriels :\n", self.eigval)


    def get_explained_variance_(self):
        """Affiche la variance expliquée par les composantes principales
        """
        print("\nProportion de variance expliquée :\n",self.acp.explained_variance_ratio_)


    def get_individuals_inertia_vs_total(self) -> pd.DataFrame:
        """Retourne un dataframe avec la part d'inertie des individus sur le total

        Returns:
            pd.DataFrame: ID = labels, d_i= distance euclidienne
        """
        return pd.DataFrame({'ID': self.X.index, 'd_i': self.di})
    

    def get_individuals_COS2(self):
        
        """Retourne un dataframe avec les corrélations entre individus et composantes principales

        Returns:
            pd.DataFrame: index = labels, COS2_X= corrélation entre individus par composantes principales
        """
        return pd.DataFrame({str("COS2_" + str(factor+1)): self.cos2_ind[:, factor] for factor in range(self.cos2_ind.shape[1])}
                            , index=self.X.index)
    

    def get_individuals_contributions_axes(self) -> pd.DataFrame:
        """Retourne un dataframe avec les contributions des individus aux composantes principales

        Returns:
            pd.DataFrame: index = labels, CTR_X= contributions des individus aux CP
        """        
        return pd.DataFrame({str("CTR_" + str(factor+1)): self.ctr[:, factor] for factor in range(self.ctr.shape[1])},
                             index=self.X.index)

    def get_variable_correlation(self):
        """Retourne un dataframe avec les corrélations entre les variables et les composantes principales

        Returns:
            pd.DataFrame: index = labels, CTR_X= corrélations entre les variables et les composantes principales
        """
        return pd.DataFrame({str("COR_VAR_" + str(k+1)): self.corr_var[:, k] for k in range(self.corr_var.shape[1])}, index=self.X.columns)
    

    def get_cos2_variable(self) -> pd.DataFrame:
        """Retourne un dataframe avec les corrélations entre les variables et les composantes principales

        Returns:
            pd.DataFrame: index = labels, CTR_X= corrélations entre les variables et les composantes principales
        """
        dict = {"COS2_VAR_" + str(k): self.cos2_var[:, k]
                for k in range(self.acp.n_components_)}
        return pd.DataFrame({"COS2_VAR_" + str(k): self.cos2_var[:, k] for k in range(self.acp.n_components_)},
                             index=self.X.columns)
    

    def get_variables_contributions_to_axes(self) -> pd.DataFrame:
        """Retourne un dataframe avec les contributions des variables aux composantes principales

        Returns:
            pd.DataFrame: index = labels, CTR_VAR_X= contributions des variables aux composantes principales
        """
        return pd.DataFrame({"CTR_VAR_" + str(k): self.ctr_var[:, k] for k in range(self.ctr_var.shape[1])},
                            index=self.X.columns)    
    


    # -------------------------------------------------PLOTS----------------------------------------------
    # Plots for visualize and intreprate results of pca 

    def scree_plot(self) -> plt.plot:
        """Affiche le screeplot (eboulis des valeurs propres)

        Returns:
            plt.plot: Eboulis des valeurs propres (screeplot)
        """
        plt.plot(np.arange(1, self.p+1), self.eigval)
        plt.title("Eboulis des valeurs propres")
        plt.ylabel("Valeurs propres")
        plt.xlabel("Composantes principales n°")
        plt.show()


    def explained_variance_vs_factors_plot(self) -> plt.plot:
        """Heatmap avec la variance expliquée par les facteurs (composantes principales)

        Returns:
            plt.plot: Variance expliquée cumulée par les facteurs
        """
        # cumul de variance expliquée
        plt.plot(np.arange(1, self.p+1),
                 np.cumsum(self.acp.explained_variance_ratio_))
        plt.title(f"Variance expliquée vs {self.p} facteurs")
        plt.ylabel("Ratio de variance expliquée cumulé")
        plt.xlabel("Numéro du facteur")
        plt.show()
        
    
    def individuals_plot(self, plan_x: int = 1, plan_y: int = 2) -> plt.plot:
        """    Tracer la représentation des individus en indiquant les axes factoriels désirés

        Args:
            plan_x (int): Premier facteur
            plan_y (int): Second facteur

        Returns:
            plt.plot: Nuages des individus sur les plans factoriels sélectionné
        """
        c = self.X_transform
        # Calcul des limites du graphique
        xlim = np.abs([c[:, plan_x-1].min(), c[:, plan_x-1].max()]).max()+1
        ylim = np.abs([c[:, plan_y-1].min(), c[:, plan_y-1].max()]).max()+1

        # positionnement des individus dans le premier plan
        fig, axes = plt.subplots(figsize=(12, 12))
        axes.set_xlim(-xlim, xlim)  # même limites en abscisse
        axes.set_ylim(-ylim, ylim)  # et en ordonnée
        # remplacement des étiquettes des observations
        for i in range(self.n):
            plt.annotate(self.X.index[i], (c[i, plan_x-1], c[i, plan_y-1]))

        # ajouter les axes
        plt.plot([-xlim, xlim], [0, 0], color='silver',
                 linestyle='-', linewidth=1)
        plt.plot([0, 0], [-ylim, ylim], color='silver',
                 linestyle='-', linewidth=1)
        plt.xlabel(f"F{plan_x} : {self.acp.explained_variance_ratio_[plan_x-1]:.2%}")
        plt.ylabel(f"F{plan_y} : {self.acp.explained_variance_ratio_[plan_y-1]:.2%}")
        plt.title(f"Représentation des individus dans le plan F{plan_x} & F{plan_y}")
        # affichage
        plt.show()

    def correlation_circle_plot(self, plan_x=1, plan_y=2) -> plt.plot:
        """Construit le cercle des corrélations des variables

        Args:
            plan_x (int, optional): Première composantes principales (facteur) à afficher. Defaults to 1.
            plan_y (int, optional): Première composantes principales (facteur) à afficher. Defaults to 2.

        Returns:
            plt.plot: Cercle des corrélations des variables
        """

        # cercle des corrélations F1/F2
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        # affichage des étiquettes (noms des variables)
        for j in range(self.p):
            plt.annotate(
                self.X.columns[j], (self.corr_var[j, plan_x-1], self.corr_var[j, plan_y-1]))

        # ajouter les axes
        plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
        cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
        axes.add_artist(cercle)
        # Tracer les flèches à partir de l'origine (0,0) vers les points
        for x, y in zip(self.corr_var[:, plan_x-1].tolist(), self.corr_var[:, plan_y-1].tolist()):
            plt.quiver(0, 0, x, y, angles='xy', scale_units='xy',
                       scale=1, color='grey', width=0.005, alpha=0.4)

        plt.title(f"Cercle des corrélations - plan F{plan_x}/F{plan_y}")
        plt.show()

    def correlation_variable_factor_plot(self) -> plt.plot:
        """Construit une heatmap avec les corrélations entre variables et composantes principales

        Returns:
            plt.plot: Corrélations entre variables et composantes principales
        """
        d = pd.DataFrame({str("F" + str(factor+1)): self.corr_var[:, factor] for factor in range(self.corr_var.shape[1])}
                            , index=self.X.columns)
        xticklabels = [f"F{factor+1}\n({self.acp.explained_variance_ratio_[factor]: .2%})" for factor in range(self.corr_var.shape[1])]
        plt.figure(figsize=(12,10))
        sns.heatmap(d, annot=True, vmin=-1, vmax=1, cmap=sns.diverging_palette(20, 250, as_cmap=True), xticklabels=xticklabels)
        plt.title("Corrélations entres variables et facteurs")
        plt.show()