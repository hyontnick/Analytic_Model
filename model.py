# importation des bibliotheque
import os

# import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# def delete_img():
# 	# Répertoire contenant les images
# 	repertoire = '/home/hyont-nick/DATA_ANALYST/Soutenance/data_im/'

# 	# Parcours des fichiers du répertoire
# 	for fichier in os.listdir(repertoire):
#     		chemin_fichier = os.path.join(repertoire, fichier)
#     		if os.path.isfile(chemin_fichier) and fichier.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#         		os.remove(chemin_fichier)
#         		print(f"Le fichier {fichier} a été supprimé avec succès.")

# MODEL COMMERCIALE :

# 1. Quels sont les facteurs clés qui influencent les ventes d'un produit ou d'un service ?


def commercial1():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial1.csv"):
        print("Le fichier commercial1.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial1.csv"
        )
    else:
        print("Le fichier commercial1.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial11.csv"
        ):
            print("Le fichier commercial11.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial11.csv")
            print("Le fichier commercial11.csv a été supprimé.")
        else:
            print("Le fichier commercial11.csv n'existe pas.")

        data = {
            "facteurs": [
                "Prix",
                "Qualité",
                "Promotion",
                "Réputation",
                "Expérience",
                "Concurrence",
                "Tendances",
                "Publicité",
                "Emballage",
            ],
            "influences": np.random.randint(1, 100, size=9),
        }
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial11.csv",
            index=False,
        )

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="influences", y="facteurs")
    plt.xlabel("Influence")
    plt.ylabel("Facteurs clés")
    plt.title("Influences des facteurs clés sur les ventes")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial1.png")
    # plt.show()


# 2. Comment puis-je segmenter la clientèle en fonction de leurs caractéristiques et comportements d'achat ?


def commercial2():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial2.csv"):
        print("Le fichier commercial2.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial2.csv"
        )
    else:
        print("Le fichier commercial2.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial22.csv"
        ):
            print("Le fichier commercial22.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial22.csv")
            print("Le fichier commercial22.csv a été supprimé.")
        else:
            print("Le fichier commercial22.csv n'existe pas.")

        # Créer un jeu de données fictif
        data = {
            "Age": np.random.randint(17, 70, size=1000),
            "Revenu_annuel": np.random.randint(100000, 10000000, size=1000),
            "Depenses_mensuelles": np.random.randint(10000, 100000, size=1000),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial22.csv",
            index=False,
        )

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)

        # Appliquer la méthode de clustering k-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_features)
        labels = kmeans.labels_

        # Ajouter les étiquettes de cluster au DataFrame
        df["Cluster"] = labels
        plt.figure(figsize=(10, 5))

        # Visualiser la segmentation par revenu annuel
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=df, x="Age", y="Revenu_annuel", hue="Cluster", palette="Set1", alpha=0.5
    )
    plt.xlabel("Age")
    plt.ylabel("Revenu_annuel")
    plt.title("Segmentation de la clientèle par Revenu annuel")
    plt.legend()

    # Visualiser la segmentation par dépenses mensuelles
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=df,
        x="Age",
        y="Depenses_mensuelles",
        hue="Cluster",
        palette="Set1",
        alpha=0.5,
    )
    plt.xlabel("Age")
    plt.ylabel("Depenses_mensuelles")
    plt.title("Segmentation de la clientèle par Depenses mensuelles")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial2.png")
    # plt.show()


# 3. Quelles sont les méthodes d'analyse prédictive pour estimer la demande future d'un produit ou d'un service ?


def commercial3():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial3.csv"):
        print("Le fichier commercial3.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial3.csv"
        )
    else:
        print("Le fichier commercial2.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial33.csv"
        ):
            print("Le fichier commerciali1.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial33.csv")
            print("Le fichier commercial33.csv a été supprimé.")
        else:
            print("Le fichier commercial33.csv n'existe pas.")

        # Créer un jeu de données fictif
        data = {
            "Mois": [
                "Jan",
                "Fév",
                "Mar",
                "Avr",
                "Mai",
                "Juin",
                "Jui",
                "Aou",
                "Sep",
                "Oct",
            ],
            "Demande": np.random.randint(1000, 9000, size=10),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial33.csv",
            index=False,
        )

        # Convertir les mois en valeurs numériques
        month_to_number = {
            "Jan": 1,
            "Fév": 2,
            "Mar": 3,
            "Avr": 4,
            "Mai": 5,
            "Juin": 6,
            "Jui": 7,
            "Aou": 8,
            "Sep": 9,
            "Oct": 10,
        }
        df["Mois_num"] = df["Mois"].map(month_to_number)

        # Diviser les données en variables d'entrée (X) et variable cible (y)
        X = df[["Mois_num"]]
        y = df["Demande"]

        # Créer et entraîner le modèle de régression linéaire
        model = LinearRegression()
        model.fit(X, y)

        # Prédire la demande future
        mois_futurs = np.arange(11, 16)  # Mois 11 à 14 pour la prédiction
        demande_predite = model.predict(mois_futurs.reshape(-1, 1))

    plt.figure(figsize=(10, 5))

    # Visualiser les données réelles
    sns.lineplot(
        data=df, x="Mois_num", y="Demande", color="blue", label="Données réelles"
    )

    # Visualiser la prédiction
    sns.lineplot(
        x=mois_futurs,
        y=demande_predite,
        color="red",
        linestyle="-.",
        label="Prédiction",
    )

    plt.xlabel("Mois")
    plt.ylabel("Demande")
    plt.title("Estimation de la demande future")
    plt.legend()
    plt.xticks(
        list(month_to_number.values()), list(month_to_number.keys())
    )  # Remplacer les valeurs numériques par les mois correspondants sur l'axe des x
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial3.png")
    # plt.show()


# 4. Comment puis-je identifier les opportunités de croissance du chiffre d'affaires en analysant les données des ventes ?


def commercial4():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial4.csv"):
        print("Le fichier commercial4.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial4.csv"
        )
    else:
        print("Le fichier commercial4.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial44.csv"
        ):
            print("Le fichier commercial44.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial44.csv")
            print("Le fichier commercial44.csv a été supprimé.")
        else:
            print("Le fichier commercial44.csv n'existe pas.")

        # Créer un jeu de données fictif
        data = {
            "Mois": [
                "Jan",
                "Fév",
                "Mar",
                "Avr",
                "Mai",
                "Juin",
                "Juil",
                "Août",
                "Sep",
                "Oct",
                "Nov",
                "Déc",
            ],
            "Ventes": np.random.randint(100000, 1000000, 12),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial44.csv",
            index=False,
        )

    plt.figure(figsize=(10, 7))

    # Visualiser les ventes mensuelles
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x="Mois", y="Ventes", marker="o")
    plt.xlabel("Mois")
    plt.ylabel("Ventes")
    plt.title("Ventes mensuelles")
    plt.xticks(rotation=45)
    plt.grid(True)

    # Calculer la croissance mensuelle
    df["Croissance"] = df["Ventes"].pct_change() * 100

    # Visualiser la croissance mensuelle
    plt.subplot(2, 1, 2)
    sns.barplot(data=df, x="Mois", y="Croissance")
    plt.xlabel("Mois")
    plt.ylabel("Croissance (%)")
    plt.title("Croissance mensuelle des ventes")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial4.png")
    # plt.show()

    # Identifier les opportunités de croissance
    opportunites = df[df["Croissance"] > 0]
    print("Opportunités de croissance :")
    print(opportunites)


# 5. Quels sont les modèles de prévision les plus appropriés pour estimer les ventes à court et à long terme ?


def commercial5():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial5.csv"):
        print("Le fichier commercial5.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial5.csv"
        )
    else:
        print("Le fichier commercial5.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial55.csv"
        ):
            print("Le fichier commercial55.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial55.csv")
            print("Le fichier commercial55.csv a été supprimé.")
        else:
            print("Le fichier commercial55.csv n'existe pas.")
        # Créer un jeu de données fictif
        data = {
            "Mois": pd.date_range(start="01-01-2022", periods=24, freq="M"),
            "Ventes": np.random.randint(100000, 1000000, 24)
            # [100, 120, 130, 140, 160, 180, 200, 220, 230, 240, 260, 280,
            # 300, 320, 330, 350, 370, 390, 410, 420, 440, 460, 480, 500]
        }

        df = pd.DataFrame(data)
        df = df.set_index("Mois")
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial55.csv",
            index=False,
        )
        # Diviser les données en ensemble d'entraînement et ensemble de test
        train_data = df.iloc[
            :-6
        ]  # Utiliser les 18 premiers mois comme ensemble d'entraînement
        test_data = df.iloc[-6:]  # Utiliser les 6 derniers mois comme ensemble de test

        # Créer et entraîner un modèle de régression linéaire
        regression_model = LinearRegression()
        regression_model.fit(
            np.arange(len(train_data)).reshape(-1, 1), train_data["Ventes"]
        )

        # Prédire les ventes à court terme avec le modèle de régression linéaire
        short_term_predictions = regression_model.predict(
            np.arange(len(df)).reshape(-1, 1)
        )

        # Créer et entraîner un modèle ARIMA pour les ventes à long terme
        arima_model = ARIMA(train_data["Ventes"], order=(1, 1, 1))
        arima_model_fit = arima_model.fit()

        # Prédire les ventes à long terme avec le modèle ARIMA
        long_term_predictions = arima_model_fit.predict(start=len(df), end=len(df) + 5)

    plt.figure(figsize=(10, 5))

    sns.lineplot(data=df, x=df.index, y="Ventes", marker="o", label="Ventes réelles")
    sns.lineplot(
        data=df,
        x=df.index,
        y=short_term_predictions,
        linestyle="--",
        label="Prédictions à court terme",
    )
    sns.lineplot(
        data=long_term_predictions,
        x=long_term_predictions.index,
        y=long_term_predictions.values,
        linestyle="--",
        label="Prédictions à long terme",
    )

    plt.xlabel("Mois")
    plt.ylabel("Ventes")
    plt.title("Prévisions des ventes")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial5.png")
    # plt.show()


# 6. Comment puis-je utiliser les données des clients pour optimiser les stratégies de fidélisation et de rétention ?


def commercial6():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial6.csv"):
        print("Le fichier commercial6.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali6.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial6.csv"
        )
    else:
        print("Le fichier commercial6.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial66.csv"
        ):
            print("Le fichier commercial66.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial66.csv")
            print("Le fichier commercial66.csv a été supprimé.")
        else:
            print("Le fichier commercial66.csv n'existe pas.")
        # Créer un jeu de données fictif des clients
        data = {
            "ClientID": np.arange(1, 101),
            "Anciennete": np.random.randint(1, 12, size=100),
            "Satisfaction": np.random.randint(2, 9, size=100),
            "Sexe": np.random.choice(["F", "M"], size=100),
            "Age": np.random.randint(18, 65, size=100),
            "Achats": np.random.normal(50000, 10000, size=100),
            "Historique_Achats": np.random.randint(0, 10, size=100),
            "Churn": np.random.choice(["Non", "Oui"], size=100),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial66.csv",
            index=False,
        )

    plt.figure(figsize=(10, 10))

    # Visualiser les achats des clients
    plt.subplot(3, 3, 1)
    sns.barplot(x="Anciennete", y="Achats", data=df)
    plt.xlabel("Ancienneté (années)")
    plt.ylabel("Achats")
    plt.title("Relation entre l'ancienneté et les achats")

    # Distribution de l'âge des clients
    plt.subplot(3, 3, 2)
    sns.histplot(data=df, x="Age", bins=15, kde=True, edgecolor="black")
    plt.xlabel("Âge")
    plt.ylabel("Nombre de clients")
    plt.title("Distribution de l'âge des clients")

    # Visualiser l'historique des achats des clients
    plt.subplot(3, 3, 3)
    sns.barplot(x="Historique_Achats", y="Satisfaction", data=df)
    plt.xlabel("Historique des achats")
    plt.ylabel("Satisfaction")
    plt.title("Relation entre l'historique des achats \n et la satisfaction")

    # Visualiser la répartition des clients par sexe
    plt.subplot(3, 3, 4)
    sns.countplot(x="Sexe", data=df)
    plt.xlabel("Sexe")
    plt.ylabel("Nombre de clients")
    plt.title("Répartition des clients par sexe")

    # Visualiser le taux de churn (attrition)
    plt.subplot(3, 3, 5)
    sns.countplot(x="Churn", data=df)
    plt.xlabel("Churn")
    plt.ylabel("Nombre de clients")
    plt.title("Taux de churn")

    # Historique d'achat des clients
    plt.subplot(3, 3, 6)
    sns.countplot(x="Historique_Achats", data=df)
    plt.xlabel("Historique des achats")
    plt.ylabel("Nombre de clients")
    plt.title("Répartition de l'historique d'achat")

    # Distribution des achats des clients
    plt.subplot(3, 3, 7)
    sns.boxplot(x=df["Achats"])
    plt.xlabel("Achats")
    plt.title("Distribution des achats")
    plt.xticks(rotation=45)

    # Visualiser l'ancienneté des clients
    plt.subplot(3, 3, 8)
    sns.boxplot(x=df["Anciennete"])
    plt.xlabel("Ancienneté (années)")
    plt.title("Ancienneté des clients")

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial6.png")
    # plt.show()


# 7. Quels sont les indicateurs de performance clés utilisés pour évaluer l'efficacité d'une campagne marketing ou d'une stratégie commerciale ?


def commercial7():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial7.csv"):
        print("Le fichier commercial7.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali7.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial7.csv"
        )
    else:
        print("Le fichier commercial7.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial77.csv"
        ):
            print("Le fichier commercial77.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial77.csv")
            print("Le fichier commercial77.csv a été supprimé.")
        else:
            print("Le fichier commercial77.csv n'existe pas.")
        # Exemple de jeu de données fictif
        data = {
            "Mois": [
                "Jan",
                "Fév",
                "Mars",
                "Avr",
                "Mai",
                "Juin",
                "Juil",
                "Aou",
                "Sept",
                "Oct",
                "Nov",
                "Dec",
            ],
            "Taux de conversion": np.random.randint(1, 100, size=12),
            "Taux d'engagement": np.random.randint(0, 100, size=12),
            "CAC": np.random.randint(1, 100, size=12),
            "Taux_churn": np.random.randint(1, 100, size=12),
            "Taux de rebond": np.random.randint(1, 100, size=12),
            "Taux_clics": np.random.randint(1, 100, size=12),
            "Taux d'ouverture": np.random.randint(1, 100, size=12),
            "Taux_retention": np.random.randint(1, 100, size=12),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial77.csv",
            index=False,
        )
        # premiere visualisation

    df_sum = df.drop("Mois", axis=1).sum()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    sns.barplot(x=df_sum.index, y=df_sum.values)
    plt.xlabel("KPI")
    plt.ylabel("Pourcentage")
    plt.title("Indicateurs de performance")
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    sns.lineplot(
        x="Mois",
        y="Taux de conversion",
        marker="o",
        label="Taux \n de conversion",
        data=df,
    )
    sns.lineplot(
        x="Mois",
        y="Taux d'engagement",
        marker="o",
        label="Taux \n d'engagement",
        data=df,
    )
    sns.lineplot(x="Mois", y="CAC", marker="o", label="CAC", data=df)
    sns.lineplot(x="Mois", y="Taux_churn", marker="o", label="Taux \n churn", data=df)
    sns.lineplot(
        x="Mois", y="Taux de rebond", marker="o", label="Taux \n de rebond", data=df
    )
    sns.lineplot(x="Mois", y="Taux_clics", marker="o", label="Taux \n clics", data=df)
    sns.lineplot(
        x="Mois", y="Taux d'ouverture", marker="o", label="Taux \n d'ouverture", data=df
    )
    sns.lineplot(
        x="Mois", y="Taux_retention", marker="o", label="Taux \n retention", data=df
    )
    plt.xlabel("Mois")
    plt.ylabel("Pourcentage")
    plt.title("Taux de conversion")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial7.png")
    # plt.show()


# 8. Comment puis-je analyser les données de prix et de concurrence pour optimiser la stratégie de tarification d'un produit ou d'un service ?


def commercial8():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial8.csv"):
        print("Le fichier commercial8.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali8.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial8.csv"
        )
    else:
        print("Le fichier commercial8.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial88.csv"
        ):
            print("Le fichier commercial88.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial88.csv")
            print("Le fichier commercial88.csv a été supprimé.")
        else:
            print("Le fichier commercial88.csv n'existe pas.")
        # Creation des donnee fictive
        data = {
            "produit_prix": np.random.randint(1000, 10000, size=100),
            "concurrent_prix": np.random.randint(1000, 10000, size=100),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial88.csv",
            index=False,
        )

    plt.figure(figsize=(10, 4))

    # Tracé de la distribution des prix de la concurrence avec Seaborn
    plt.subplot(1, 3, 1)
    sns.histplot(df["concurrent_prix"], bins=10)
    plt.xlabel("Concurrence")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Prix de la Concurrence")

    # Tracé de la distribution des prix du produit avec Seaborn
    plt.subplot(1, 3, 2)
    sns.histplot(df["produit_prix"], bins=10)
    plt.xlabel("Prix")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Prix")

    # Comparaison des prix avec les concurrents avec Seaborn
    plt.subplot(1, 3, 3)
    labels = ["Produit", "Concurrent"]
    prix = [df["produit_prix"].mean(), df["concurrent_prix"].mean()]
    sns.barplot(x=labels, y=prix)
    plt.ylabel("Prix moyen")
    plt.title("Comparaison des prix avec les concurrents")

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial8.png")
    # plt.show()


# 9. Quelles sont les meilleures pratiques pour l'analyse des données de vente en ligne, telles que le suivi des conversions ou l'analyse du parcours client ?


def commercial9():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial9.csv"):
        print("Le fichier commercial9.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali9.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial9.csv"
        )
    else:
        print("Le fichier commercial9.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial99.csv"
        ):
            print("Le fichier commercial99.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial99.csv")
            print("Le fichier commercial99.csv a été supprimé.")
        else:
            print("Le fichier commercial99.csv n'existe pas.")
        # Creation des donnee fictive
        data = {
            "mois": [
                "Jan",
                "Fév",
                "Mars",
                "Avr",
                "Mai",
                "Juin",
                "Juil",
                "Aou",
                "Sept",
                "Oct",
                "Nov",
                "Dec",
            ],
            "conversions": np.random.randint(100, 900, size=12),
            "revenus": np.random.randint(100000, 10000000, size=12),
            "visiteurs": np.random.randint(1000, 9000, size=12),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial99.csv",
            index=False,
        )

    plt.figure(figsize=(10, 6))

    # Graphique en barres pour les conversions mensuelles avec Seaborn
    plt.subplot(2, 2, 1)
    sns.barplot(x="mois", y="conversions", data=df)
    plt.title("Conversions mensuelles")
    plt.xlabel("Mois")
    plt.ylabel("Nombre de conversions")

    # Graphique linéaire pour les revenus mensuels avec Seaborn
    plt.subplot(2, 2, 2)
    sns.lineplot(x="mois", y="revenus", data=df, marker="o")
    plt.title("Revenus mensuels")
    plt.xlabel("Mois")
    plt.ylabel("Revenus")

    # Graphique en barres pour le nombre de visiteurs mensuels avec Seaborn
    plt.subplot(2, 2, 3)
    sns.barplot(x="mois", y="visiteurs", data=df)
    plt.title("Nombre de visiteurs mensuels")
    plt.xlabel("Mois")
    plt.ylabel("Nombre de visiteurs")

    # Diagramme de dispersion pour les revenus en fonction des conversions avec Seaborn
    plt.subplot(2, 2, 4)
    sns.scatterplot(x="conversions", y="revenus", data=df, alpha=0.5)
    plt.title("Revenus en fonction des conversions")
    plt.xlabel("Conversions")
    plt.ylabel("Revenus")

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial9.png")
    # plt.show()


# 10. Comment puis-je utiliser l'apprentissage automatique pour détecter les schémas de comportement des clients et personnaliser les offres et les recommandations ?


def commercial10():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial10.csv"):
        print("Le fichier commercial10.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/commerciali10.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial10.csv"
        )
    else:
        print("Le fichier commercial10.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial1010.csv"
        ):
            print("Le fichier commercial1010.csv existe.")
            os.remove(
                "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial1010.csv"
            )
            print("Le fichier commercial1010.csv a été supprimé.")
        else:
            print("Le fichier commercial1010.csv n'existe pas.")
        # Creation des donnee fictive
        data = {
            "cluster1": np.random.normal(50, 5, (100, 2)),
            "cluster2": np.random.normal(30, 5, (80, 2)),
            "cluster3": np.random.normal(70, 10, (120, 2)),
        }

        # Concaténation des clusters
        df = pd.DataFrame(
            np.concatenate(
                [data["cluster1"], data["cluster2"], data["cluster3"]], axis=0
            ),
            columns=["Achats mensuels", "Montant des achats"],
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/commercial1010.csv",
            index=False,
        )
        # Clustering K-means
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(df)

        # Ajout des étiquettes de cluster à la DataFrame
        df["Cluster"] = kmeans.labels_

        # Définition des noms de cluster
        cluster_names = ["Achats réguliers", "Achats occasionnels", "Achats impulsifs"]

        # Affichage des points de données pour chaque cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Achats mensuels",
        y="Montant des achats",
        hue="Cluster",
        data=df,
        palette="colorblind",
        alpha=0.5,
    )

    # Affichage des centres de cluster
    sns.scatterplot(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        marker="x",
        color="green",
        s=100,
        label="Centres de cluster",
    )

    plt.title("Clustering K-means")
    plt.xlabel("Achats mensuels")
    plt.ylabel("Montant des achats")
    plt.legend(title="Clusters", labels=cluster_names)
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/commercial10.png")
    # plt.show()


# MODEL ECONOMIQUE :


# 1. Quelles sont les tendances économiques actuelles au niveau national ou mondial ?
def economique1():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1.csv"):
        print("Le fichier economique1.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1.csv"
        )
    else:
        print("Le fichier economique1.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique11.csv"
        ):
            print("Le fichier economique11.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique11.csv")
            print("Le fichier economique11.csv a été supprimé.")
        else:
            print("Le fichier economique11.csv n'existe pas.")

        # Générer un jeu de données fictif
        data = {
            "Année": np.arange(2016, 2026),
            "Croissance_du_PIB_national": np.random.uniform(2.0, 3.9, 10),
            "Croissance_du_PIB_mondial": np.random.uniform(6.0, 10.9, 10),
            "Taux_de_chômage_national": np.random.uniform(4.0, 10.9, 10),
            "Taux_de_chômage_mondial": np.random.uniform(8.0, 20.9, 10),
            "Taux_d'inflation": np.random.uniform(1.0, 2.9, 10),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique11.csv",
            index=False,
        )
    # Utiliser Seaborn pour une meilleure apparence
    sns.set(style="whitegrid")

    # Créer une figure et des sous-graphiques
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 6))

    # Graphique des tendances économiques nationales
    sns.lineplot(
        data=df,
        x="Année",
        y="Croissance_du_PIB_national",
        ax=ax1,
        marker="o",
        color="b",
        label="Croissance du PIB",
    )
    sns.lineplot(
        data=df,
        x="Année",
        y="Taux_de_chômage_national",
        ax=ax1,
        marker="o",
        color="r",
        label="Taux de chômage",
    )

    ax1.set_xlabel("Année")
    ax1.set_ylabel("Pourcentage (%)")
    ax1.set_title("Tendances économiques nationales")
    ax1.legend()

    # Affichage des chiffres du taux de chômage et de la croissance du PIB
    for i in range(len(df)):
        ax1.annotate(
            f"{df['Taux_de_chômage_national'][i]:.2f}%",
            (df["Année"][i], df["Taux_de_chômage_national"][i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="r",
        )
        ax1.annotate(
            f"{df['Croissance_du_PIB_national'][i]:.2f}%",
            (df["Année"][i], df["Croissance_du_PIB_national"][i]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="center",
            color="b",
        )

    # Graphique des tendances économiques mondiales
    sns.lineplot(
        data=df,
        x="Année",
        y="Croissance_du_PIB_mondial",
        ax=ax3,
        marker="o",
        color="b",
        label="Croissance du PIB",
    )
    sns.lineplot(
        data=df,
        x="Année",
        y="Taux_de_chômage_mondial",
        ax=ax3,
        marker="o",
        color="r",
        label="Taux de chômage",
    )

    ax3.set_xlabel("Année")
    ax3.set_ylabel("Pourcentage (%)")
    ax3.set_title("Tendances économiques mondiales")
    ax3.legend()

    # Affichage des chiffres du taux de chômage
    for i in range(len(df)):
        ax3.annotate(
            f"{df['Taux_de_chômage_mondial'][i]:.2f}%",
            (df["Année"][i], df["Taux_de_chômage_mondial"][i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="r",
        )
        ax3.annotate(
            f"{df['Croissance_du_PIB_mondial'][i]:.2f}%",
            (df["Année"][i], df["Croissance_du_PIB_mondial"][i]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="center",
            color="b",
        )

    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique1.png")
    # plt.show()


# 2. Comment puis-je identifier les principaux indicateurs économiques qui influencent la croissance économique ?


def economique2():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique2.csv"):
        print("Le fichier economique2.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique2.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique2.csv"
        )
    else:
        print("Le fichier economique2.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique22.csv"
        ):
            print("Le fichier economique11.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique22.csv")
            print("Le fichier economique22.csv a été supprimé.")
        else:
            print("Le fichier economique22.csv n'existe pas.")
        # Générer un jeu de données fictif
        data = {
            "PIB": np.random.uniform(1.0, 100.0, 12),
            "Investissement": np.random.uniform(1.0, 100.0, 12),
            "Consommation": np.random.uniform(1.0, 100.0, 12),
            "Exportations": np.random.uniform(1.0, 100.0, 12),
            "Innovation \n technologique": np.random.uniform(1.0, 100.0, 12),
            "Politiques \n gouvernementales": np.random.uniform(1.0, 100.0, 12),
            "Inflation": np.random.uniform(1.0, 100.0, 12),
            "Taux de \n chômage": np.random.uniform(1.0, 100.0, 12),
        }

        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique22.csv",
            index=False,
        )

    plt.figure(figsize=(12, 10))
    # Calculate correlations
    correlation_matrix = df.corr()

    # Extract correlations of "PIB" column
    correlations = correlation_matrix["PIB"]

    # Create correlation matrix heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap="RdYlBu")
    plt.title(
        "Corrélation entre les indicateurs économiques et la croissance économique"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique2.png")
    # plt.show()


# 3. Quels modèles économétriques puis-je utiliser pour prévoir l'évolution d'une variable économique spécifique, telle que le PIB ou l'inflation


def economique3():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique3.csv"):
        print("Le fichier economique3.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique3.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique3.csv"
        )
    else:
        print("Le fichier economique3.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique33.csv"
        ):
            print("Le fichier economique33.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique33.csv")
            print("Le fichier economique33.csv a été supprimé.")
        else:
            print("Le fichier economique33.csv n'existe pas.")

        # Création d'un dataframe fictif avec des données d'inflation
        data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2010-01-01", periods=120, freq="M"),
                "Inflation": np.random.randn(120),
            }
        )
        # Conversion de la colonne 'Date' en index temporel
        data.set_index("Date", inplace=True)
        data.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique33.csv",
            index=False,
        )

    plt.figure(figsize=(12, 5))
    # Ajustement du modèle SARIMA
    model = sm.tsa.SARIMAX(
        data["Inflation"], order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)
    )
    results = model.fit()

    # Prévision de l'inflation pour de nouvelles périodes
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2024-12-01")
    forecast = results.get_forecast(steps=(end_date.year - start_date.year + 1) * 12)
    forecasted_inflation = forecast.predicted_mean

    # Intervalles de confiance pour les prévisions
    confidence_interval = forecast.conf_int()

    # Visualisation de l'inflation réelle, prévue et des intervalles de confiance
    plt.plot(data.index, data["Inflation"], label="Inflation réelle", marker="o")
    plt.plot(forecasted_inflation.index, forecasted_inflation, label="Inflation prévue")
    plt.fill_between(
        confidence_interval.index,
        confidence_interval.iloc[:, 0],
        confidence_interval.iloc[:, 1],
        alpha=0.2,
        label="Intervalles de confiance",
    )
    plt.title("Prévision de l'inflation")
    plt.xlabel("Date")
    plt.ylabel("Inflation")
    plt.legend()
    sns.set(style="whitegrid")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique3.png")
    # plt.show()


# 4. Comment puis-je analyser les données du marché du travail pour évaluer la situation de l'emploi et détecter les tendances émergentes ?


def economique4():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique4.csv"):
        print("Le fichier economique4.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique4.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique4.csv"
        )
    else:
        print("Le fichier economique4.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique44.csv"
        ):
            print("Le fichier economique44.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique44.csv")
            print("Le fichier economique33.csv a été supprimé.")
        else:
            print("Le fichier economique44.csv n'existe pas.")
        # Generate fictitious employment data
        data = {
            "Année": np.arange(2010, 2022),
            "Emploi": np.random.randint(1000, 9000, 12),
            "Chômage": np.random.randint(1000, 90000, 12),
        }

        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique44.csv",
            index=False,
        )

        df["main-d'œuvre"] = df["Emploi"] + df["Chômage"]

    plt.figure(figsize=(10, 6))
    # Plot employment, unemployment, and labor force over time
    plt.subplot(2, 1, 1)
    plt.plot(df["Année"], df["Emploi"], label="Emploi", marker="o")
    plt.plot(df["Année"], df["Chômage"], label="Chômage", marker="<")
    plt.plot(df["Année"], df["main-d'œuvre"], label="main-d'œuvre", marker="x")
    plt.xlabel("Année")
    plt.ylabel("Nombre d'individus ")
    plt.title("Emploi, Chômage et Force de travail au fil du temps")
    plt.legend()
    plt.grid(True)

    # Calculer et tracer le taux de chômage
    df["Taux_de_chômage"] = df["Chômage"] / df["main-d'œuvre"] * 100
    plt.subplot(2, 1, 2)
    plt.plot(df["Année"], df["Taux_de_chômage"], marker="o")
    plt.xlabel("Année")
    plt.ylabel("Taux de chômage (%)")
    plt.title("Taux de chômage au fil du temps")
    plt.grid(True)
    plt.tight_layout()
    sns.set(style="whitegrid")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique4.png")
    # plt.show()


# 5. Quelles sont les méthodes d'analyse économique pour évaluer l'impact des politiques publiques ou des changements réglementaires sur l'économie ?


def economique5():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique5.csv"):
        print("Le fichier economique5.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique5.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique5.csv"
        )
    else:
        print("Le fichier economique5.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique55.csv"
        ):
            print("Le fichier economique55.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique55.csv")
            print("Le fichier economique55.csv a été supprimé.")
        else:
            print("Le fichier economique55.csv n'existe pas.")
        # Charger les données fictives avant et après la politique publique
        data = {
            "Année_avant": np.arange(2009, 2016),
            "PIB_avant": np.random.randint(1000000, 1000000000, 7),
            "Investissement_avant": np.random.randint(100000, 100000000, 7),
            "Année_apres": np.arange(2016, 2023),
            "PIB_apres": np.random.randint(1000000, 1000000000, 7),
            "Investissement_apres": np.random.randint(100000, 100000000, 7),
        }

        # Convertir les données en DataFrames
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique55.csv",
            index=False,
        )

    plt.figure(figsize=(10, 6))
    # Plotter le PIB avant et après la politique publique
    plt.subplot(2, 1, 1)
    sns.lineplot(
        x="Année_avant",
        y="PIB_avant",
        data=df,
        label="Avant la politique publique",
        marker="o",
    )
    sns.lineplot(
        x="Année_apres",
        y="PIB_apres",
        data=df,
        label="Après la politique publique",
        marker="o",
    )
    plt.xlabel("Année")
    plt.ylabel("PIB")
    plt.title("Impact de la politique publique sur le PIB")
    plt.legend()

    # Plotter l'investissement avant et après la politique publique
    plt.subplot(2, 1, 2)
    sns.lineplot(
        x="Année_avant",
        y="Investissement_avant",
        data=df,
        label="Avant la politique publique",
        marker="o",
    )
    sns.lineplot(
        x="Année_apres",
        y="Investissement_apres",
        data=df,
        label="Après la politique publique",
        marker="o",
    )
    plt.xlabel("Année")
    plt.ylabel("Investissement")
    plt.title("Impact de la politique publique sur l'investissement")
    plt.legend()
    plt.tight_layout()
    sns.set(style="whitegrid")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique5.png")
    # plt.show()


# 6 Comment puis-je utiliser les données de consommation pour analyser les habitudes de consommation et identifier les segments de marché ?


def economique6():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique6.csv"):
        print("Le fichier economique6.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique6.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique6.csv"
        )
    else:
        print("Le fichier economique6.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique66.csv"
        ):
            print("Le fichier economique66.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique66.csv")
            print("Le fichier economique66.csv a été supprimé.")
        else:
            print("Le fichier economique66.csv n'existe pas.")

        # Charger les données fictives sur les habitudes de consommation
        data = {
            "Dépenses \n Alimentaires": np.random.randint(1000, 90000, 10000),
            "Dépenses \n Mode": np.random.randint(1000, 90000, 10000),
            "Dépenses \n Électroniques": np.random.randint(1000, 9000, 10000),
            "Depenses \n Abonement": np.random.randint(4000, 30000, 10000),
            "Depenses \n Voyages": np.random.randint(10000, 100000, 10000),
        }

        # Convertir les données en DataFrame
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique66.csv",
            index=False,
        )

    # Sélectionner les variables à utiliser pour la segmentation
    features = [
        "Dépenses \n Alimentaires",
        "Dépenses \n Électroniques",
        "Dépenses \n Mode",
        "Depenses \n Abonement",
        "Depenses \n Voyages",
    ]
    X = df[features]

    # Standardiser les variables pour avoir une même échelle
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Appliquer l'algorithme de clustering (K-Means)
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # Ajouter les étiquettes de cluster au DataFrame
    df["Segment"] = labels

    # Compter le nombre d'éléments dans chaque cluster
    segment_counts = df["Segment"].value_counts().sort_index()

    # Obtenir les centres de chaque cluster
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Générer une palette de couleurs correspondant au nombre de segments
    colors = sns.color_palette("husl", n_colors=len(segment_counts))

    # Afficher le graphique en barres avec les couleurs correspondantes et la légende
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    segment_counts.plot(kind="bar", color=colors)

    plt.xlabel("Segments")
    plt.ylabel("Nombre d'individus")
    plt.title("Distribution des segments de consommation")

    # Ajouter les valeurs des segments en légende avec les noms des champs correspondants
    for i, count in enumerate(segment_counts):
        plt.text(i, count, str(count), ha="center", va="bottom")
        plt.text(i, count - 200, features[i], ha="center", va="top", rotation=45)

    # Visualiser les segments de marché avec le nombre d'éléments en légende
    plt.subplot(1, 2, 2)
    for i in range(len(colors)):
        cluster_data = df[df["Segment"] == i]
        plt.scatter(
            cluster_data["Dépenses \n Alimentaires"],
            cluster_data["Dépenses \n Mode"],
            c=[colors[i]],
            label="Segment {} ({})".format(i + 1, segment_counts[i]),
        )

    # Représenter les centres de chaque cluster
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        color="black",
        marker="x",
        label="Centres de cluster",
        alpha=0.1,
    )

    plt.xlabel("Dépenses Alimentaires")
    plt.ylabel("Dépenses Mode")
    plt.title("Segments de marché basés sur les habitudes de consommation")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique6.png")
    # plt.show()


# 7. Quels sont les modèles de prévision les plus appropriés pour estimer les ventes ou les revenus d'une entreprise ?


def economique7():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique7.csv"):
        print("Le fichier economique7.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique5.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique7.csv"
        )
    else:
        print("Le fichier economique5.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique77.csv"
        ):
            print("Le fichier economique77.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique77.csv")
            print("Le fichier economique77.csv a été supprimé.")
        else:
            print("Le fichier economique77.csv n'existe pas.")

        # Création d'un dataframe fictif avec des données de ventes
        data = {
            "Temps": pd.date_range("2021-01-01", periods=100, freq="M"),
            "Ventes": np.random.randint(50, 100, size=100),
        }
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique77.csv",
            index=False,
        )
        # Conversion de la colonne 'Temps' en index temporel
        df.set_index("Temps", inplace=True)

        # Ajustement du modèle SARIMA
        model = sm.tsa.SARIMAX(
            df["Ventes"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)
        )
        results = model.fit()

        # Prévision des ventes pour de nouvelles périodes
        start_date = pd.to_datetime("2023-01-01")
        end_date = pd.to_datetime("2023-12-01")
        forecast = results.get_forecast(
            steps=(end_date.year - start_date.year + 1) * 12
        )
        forecasted_sales = forecast.predicted_mean

        # Intervalles de confiance pour les prévisions
        confidence_interval = forecast.conf_int()

    # Application du style Seaborn
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")

    # Visualisation des ventes réelles, prévues et intervalles de confiance
    plt.figure(figsize=(10, 7))
    plt.plot(df.index, data["Ventes"], label="Ventes réelles")
    plt.plot(forecasted_sales.index, forecasted_sales, label="Ventes prévues")
    plt.fill_between(
        confidence_interval.index,
        confidence_interval.iloc[:, 0],
        confidence_interval.iloc[:, 1],
        alpha=0.2,
        label="Intervalles de confiance",
    )
    plt.title("Prévision des ventes")
    plt.xlabel("Temps")
    plt.ylabel("Ventes")
    plt.legend()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique7.png")
    # plt.show()


# 8. Comment puis-je analyser les données financières pour évaluer la performance et la stabilité financière d'une entreprise ?


def economique8():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique8.csv"):
        print("Le fichier economique8.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique8.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique8.csv"
        )
    else:
        print("Le fichier economique8.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique88.csv"
        ):
            print("Le fichier economique88.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique88.csv")
            print("Le fichier economique88.csv a été supprimé.")
        else:
            print("Le fichier economique88.csv n'existe pas.")
        # Génération de données fictives pour l'exemple
        # np.random.seed(0)
        years = range(2010, 2024)
        revenue = np.random.randint(1000, 10000, len(years))
        expenses = np.random.randint(500, 5000, len(years))
        profit = revenue - expenses
        assets = np.random.randint(10000, 100000, len(years))
        liabilities = np.random.randint(5000, 50000, len(years))
        equity = assets - liabilities

        # Création du DataFrame
        df = pd.DataFrame(
            {
                "Année": years,
                "Revenus": revenue,
                "Dépenses": expenses,
                "Profit": profit,
                "Actifs": assets,
                "Passifs": liabilities,
                "Capitaux propres": equity,
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique88.csv",
            index=False,
        )
    plt.figure(figsize=(10, 10))
    # Visualisation des revenus, des dépenses et des profits
    plt.subplot(3, 1, 1)
    sns.lineplot(data=df, x="Année", y="Revenus", label="Revenus")
    sns.lineplot(data=df, x="Année", y="Dépenses", label="Dépenses")
    sns.lineplot(data=df, x="Année", y="Profit", label="Profit")
    plt.xlabel("Année")
    plt.ylabel("Montant")
    plt.title("Revenus, Dépenses et Profit")
    plt.legend()

    # Visualisation des actifs, des passifs et des capitaux propres
    plt.subplot(3, 1, 2)
    sns.lineplot(data=df, x="Année", y="Actifs", label="Actifs")
    sns.lineplot(data=df, x="Année", y="Passifs", label="Passifs")
    sns.lineplot(data=df, x="Année", y="Capitaux propres", label="Capitaux propres")
    plt.xlabel("Année")
    plt.ylabel("Montant")
    plt.title("Actifs, Passifs et Capitaux propres")
    plt.legend()

    # Calcul du ratio de rentabilité (profit/revenu)
    df["Ratio de rentabilité"] = df["Profit"] / df["Revenus"]

    # Visualisation du ratio de rentabilité
    plt.subplot(3, 1, 3)
    sns.lineplot(data=df, x="Année", y="Ratio de rentabilité")
    plt.xlabel("Année")
    plt.ylabel("Ratio de rentabilité")
    plt.title("Évolution du ratio de rentabilité")
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique8.png")
    # plt.show()


# 9. Quelles sont les meilleures pratiques pour analyser les données du commerce international, telles que les échanges commerciaux et les flux de capitaux ?


def economique9():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique9.csv"):
        print("Le fichier economique9.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique9.csv')
        df_combined = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique9.csv"
        )
    else:
        print("Le fichier economique9.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique99.csv"
        ):
            print("Le fichier economique99.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique99.csv")
            print("Le fichier economique99.csv a été supprimé.")
        else:
            print("Le fichier economique99.csv n'existe pas.")
        # Génération de données fictives pour l'exemple
        continents = ["Afrique", "Amérique", "Asie", "Europe", "Océanie"]
        years = range(2012, 2023)
        exports = np.random.randint(1000, 10000, size=(len(continents), len(years)))
        imports = np.random.randint(500, 5000, size=(len(continents), len(years)))
        capital_flows = np.random.randint(
            -1000, 1000, size=(len(continents), len(years))
        )

        # Création des DataFrames pour chaque catégorie
        df_exports = pd.DataFrame(exports, columns=years, index=continents)
        df_imports = pd.DataFrame(imports, columns=years, index=continents)
        df_balance = pd.DataFrame(exports - imports, columns=years, index=continents)
        df_capital_flows = pd.DataFrame(capital_flows, columns=years, index=continents)

        # Concaténation des DataFrames en une seule DataFrame
        df_combined = pd.concat(
            [df_exports, df_imports, df_balance, df_capital_flows],
            keys=["Exports", "Imports", "Balance", "Capital Flows"],
            axis=1,
        )

        df_combined.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique99.csv",
            index=False,
        )
    # Application du style Seaborn
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")

    # Création de la figure et des sous-graphiques
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    # Graphique des exportations par pays
    df_combined["Exports"].T.plot(kind="bar", stacked=True, ax=axes[0, 0])
    axes[0, 0].set_xlabel("Année")
    axes[0, 0].set_ylabel("Montant des exportations")
    axes[0, 0].set_title("Exportations par continents")

    # Graphique des importations par pays
    df_combined["Imports"].T.plot(kind="bar", stacked=True, ax=axes[0, 1])
    axes[0, 1].set_xlabel("Année")
    axes[0, 1].set_ylabel("Montant des importations")
    axes[0, 1].set_title("Importations par continents")

    # Graphique du solde commercial par pays
    df_combined["Balance"].sum().plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_xlabel("Année")
    axes[1, 0].set_ylabel("Solde commercial")
    axes[1, 0].set_title("Solde commercial par continents")

    # Graphique des flux de capitaux par continents
    df_combined["Capital Flows"].T.plot(kind="bar", stacked=True, ax=axes[1, 1])
    axes[1, 1].set_xlabel("Année")
    axes[1, 1].set_ylabel("Flux de capitaux")
    axes[1, 1].set_title("Flux de capitaux par continents")

    # Ajustement des espacements entre les sous-graphiques
    plt.tight_layout()

    # Affichage de la figure
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique9.png")
    # plt.show()


# 10. Comment puis-je utiliser l'apprentissage automatique pour détecter les anomalies économiques ou les schémas de fraude ?


def economique10():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique10.csv"):
        print("Le fichier economique10.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique10.csv')
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique10.csv"
        )
    else:
        print("Le fichier economique10.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1010.csv"
        ):
            print("Le fichier economique1010.csv existe.")
            os.remove(
                "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1010.csv"
            )
            print("Le fichier economique1010.csv a été supprimé.")
        else:
            print("Le fichier economique1010.csv n'existe pas.")
        # Génération de données fictives pour l'exemple

        dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
        revenue = np.random.normal(loc=10000, scale=2000, size=len(dates))
        expenses = np.random.normal(loc=8000, scale=1500, size=len(dates))
        profit = revenue - expenses

        # Création du DataFrame
        df = pd.DataFrame(
            {"Date": dates, "Revenue": revenue, "Expenses": expenses, "Profit": profit}
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/economique1010.csv",
            index=False,
        )

        # Standardisation des données
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[["Revenue", "Expenses", "Profit"]])

        # Détection des anomalies avec l'Isolation Forest
        isolation_forest = IsolationForest(contamination=0.05)
        isolation_forest.fit(df_scaled)
        outliers = isolation_forest.predict(df_scaled)

    # Visualisation des données avec les anomalies en rouge
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Date"], df["Profit"], c=outliers, cmap="coolwarm", alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Profit")
    plt.title("Détection d'anomalies dans les profits")
    plt.colorbar()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/economique10.png")
    # plt.show()


# MODEL EDUCATIF:


# 1. Quelles sont les tendances actuelles en matière de résultats scolaires dans une région ou un système éducatif donné ?


def educatif1():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1.csv"):
        print("Le fichier educatif1.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1.csv")
    else:
        print("Le fichier educatif1.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif11.csv"
        ):
            print("Le fichier educatif1.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif11.csv")
            print("Le fichier educatif11.csv a été supprimé.")
        else:
            print("Le fichier educatif11.csv n'existe pas.")
        # Données fictives
        data = {
            "annees": np.arange(2010, 2023),
            "resultats": np.random.uniform(70.0, 99.0, 13),
        }

        # creation de la dataframe
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif11.csv", index=False
        )

    # creation de l'image
    plt.figure(figsize=(10, 7))

    # Création du graphique
    # sns.barplot(x=df['annees'], y=df['resultats'], color='b')
    sns.barplot(x="annees", y="resultats", data=df)
    # Ajout de labels et de titre
    plt.xlabel("Année")
    plt.ylabel("Résultats (%)")
    plt.title("Tendances des résultats scolaires")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif1.png")
    # Affichage du graphique
    # plt.show()


# 2. Comment puis-je identifier les facteurs qui influencent la réussite scolaire des élèves ?


def educatif2():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif2.csv"):
        print("Le fichier educatif2.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif2.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif2.csv")
    else:
        print("Le fichier educatif2.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif22.csv"
        ):
            print("Le fichier educatif2.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif22.csv")
            print("Le fichier educatif22.csv a été supprimé.")
        else:
            print("Le fichier educatif22.csv n'existe pas.")

        # Jeu de données fictif
        donnees = {
            "Note": np.random.randint(60, 99, 5),
            "Temps \n d'étude (heures)": np.random.randint(2, 9, 5),
            "Participation \n en classe": np.random.randint(5, 9, 5),
            "Nombre \n d'absences": np.random.randint(1, 9, 5),
            "Niveau \n socio-économique": np.random.randint(1, 5, 5),
        }

        # Création du DataFrame à partir du jeu de données
        df = pd.DataFrame(donnees)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif22.csv", index=False
        )

        # Calcul de la matrice de corrélation
        matrice_correlation = df.corr()

    # Création de la heatmap de la matrice de corrélation
    sns.heatmap(matrice_correlation, annot=True, cmap="coolwarm")
    # Affichage de la heatmap
    plt.title("Matrice de corrélation des facteurs de réussite scolaire")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif2.png")
    # plt.show()


# 3. Quels sont les modèles prédictifs les plus appropriés pour estimer les performances futures des élèves ?


def educatif3():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif3.csv"):
        print("Le fichier educatif3.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif3.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif3.csv")
    else:
        print("Le fichier educatif3.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif33.csv"
        ):
            print("Le fichier educatif3.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif33.csv")
            print("Le fichier educatif33.csv a été supprimé.")
        else:
            print("Le fichier educatif33.csv n'existe pas.")
        # Génération de données fictives
        study_hours = np.random.randint(1, 10, size=100)  # Heures d'étude
        test_scores = (
            50 + 5 * study_hours + np.random.normal(0, 5, size=100)
        )  # Performances aux tests (relation linéaire avec les heures d'étude)

        df = pd.DataFrame({"study_hours": study_hours, "test_scores": test_scores})
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif33.csv", index=False
        )
        # Création du modèle de régression linéaire
        model = LinearRegression()
        model.fit(df["study_hours"].values.reshape(-1, 1), df["test_scores"])

        # Prédictions des performances futures
        future_hours = np.linspace(1, 10, num=100).reshape(
            -1, 1
        )  # Heures d'étude futures
        future_scores = model.predict(future_hours)

    # Visualisation des données et des prédictions
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x="study_hours", y="test_scores", data=df, alpha=0.5, label="Données réelles"
    )
    sns.lineplot(
        x=future_hours.flatten(), y=future_scores, color="red", label="Prédictions"
    )
    plt.xlabel("Heures d'étude")
    plt.ylabel("Performances aux tests")
    plt.title("Estimation des performances futures des élèves")
    plt.legend()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif3.png")
    # plt.show()


# 4. Comment puis-je utiliser les données sur les effectifs des classes pour optimiser la répartition des ressources éducatives ?


def educatif4():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif4.csv"):
        print("Le fichier educatif4.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif4.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif4.csv")
    else:
        print("Le fichier educatif4.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif44.csv"
        ):
            print("Le fichier educatif4.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif44.csv")
            print("Le fichier educatif44.csv a été supprimé.")
        else:
            print("Le fichier educatif44.csv n'existe pas.")
        # Données de ressources éducatives
        ressources = [
            "Professeurs",
            "Matériel \n pédagogique",
            "Espaces \n disponibles",
            "Budget",
            "Bibliothèque",
            "Laboratoires",
        ]
        quantites = np.random.uniform(1.0, 100.0, size=6)

        # Création du DataFrame pour les ressources éducatives
        df = pd.DataFrame({"Ressources": ressources, "Quantités": quantites})
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif44.csv", index=False
        )

    # Affichage des ressources éducatives
    plt.figure(figsize=(6, 6))
    sns.barplot(x="Ressources", y="Quantités", data=df)
    plt.xlabel("Ressources")
    plt.ylabel("Quantités")
    plt.title("Ressources éducatives")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif4.png")
    # plt.show()


# 5. Quelles sont les méthodes d'analyse statistique pour évaluer l'efficacité d'une intervention éducative ou d'un programme scolaire ?


def educatif5():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif5.csv"):
        print("Le fichier educatif5.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif5.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif5.csv")
    else:
        print("Le fichier educatif5.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif55.csv"
        ):
            print("Le fichier educatif5.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif55.csv")
            print("Le fichier educatif55.csv a été supprimé.")
        else:
            print("Le fichier educatif55.csv n'existe pas.")
        # Données de performances avant et après l'intervention
        performances_avant = np.array([70, 75, 80, 85, 90])
        performances_apres = np.array([80, 85, 90, 95, 100])

        # Calcul des moyennes des colonnes
        moyenne_avant = performances_avant.mean()
        moyenne_apres = performances_apres.mean()

        # Création du DataFrame avec les données de performances et les moyennes
        df = pd.DataFrame(
            {
                "Performances": ["Avant", "Après"],
                "Moyenne": [moyenne_avant, moyenne_apres],
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif55.csv", index=False
        )

    # Utilisation de Seaborn pour la visualisation du graphique à barres
    plt.figure(figsize=(6, 6))
    sns.barplot(x="Performances", y="Moyenne", data=df)
    # Personnalisation des axes et du titre
    plt.xlabel("Intervention")
    plt.ylabel("Performances moyennes")
    plt.title("Comparaison des performances avant et après l'intervention")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif5.png")
    # Affichage du graphique
    # plt.show()


# 6. Comment puis-je utiliser les données démographiques pour analyser les disparités d'accès à l'éducation ou de performances entre différents groupes d'élèves ?


def educatif6():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif6.csv"):
        print("Le fichier educatif6.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif6.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif6.csv")
    else:
        print("Le fichier educatif6.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif66.csv"
        ):
            print("Le fichier educatif6.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif66.csv")
            print("Le fichier educatif66.csv a été supprimé.")
        else:
            print("Le fichier educatif66.csv n'existe pas.")
        # Création d'un jeu de données fictif
        df = pd.DataFrame(
            {
                "Gender": np.random.choice(["Male", "Female"], 10),
                "Ethnicity": [
                    "Centre",
                    "Ouest",
                    "Nord",
                    "Littoral",
                    "Est",
                    "Nord-Ouest",
                    "Sud-Ouest",
                    "Adamaoua",
                    "Extrême-Nord",
                    "Sud",
                ],
                "Access to Education": np.random.choice(["Yes", "No"], 10),
                "Performance": np.random.randint(60, 99, 10),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif66.csv", index=False
        )

    plt.figure(figsize=(12, 8))

    # Visualisation des disparités d'accès à l'éducation par genre
    ax1 = plt.subplot(2, 2, 1)
    sns.barplot(
        x="Gender",
        y="Proportion",
        hue="Access to Education",
        data=df.groupby(["Gender", "Access to Education"])
        .size()
        .reset_index()
        .rename(columns={0: "Proportion"}),
        ax=ax1,
    )
    plt.xlabel("Genre")
    plt.ylabel("Proportion d'élèves")
    plt.title("Disparités d'accès à l'éducation par genre")

    # Visualisation des disparités d'accès à l'éducation par Région
    ax2 = plt.subplot(2, 2, 2)
    sns.barplot(
        x="Ethnicity",
        y="Proportion",
        hue="Access to Education",
        data=df.groupby(["Ethnicity", "Access to Education"])
        .size()
        .reset_index()
        .rename(columns={0: "Proportion"}),
        ax=ax2,
    )
    plt.xlabel("Région")
    plt.ylabel("Proportion d'élèves")
    plt.title("Disparités d'accès à l'éducation par Région")
    plt.xticks(rotation=45)

    # Visualisation des disparités de performances par genre
    ax3 = plt.subplot(2, 2, 3)
    sns.barplot(x="Gender", y="Performance", data=df, ax=ax3)
    plt.xlabel("Genre")
    plt.ylabel("Performance moyenne")
    plt.title("Disparités de performances par genre")

    # Visualisation des disparités de performances par Région
    ax4 = plt.subplot(2, 2, 4)
    sns.barplot(x="Ethnicity", y="Performance", data=df, ax=ax4)
    plt.xlabel("Région")
    plt.ylabel("Performance moyenne")
    plt.title("Disparités de performances par Région")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif6.png")
    # plt.show()


# 7. Quels sont les indicateurs clés de performance utilisés pour évaluer la qualité d'une école ou d'un système éducatif ?


def educatif7():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif7.csv"):
        print("Le fichier educatif7.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif7.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif7.csv")
    else:
        print("Le fichier educatif7.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif77.csv"
        ):
            print("Le fichier educatif7.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif77.csv")
            print("Le fichier educatif77.csv a été supprimé.")
        else:
            print("Le fichier educatif77.csv n'existe pas.")
        # Création du jeu de données fictif
        df = pd.DataFrame(
            {
                "School": ["Vogt", "Liberman", "Jean_tabi", "Bepanda", "Nkolbisson"],
                "Success Rate": np.random.uniform(0.70, 0.99, 5),
                "Dropout Rate": np.random.uniform(0.01, 0.09, 5),
                "Student Satisfaction": np.random.uniform(4.1, 4.9, 5),
                "Teacher Satisfaction": np.random.uniform(4.1, 4.9, 5),
                "Student Competence": np.random.randint(70, 99, 5),
                "Student Progress": np.random.uniform(0.60, 0.99, 5),
                "Participation Rate": np.random.uniform(0.7, 0.99, 5),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif77.csv", index=False
        )
        # Convertir les colonnes numériques en type float
        numeric_columns = [
            "Success Rate",
            "Dropout Rate",
            "Student Satisfaction",
            "Teacher Satisfaction",
            "Student Competence",
            "Student Progress",
            "Participation Rate",
        ]
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Calcul de la somme des indicateurs par école
        df["Total"] = df[numeric_columns].sum(axis=1)

    # Création de la figure et des sous-graphiques
    fig, axs = plt.subplots(2, 4, figsize=(16, 10))

    # Taux de réussite
    sns.barplot(x="School", y="Success Rate", data=df, ax=axs[0, 0])
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].set_xlabel("École")
    axs[0, 0].set_ylabel("Taux de réussite")
    axs[0, 0].set_title("Taux de réussite par école")
    axs[0, 0].set_xticklabels(df["School"], rotation=45)

    # Taux d'abandon
    sns.barplot(x="School", y="Dropout Rate", data=df, ax=axs[0, 1])
    axs[0, 1].set_ylim(0, 0.1)
    axs[0, 1].set_xlabel("École")
    axs[0, 1].set_ylabel("Taux d'abandon")
    axs[0, 1].set_title("Taux d'abandon par école")
    axs[0, 1].set_xticklabels(df["School"], rotation=45)

    # Satisfaction des élèves
    sns.barplot(x="School", y="Student Satisfaction", data=df, ax=axs[0, 2])
    axs[0, 2].set_ylim(0, 5)
    axs[0, 2].set_xlabel("École")
    axs[0, 2].set_ylabel("Satisfaction des élèves")
    axs[0, 2].set_title("Satisfaction des élèves par école")
    axs[0, 2].set_xticklabels(df["School"], rotation=45)

    # Satisfaction des enseignants
    sns.barplot(x="School", y="Teacher Satisfaction", data=df, ax=axs[0, 3])
    axs[0, 3].set_ylim(0, 5)
    axs[0, 3].set_xlabel("École")
    axs[0, 3].set_ylabel("Satisfaction des enseignants")
    axs[0, 3].set_title("Satisfaction des enseignants par école")
    axs[0, 3].set_xticklabels(df["School"], rotation=45)

    # Niveau de compétence des élèves
    sns.barplot(x="School", y="Student Competence", data=df, ax=axs[1, 0])
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].set_xlabel("École")
    axs[1, 0].set_ylabel("Niveau de compétence des élèves")
    axs[1, 0].set_title("Niveau de compétence des élèves par école")
    axs[1, 0].set_xticklabels(df["School"], rotation=45)

    # Taux de progression des élèves
    sns.barplot(x="School", y="Student Progress", data=df, ax=axs[1, 1])
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].set_xlabel("École")
    axs[1, 1].set_ylabel("Taux de progression des élèves")
    axs[1, 1].set_title("Taux de progression des élèves par école")
    axs[1, 1].set_xticklabels(df["School"], rotation=45)

    # Somme des indicateurs par école
    sns.barplot(x="School", y="Total", data=df, ax=axs[1, 2], color="red")
    axs[1, 2].set_xlabel("École")
    axs[1, 2].set_ylabel("Somme des indicateurs")
    axs[1, 2].set_title("Somme des indicateurs par école")
    axs[1, 2].set_xticklabels(df["School"], rotation=45)

    # Supprimer les sous-graphiques vides
    fig.delaxes(axs[1, 3])

    # Ajuster la disposition des sous-graphiques
    fig.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif7.png")
    # Afficher le graphique
    # plt.show()


# 8. Comment puis-je analyser les données des évaluations nationales ou internationales pour comparer les performances des élèves à l'échelle nationale ou internationale ?


def educatif8():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif8.csv"):
        print("Le fichier educatif8.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif8.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif8.csv")
    else:
        print("Le fichier educatif8.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif88.csv"
        ):
            print("Le fichier educatif8.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif88.csv")
            print("Le fichier educatif88.csv a été supprimé.")
        else:
            print("Le fichier educatif88.csv n'existe pas.")
        # Création du jeu de données fictif
        df = pd.DataFrame(
            {
                "Country": np.random.choice(
                    [
                        "Cameroun",
                        "République \n centrafricaine",
                        "Tchad",
                        "Congo",
                        "Guinée \n équatoriale",
                        "RDC",
                        "Angola",
                    ],
                    35,
                ),
                "Score": np.random.randint(60, 99, 35),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif88.csv", index=False
        )

    # Création du graphique en boîte avec Seaborn
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Country", y="Score", data=df)
    plt.xlabel("Pays")
    plt.ylabel("Score")
    plt.title("Comparaison des performances des élèves par pays")

    # Rotation des étiquettes des pays pour une meilleure lisibilité
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif8.png")
    # Afficher le graphique
    # plt.show()


# 9. Quelles sont les meilleures pratiques pour l'utilisation des technologies éducatives basées sur les données ?


def educatif9():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif9.csv"):
        print("Le fichier educatif9.csv existe.")
        # os.remove('/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif9.csv')
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif9.csv")
    else:
        print("Le fichier educatif9.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif99.csv"
        ):
            print("Le fichier educatif9.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif99.csv")
            print("Le fichier educatif99.csv a été supprimé.")
        else:
            print("Le fichier educatif99.csv n'existe pas.")
        # Création d'un jeu de données fictif
        df = pd.DataFrame(
            {
                "Région": [
                    "Amérique du Nord",
                    "Europe",
                    "Asie",
                    "Afrique",
                    "Amérique du Sud",
                ],
                "Pourcentage": np.random.randint(40, 99, 5),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif99.csv", index=False
        )

    # Création du graphique en secteurs
    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        df["Pourcentage"], labels=df["Région"], autopct="%1.1f%%", startangle=90
    )

    # Amélioration du style avec Seaborn
    sns.set_palette("Set3")
    sns.set_context("notebook", font_scale=1.2)

    # Personnalisation des étiquettes en pourcentage
    for autotext in autotexts:
        autotext.set_color("white")

    plt.axis("equal")
    plt.title("Utilisation des technologies éducatives par région")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif9.png")
    # plt.show()


# 10. Comment puis-je utiliser l'apprentissage automatique pour détecter les schémas d'apprentissage des élèves et personnaliser les méthodes d'enseignement ?


def educatif10():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif10.csv"):
        print("Le fichier educatif10.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif10.csv")
    else:
        print("Le fichier educatif10.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1010.csv"
        ):
            print("Le fichier educatif1010.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1010.csv")
            print("Le fichier educatif1010.csv a été supprimé.")
        else:
            print("Le fichier educatif1010.csv n'existe pas.")

        # Création d'un jeu de données fictif pour les performances des élèves
        df = pd.DataFrame(
            {
                "Élève": [
                    "Momo",
                    "Tsague",
                    "Epassi",
                    "Dongmo",
                    "Meli",
                    "Eteme",
                    "Djamen",
                    "Mohamed",
                ],
                "Note Mathématiques": [85, 72, 78, 91, 88, 79, 92, 95],
                "Note Sciences": [78, 82, 75, 89, 84, 80, 93, 90],
                "Note Français": [82, 75, 80, 92, 87, 77, 91, 94],
                "Note Histoire": [76, 80, 73, 88, 82, 75, 90, 87],
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/educatif1010.csv",
            index=False,
        )

    # Sélection des colonnes de données pour l'apprentissage automatique
    X = df[["Note Mathématiques", "Note Sciences", "Note Français", "Note Histoire"]]

    # Réduction de dimension avec PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Application de l'algorithme de clustering K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    labels = kmeans.labels_

    # Ajout des labels au jeu de données
    df["Cluster"] = labels

    plt.figure(figsize=(6, 6))
    # Visualisation des schémas d'apprentissage en 2D avec les clusters
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], alpha=0.5, cmap="viridis")

    # Affichage des élèves par cluster
    for cluster in df["Cluster"].unique():
        cluster_data = df[df["Cluster"] == cluster]
        print(f"Cluster {cluster}:")
        for _, row in cluster_data.iterrows():
            print(row["Élève"])
        print("\n")

    # Ajout des annotations pour chaque point
    for i, txt in enumerate(df["Élève"]):
        plt.text(X_pca[i, 0], X_pca[i, 1], txt, ha="center", va="center")

    plt.xlabel("Composante Principale 1")
    plt.ylabel("Composante Principale 2")
    plt.title("Schémas d'apprentissage des élèves")
    plt.colorbar()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/educatif10.png")
    # plt.show()


# MODEL SANITAIRE


# 1. Quelles sont les tendances actuelles des maladies ou des conditions de santé dans une population donnée ?


def sanitaire1():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire1.csv"):
        print("Le fichier sanitaire1.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire1.csv")
    else:
        print("Le fichier sanitaire1.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire11.csv"
        ):
            print("Le fichier sanitaire11.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire11.csv")
            print("Le fichier sanitaire11.csv a été supprimé.")
        else:
            print("Le fichier sanitaire11.csv n'existe pas.")
        # Créer un jeu de données fictif
        data = {
            "Condition": [
                "Diabète",
                "Hypertension",
                "Obésité",
                "Asthme",
                "Cancer",
                "Typhoide",
                "Paludisme",
            ],
            "Nombre de cas": np.random.randint(1000, 9000, 7),
        }
        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire11.csv", index=False
        )
        # Créer le graphique à barres
    plt.figure(figsize=(7, 6))
    sns.barplot(x="Condition", y="Nombre de cas", data=df)
    plt.xlabel("Condition")
    plt.ylabel("Nombre de cas")
    plt.title("Tendances de santé dans la population")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire1.png")
    # Afficher le graphique
    # plt.show()


# 2. Comment puis-je identifier les facteurs de risque associés à une maladie spécifique ?


def sanitaire2():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire2.csv"):
        print("Le fichier sanitaire2.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire2.csv")
    else:
        print("Le fichier sanitaire2.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire22.csv"
        ):
            print("Le fichier sanitaire22.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire22.csv")
            print("Le fichier sanitaire22.csv a été supprimé.")
        else:
            print("Le fichier sanitaire22.csv n'existe pas.")
        # Créer un jeu de données fictif
        data = {
            "Age": np.random.randint(20, 60, 10),
            "IMC": np.random.randint(20, 40, 10),
            "Pression artérielle": np.random.randint(100, 150, 10),
            "Cholestérol": np.random.randint(100, 300, 10),
            "Fumeur": np.random.choice([0, 1], 10),
            "Diabète": np.random.choice([0, 1], 10),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire22.csv", index=False
        )
    # Créer un graphique en nuage de points pour les facteurs de risque
    plt.figure(figsize=(12, 12))
    sns.pairplot(
        df, vars=["Age", "IMC", "Pression artérielle", "Cholestérol"], hue="Diabète"
    )
    plt.suptitle("Relation entre les facteurs de risque et le diabète")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire2.png")
    # plt.show()


# 3. Quelles méthodes d'analyse statistique puis-je utiliser pour évaluer l'efficacité d'un traitement médical ?


def sanitaire3():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire3.csv"):
        print("Le fichier sanitaire3.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire3.csv")
    else:
        print("Le fichier sanitaire3.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire33.csv"
        ):
            print("Le fichier sanitaire33.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire33.csv")
            print("Le fichier sanitaire33.csv a été supprimé.")
        else:
            print("Le fichier sanitaire33.csv n'existe pas.")
        # Créer un jeu de données fictif
        data = {
            "Traitement": np.random.choice(
                [
                    "anaca A",
                    "anaca B",
                    "anaca C",
                    "anaca D",
                    "anaca E",
                    "anaca F",
                ],
                20,
            ),
            "Résultat": np.random.randint(4, 15, 20),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire33.csv", index=False
        )
    # Boxplot pour comparer les traitements
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Traitement", y="Résultat", data=df)
    plt.title("Comparaison des traitements")
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire3.png")
    # plt.show()


# 4. Comment puis-je utiliser les données épidémiologiques pour prédire la propagation d'une maladie ?


def sanitaire4():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire4.csv"):
        print("Le fichier sanitaire4.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire4.csv")
    else:
        print("Le fichier sanitaire4.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire44.csv"
        ):
            print("Le fichier sanitaire44.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire44.csv")
            print("Le fichier sanitaire4.csv a été supprimé.")
        else:
            print("Le fichier sanitaire44.csv n'existe pas.")
        # Créer un jeu de données fictif
        dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        cases = np.random.randint(0, 10000, len(dates))
        df = pd.DataFrame({"Date": dates, "Nombre de cas": cases})
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire44.csv", index=False
        )

        # Prévoir la propagation avec un modèle SARIMA
        model = SARIMAX(
            df["Nombre de cas"], order=(1, 0, 0), seasonal_order=(1, 1, 1, 7)
        )
        model_fit = model.fit()

        # Prédictions et intervalles de confiance
        pred = model_fit.get_prediction(start=len(df), end=len(df) + 30, dynamic=False)
        pred_mean = pred.predicted_mean
        conf_int = pred.conf_int()

    # Visualiser les prédictions avec intervalle de confiance
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Nombre de cas"], label="Données réelles")
    plt.plot(
        pd.date_range(start=df["Date"].iloc[-1], periods=31, freq="D"),
        pred_mean,
        label="Prédictions",
    )
    plt.fill_between(
        pd.date_range(start=df["Date"].iloc[-1], periods=31, freq="D"),
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3,
        color="gray",
        label="Intervalle de confiance",
    )
    plt.xlabel("Date")
    plt.ylabel("Nombre de cas")
    plt.title("Prévisions de la propagation de la maladie avec intervalle de confiance")
    plt.legend()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire4.png")
    # plt.show()


# 5. Quels sont les indicateurs clés de performance utilisés pour évaluer la qualité des soins de santé ?


def sanitaire5():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire5.csv"):
        print("Le fichier sanitaire5.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire5.csv")
    else:
        print("Le fichier sanitaire5.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire55.csv"
        ):
            print("Le fichier sanitaire55.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire55.csv")
            print("Le fichier sanitaire55.csv a été supprimé.")
        else:
            print("Le fichier sanitaire55.csv n'existe pas.")
        # Créer un jeu de données fictif
        data = {
            "Indicateur": [
                "Taux de \n mortalité",
                "Taux de \n survie",
                "Taux de \n complications",
                "Taux de \n réadmission",
                "Satisfaction des \n patients",
                "Taux d'infection \n nosocomiale",
                "Taux d'erreur \n médicale",
            ],
            "Score": np.random.uniform(0.0, 1.0, 7),
        }

        df = pd.DataFrame(data)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire55.csv", index=False
        )

    # Barplot pour visualiser les scores des indicateurs clés de performance
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Score", y="Indicateur", data=df)
    plt.xlabel("Score")
    plt.ylabel("Indicateur")
    plt.title("Scores des indicateurs clés de performance")
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire5.png")
    # plt.show()


# 6. Comment puis-je analyser les données de remboursement des soins de santé pour identifier les inefficiences ou les coûts élevés ?


def sanitaire6():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire6.csv"):
        print("Le fichier sanitaire6.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire6.csv")
    else:
        print("Le fichier sanitaire6.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire66.csv"
        ):
            print("Le fichier sanitaire66.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire66.csv")
            print("Le fichier sanitaire66.csv a été supprimé.")
        else:
            print("Le fichier sanitaire66.csv n'existe pas.")
        # Jeu de données fictif
        remboursements = {
            "types_de_soins": [
                "Consultation",
                "Chirurgie",
                "Radiologie",
                "Laboratoire",
                "Médicaments",
                "Soins infirmiers",
                "Thérapie physique",
                "Soins dentaires",
                "Soins oculaires",
                "Soins de maternité",
                "Soins psychiatriques",
                "Soins de réadaptation",
            ],
            "mois": [
                "Janvier",
                "Février",
                "Mars",
                "Avril",
                "Mai",
                "Juin",
                "Juillet",
                "Août",
                "Septembre",
                "Octobre",
                "Novembre",
                "Décembre",
            ],
            "montants": np.random.randint(20000, 90000, 12),
            "couts": np.random.randint(20000, 90000, 12),
        }

        df = pd.DataFrame(remboursements)
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire66.csv", index=False
        )

    plt.figure(figsize=(12, 7))

    plt.subplot(1, 2, 1)
    sns.barplot(x="mois", y="montants", data=df)
    plt.title("Remboursements mensuels des soins de santé")
    plt.xlabel("Mois")
    plt.ylabel("Montant (en euros)")
    plt.xticks(rotation=45)

    sns.set(style="whitegrid")
    plt.subplot(1, 2, 2)
    plt.pie(df["couts"], labels=df["types_de_soins"], autopct="%1.1f%%", startangle=90)
    plt.title("Répartition des coûts par type de soins")
    plt.axis("equal")

    # Affichage du graphique
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire6.png")
    # plt.show()


# 7. Quels sont les modèles de prédiction les plus appropriés pour estimer la probabilité de réadmission à l'hôpital pour un patient donné ?


def sanitaire7():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire7.csv"):
        print("Le fichier sanitaire7.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire7.csv")
    else:
        print("Le fichier sanitaire7.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire77.csv"
        ):
            print("Le fichier sanitaire77.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire77.csv")
            print("Le fichier sanitaire77.csv a été supprimé.")
        else:
            print("Le fichier sanitaire77.csv n'existe pas.")
        # Création d'un jeu de données fictif
        df = pd.DataFrame(
            {
                "Age": np.random.randint(18, 90, 1000),
                "Pression artérielle": np.random.randint(80, 180, 1000),
                "Cholestérol": np.random.randint(120, 300, 1000),
                "Diabète": np.random.choice([0, 1], size=1000),
                "Readmission": np.random.choice([0, 1], size=1000, p=[0.8, 0.2]),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire77.csv", index=False
        )

        # Séparer les caractéristiques (variables indépendantes) et la cible (variable dépendante)
        features = df.drop("Readmission", axis=1)
        target = df["Readmission"]

        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Créer le modèle de régression logistique
        model = LogisticRegression()

        # Entraîner le modèle sur l'ensemble d'entraînement
        model.fit(X_train, y_train)

        # Prédire les probabilités de réadmission pour l'ensemble de test
        probabilities = model.predict_proba(X_test)[:, 1]

        # Calculer l'AUC-ROC
        auc = roc_auc_score(y_test, probabilities)
        print("AUC-ROC:", auc)

        # Calculer les taux de faux positifs (FPR) et les taux de vrais positifs (TPR)
        fpr, tpr, _ = roc_curve(y_test, probabilities)

    sns.set(style="whitegrid")
    # Tracer la courbe ROC avec Matplotlib
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbe ROC")

    # Ajouter une légende
    plt.legend(["Courbe ROC (AUC = %0.2f)" % auc, "Aléatoire"], loc="lower right")

    # Comparer les moyennes des probabilités de réadmission et de non-réadmission
    comparison = pd.DataFrame({"Probabilité": probabilities, "Readmission": y_test})
    readmission_prob = comparison[comparison["Readmission"] == 1]["Probabilité"]
    non_readmission_prob = comparison[comparison["Readmission"] == 0]["Probabilité"]

    if readmission_prob.mean() > non_readmission_prob.mean():
        plt.text(
            0.5,
            0.2,
            "Probabilité de réadmission > Non-réadmission",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    elif readmission_prob.mean() < non_readmission_prob.mean():
        plt.text(
            0.5,
            0.2,
            "Probabilité de non-réadmission > Réadmission",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    else:
        plt.text(
            0.5,
            0.2,
            "Probabilités de réadmission et non-réadmission égales",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire7.png")
    # plt.show()


# 8. Comment puis-je utiliser l'apprentissage automatique pour détecter les schémas d'anomalies dans les données de santé ?


def sanitaire8():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire8.csv"):
        print("Le fichier sanitaire8.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire8.csv")
    else:
        print("Le fichier sanitaire8.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire88.csv"
        ):
            print("Le fichier sanitaire88.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire88.csv")
            print("Le fichier sanitaire88.csv a été supprimé.")
        else:
            print("Le fichier sanitaire88.csv n'existe pas.")
        # Création d'un jeu de données fictif
        df = pd.DataFrame(
            {
                "Pression artérielle": np.random.normal(120, 10, 1000),
                "Cholestérol": np.random.normal(200, 20, 1000),
                "Fréquence cardiaque": np.random.normal(70, 5, 1000),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire88.csv", index=False
        )

        # Ajouter quelques anomalies
        df.iloc[50, 0] = 200
        df.iloc[100, 1] = 300
        df.iloc[150, 2] = 90

        # Utiliser l'algorithme d'Isolation Forest pour détecter les anomalies
        model = IsolationForest(contamination=0.05)  # 5% de contamination
        model.fit(df)

        # Prédire les anomalies
        predictions = model.predict(df)
        anomalies = df[predictions == -1]

    # Visualiser les données et les anomalies détectées avec Seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Pression artérielle",
        y="Cholestérol",
        data=df,
        color="blue",
        alpha=0.3,
        label="Données",
    )
    sns.scatterplot(
        x="Pression artérielle",
        y="Cholestérol",
        data=anomalies,
        color="red",
        alpha=0.3,
        label="Anomalies",
    )
    plt.xlabel("Pression artérielle")
    plt.ylabel("Cholestérol")
    plt.title("Détection des anomalies dans les données de santé")
    plt.legend()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire8.png")
    # plt.show()


# 9. Comment puis-je analyser les données des essais cliniques pour évaluer l'efficacité et la sécurité d'un médicament ?


def sanitaire9():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire9.csv"):
        print("Le fichier sanitaire9.csv existe.")
        df = pd.read_csv("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire9.csv")
    else:
        print("Le fichier sanitaire9.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire99.csv"
        ):
            print("Le fichier sanitaire99.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire99.csv")
            print("Le fichier sanitaire99.csv a été supprimé.")
        else:
            print("Le fichier sanitaire99.csv n'existe pas.")
        # Création d'un jeu de données fictif pour les essais cliniques
        df = pd.DataFrame(
            {
                "Groupe": np.random.choice(["Placebo", "Traitement"], size=10000),
                "Résultat": np.random.choice(
                    ["Amélioration", "Pas d'amélioration", "Effets secondaires"],
                    size=10000,
                ),
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire99.csv", index=False
        )

        # Compter le nombre de participants dans chaque groupe
        group_counts = df["Groupe"].value_counts()

    # Visualiser le nombre de participants par groupe
    plt.figure(figsize=(12, 7))

    plt.subplot(1, 2, 1)
    sns.barplot(x=group_counts.index, y=group_counts.values)
    plt.xlabel("Groupe")
    plt.ylabel("Nombre de participants")
    plt.title("Répartition des participants par groupe")

    # Compter les résultats dans chaque groupe
    result_counts = df["Résultat"].value_counts()

    # Visualiser les résultats par groupe
    plt.subplot(1, 2, 2)
    sns.barplot(x=result_counts.index, y=result_counts.values)
    plt.xlabel("Résultat")
    plt.ylabel("Nombre de participants")
    plt.title("Répartition des résultats par groupe")
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire9.png")
    # plt.show()


# 10. Quelles sont les meilleures pratiques pour la gestion et la sécurité des données de santé sensibles ?


def sanitaire10():
    if os.path.isfile("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire10.csv"):
        print("Le fichier sanitaire10.csv existe.")
        df = pd.read_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire10.csv"
        )
    else:
        print("Le fichier sanitaire10.csv n'existe pas.")

        if os.path.isfile(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire1010.csv"
        ):
            print("Le fichier sanitaire1010.csv existe.")
            os.remove("/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire1010.csv")
            print("Le fichier sanitaire1010.csv a été supprimé.")
        else:
            print("Le fichier sanitaire1010.csv n'existe pas.")
        # Création d'un jeu de données fictif pour représenter la conformité aux réglementations de sécurité des données de santé
        df = pd.DataFrame(
            {
                "Règlement": ["RGPD", "HIPAA", "ISO 27001", "HITRUST", "PCI DSS"],
                "Conformité": [90, 85, 95, 80, 70],
            }
        )
        df.to_csv(
            "/home/hyont-nick/DATA_ANALYST/Soutenance/data/sanitaire1010.csv",
            index=False,
        )

    # Visualiser la conformité aux réglementations de sécurité des données de santé
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Règlement", y="Conformité", data=df)
    plt.ylim(0, 100)
    plt.xlabel("Règlement")
    plt.ylabel("Taux de conformité (%)")
    plt.title("Conformité aux réglementations de sécurité des données de santé")
    plt.xticks(
        rotation=45
    )  # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.tight_layout()
    plt.savefig("/home/hyont-nick/DATA_ANALYST/Soutenance/app/data_img/sanitaire10.png")
    # plt.show()

def visualisation():
    commercial1()
    commercial2()
    commercial3()
    commercial4()
    commercial5()
    commercial6()
    commercial7()
    commercial8()
    commercial9()
    commercial10()
    economique1()
    economique2()
    economique3()
    economique4()
    economique5()
    economique6()
    economique7()
    economique8()
    economique9()
    economique10()
    educatif1()
    educatif2()
    educatif3()
    educatif4()
    educatif5()
    educatif6()
    educatif7()
    educatif8()
    educatif9()
    educatif10()
    sanitaire1()
    sanitaire2()
    sanitaire3()
    sanitaire4()
    sanitaire5()
    sanitaire6()
    sanitaire7()
    sanitaire8()
    sanitaire9()
    sanitaire10()


visualisation()
