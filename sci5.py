# Importation des bibliothèques
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Fonction principale de l'application Streamlit
def main():
    st.title("Exploration de Data Science et Machine Learning")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisissez une page", ["Introduction", "Statistiques Descriptives", "Analyse Exploratoire", "ANOVA", "Modèles ML"])
    
    if page == "Introduction":
        st.subheader("Introduction")
        st.write("Dans cette application, nous explorerons le dataset Iris en utilisant différentes techniques de data science et de machine learning.")
        
    # Chargement des données
    iris = datasets.load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
    if page != "Introduction":
        st.write(f"Dataset : Iris")
        
    if page == "Statistiques Descriptives":
        st.subheader("Statistiques Descriptives")
        st.write(df.describe())
        st.subheader("Conclusion")
        st.write("La description statistique nous donne un aperçu des tendances centrales, de la dispersion et de la forme de la distribution.")

    if page == "Analyse Exploratoire":
        st.subheader("Analyse Exploratoire de Données (EDA)")
        sns.pairplot(df, hue="target")
        plt.savefig("pairplot.png")
        st.image("pairplot.png")
        st.subheader("Conclusion")
        st.write("Le pairplot permet d'identifier les relations entre différentes caractéristiques. Il peut également indiquer les groupes distincts ou les tendances.")

    if page == "ANOVA":
        st.subheader("Analyse de Variance (ANOVA)")
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.savefig("heatmap.png")
        st.image("heatmap.png")
        st.subheader("Conclusion")
        st.write("L'ANOVA et la heatmap de corrélation nous aident à comprendre l'importance de chaque caractéristique.")

    if page == "Modèles ML":
        st.subheader("Modèles de Machine Learning")
        # Division des données
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Forêt aléatoire
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        st.write("Score de la forêt aléatoire:", rf.score(X_test, y_test))
        st.write(confusion_matrix(y_test, y_pred_rf))
        st.write(classification_report(y_test, y_pred_rf))
        
        # Régression logistique
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        st.write("Score de la régression logistique:", lr.score(X_test, y_test))
        st.write(confusion_matrix(y_test, y_pred_lr))
        st.write(classification_report(y_test, y_pred_lr))

        st.subheader("Conclusion")
        st.write("Dans cette section, nous avons comparé plusieurs modèles de machine learning pour prédire la classe d'une fleur Iris. Cela permet de comprendre les avantages et les inconvénients de chaque modèle.")

# Point d'entrée de l'application
if __name__ == '__main__':
    main()
