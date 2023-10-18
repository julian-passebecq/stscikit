# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Fonction principale de l'application Streamlit
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Statistiques Descriptives", "Analyse de Variance (ANOVA)", "Corrélation", "Modèles de Machine Learning"])

    if page == "Accueil":
        st.title("Exploration de Data Science et Machine Learning sur le dataset Iris")

    elif page == "Statistiques Descriptives":
        st.title("Statistiques Descriptives")
        # Chargement des données
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        st.write(df.describe())

    elif page == "Analyse de Variance (ANOVA)":
        st.title("Analyse de Variance (ANOVA)")
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        # Utilisation d'une heatmap
        sns.heatmap(df.corr(), annot=True)
        plt.savefig("heatmap.png")
        st.image("heatmap.png")
    
    elif page == "Corrélation":
        st.title("Corrélation entre les variables")
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        # Encodage
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        # Heatmap de corrélation
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.savefig("correlation.png")
        st.image("correlation.png")
        
    elif page == "Modèles de Machine Learning":
        st.title("Modèles de Machine Learning")
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        # Split des données
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Forêt aléatoire
        st.subheader("Forêt aléatoire")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        st.write('Score de la forêt aléatoire:', rf.score(X_test, y_test))
        # Heatmap de la matrice de confusion
        sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt='d')
        plt.savefig("confusion_matrix_rf.png")
        st.image("confusion_matrix_rf.png")

        # Régression logistique
        st.subheader("Régression logistique")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        st.write('Score de la régression logistique:', lr.score(X_test, y_test))

        # SVM
        st.subheader("Machine à vecteurs de support (SVM)")
        svm = SVC()
        svm.fit(X_train, y_train)
        st.write('Score du SVM:', svm.score(X_test, y_test))

if __name__ == '__main__':
    main()
