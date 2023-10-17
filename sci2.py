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

# Fonction principale de l'application Streamlit
def main():
    # Titre de la page
    st.title("Exploration de Data Science et Machine Learning")
    
    # Sélection du dataset
    option = st.selectbox('Quel dataset voulez-vous explorer?', ('Iris', 'Diabetes', 'Wine', 'Breast Cancer'))
    st.write(f"Vous avez sélectionné le dataset {option}")

    if option == 'Iris':
        # Chargement des données
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        
        # Statistiques descriptives
        st.subheader("Statistique Descriptive")
        st.write(df.describe())
        
        # EDA : Pairplot
        st.subheader("Analyse Exploratoire de Données (EDA) via Pairplot")
        sns.pairplot(df, hue="target")
        plt.savefig("pairplot.png")
        st.image("pairplot.png")
        
        # ANOVA
        st.subheader("Analyse de Variance (ANOVA)")
        fvalue, pvalue = stats.f_oneway(df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'], df['petal width (cm)'])
        st.write("F-value:", fvalue)
        st.write("P-value:", pvalue)
        
        # Boxplots
        st.subheader("Boxplots pour chaque caractéristique en fonction de la classe")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='target', y='sepal length (cm)', data=df)
        plt.savefig("boxplot.png")
        st.image("boxplot.png")
        
        # Encodage et calcul de la corrélation
        st.subheader("Encodage des variables catégorielles et corrélation")
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        st.write(df.corr())
        
        # Utilisation d'un modèle d'arbre de décision
        st.subheader("Utilisation d'un arbre de décision")
        X = df.drop('target', axis=1)
        y = df['target']
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        st.write('Score avec les données d\'entraînement:', clf.score(X, y))

        # Visualisation de l'arbre de décision
        st.subheader("Visualisation de l'arbre de décision")
        plt.figure(figsize=(20, 10))
        plot_tree(clf, filled=True, feature_names=iris['feature_names'], class_names=[str(i) for i in iris['target_names']])
        plt.savefig("decision_tree.png")
        st.image("decision_tree.png")
        
        # Utilisation de différents algorithmes de ML
        st.subheader("Différents types d'algorithmes de machine learning")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        st.write('Score de la régression logistique:', lr.score(X_test, y_test))
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        st.write('Score de la forêt aléatoire:', rf.score(X_test, y_test))
        svm = SVC()
        svm.fit(X_train, y_train)
        st.write('Score du SVM:', svm.score(X_test, y_test))

# Point d'entrée de l'application
if __name__ == '__main__':
    main()
