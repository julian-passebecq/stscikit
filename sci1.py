# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
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
    
    # Sélection du dataset à utiliser
    option = st.selectbox('Quel dataset voulez-vous explorer?', ('Iris', 'Diabetes', 'Wine', 'Breast Cancer'))
    st.write(f"Vous avez sélectionné le dataset {option}")

    if option == 'Iris':
        # Partie 1: Chargement du dataset
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
        
        # Partie 2: Statistique Descriptive
        st.subheader("Statistique Descriptive")
        st.write(df.describe())
        
        # Partie 3: EDA (Analyse Exploratoire de Données)
        st.subheader("Analyse Exploratoire de Données (EDA)")
        sns.pairplot(df, hue="target")
        plt.savefig("pairplot.png")
        st.image("pairplot.png")
        
        # Partie 4: ANOVA (Analyse de Variance)
        st.subheader("Analyse de Variance (ANOVA)")
        fvalue, pvalue = stats.f_oneway(df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'], df['petal width (cm)'])
        st.write("F-value:", fvalue)
        st.write("P-value:", pvalue)
        
        # Partie 5: Boxplots groupés
        st.subheader("Boxplots Groupés")
        sns.boxplot(x='target', y='sepal length (cm)', data=df)
        plt.savefig("boxplot.png")
        st.image("boxplot.png")
        
        # Partie 6: Encodage et corrélation
        st.subheader("Encodage des variables catégorielles et calcul de la corrélation")
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        st.write(df.corr())
        
        # Partie 7: Arbre de décision
        st.subheader("Utilisation d'un arbre de décision")
        X = df.drop('target', axis=1)
        y = df['target']
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        st.write('Score:', clf.score(X, y))
        
        # Partie 8: Algorithmes de ML
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
