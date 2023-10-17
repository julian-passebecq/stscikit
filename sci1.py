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

    # Page d'accueil
    st.title("Exploration de Data Science et Machine Learning")
    
    option = st.selectbox(
        'Quel dataset voulez-vous explorer?',
        ('Iris', 'Diabetes', 'Wine', 'Breast Cancer'))

    st.write(f"Vous avez sélectionné le dataset {option}")

    if option == 'Iris':
        # Part 1: Charger le dataset
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])

        # Part 2: Statistique Descriptive
        st.subheader("Statistique Descriptive")
        st.write(df.describe())

        # Part 3: EDA
        st.subheader("Analyse Exploratoire de Données (EDA)")
        sns.pairplot(df, hue="target")
        st.pyplot()

        # Part 4: ANOVA
        st.subheader("Analyse de Variance (ANOVA)")
        fvalue, pvalue = stats.f_oneway(df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'], df['petal width (cm)'])
        st.write("F-value:", fvalue)
        st.write("P-value:", pvalue)

        # Part 5: Boxplots groupés
        st.subheader("Boxplots Groupés")
        sns.boxplot(x='target', y='sepal length (cm)', data=df)
        st.pyplot()

        # Part 6: Encodage et corrélation
        st.subheader("Encodage des variables catégorielles et calcul de la corrélation")
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['target'])
        st.write(df.corr())

        # Part 7: Arbre de décision
        st.subheader("Utilisation d'un arbre de décision")
        X = df.drop('target', axis=1)
        y = df['target']
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        st.write('Score:', clf.score(X, y))

        # Part 8: Algorithmes de ML
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

if __name__ == '__main__':
    main()
