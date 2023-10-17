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
        
        # ANOVA pour chaque caractéristique
        st.subheader("Analyse de Variance (ANOVA) pour chaque caractéristique")
        for feature in iris['feature_names']:
            st.write(f"ANOVA pour {feature}")
            groups = [df[df['target']==target][feature] for target in np.unique(df['target'])]
            fvalue, pvalue = stats.f_oneway(*groups)
            st.write("F-value:", fvalue)
            st.write("P-value:", pvalue)
            st.write("Si la valeur F est grande et la valeur p est petite (généralement moins de 0.05), cela signifie que les groupes sont statistiquement différents.")
        
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
        
        # Utilisation de la forêt aléatoire
        st.subheader("Utilisation de la forêt aléatoire")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        st.write('Score de la forêt aléatoire:', rf.score(X_test, y_test))
        st.write("La forêt aléatoire utilise plusieurs arbres de décision pour prendre une décision. Cela aide souvent à améliorer les performances et à réduire le surajustement.")
        st.write("Matrice de confusion :")
        y_pred = rf.predict(X_test)
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Rapport de classification :")
        st.write(classification_report(y_test, y_pred))
        
        # Autres algorithmes de ML
        st.subheader("Autres algorithmes de machine learning")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        st.write('Score de la régression logistique:', lr.score(X_test, y_test))
        svm = SVC()
        svm.fit(X_train, y_train)
        st.write('Score du SVM:', svm.score(X_test, y_test))

# Point d'entrée de l'application
if __name__ == '__main__':
    main()
