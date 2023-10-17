
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway
import numpy as np
import seaborn as sns

st.title('Machine Learning sur la feuille GPT')

st.write("Modèle utilisé : RandomForestRegressor")
st.write("Pourcentage des données utilisé en entraînement : 80%")
st.write("Pourcentage des données utilisé en test : 20%")

# Upload du fichier Excel
uploaded_file = st.file_uploader("Choisissez un fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    # Lecture du fichier Excel et de la feuille GPT
    df = pd.read_excel(uploaded_file, sheet_name='GPT')
    
    # Visualisation des données
    st.subheader("Visualisation des données")
    st.write(df.head())
    
    # Suppression des lignes où les valeurs cibles sont manquantes
    df.dropna(subset=['PaymentDuration', '%Paiements', 'NombrePaimentClient'], inplace=True)
    
    # Gestion des valeurs manquantes
    df.fillna(0, inplace=True)
    
    # Conversion des colonnes problématiques en chaînes de caractères
    for col in ['Division', 'ProjectManager', 'OrganizationName']:
        df[col] = df[col].astype(str)
    
    # Encodage des variables catégorielles
    for col in ['Division', 'ProjectManager', 'OrganizationName']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # Séparation des features et des targets
    features = ['InvoiceAmount', 'DuePeriod', 'PaymentDuration', '%Paiements', 'NombrePaimentClient', 'Division', 'ProjectManager', 'OrganizationName']
    targets = ['PaymentDuration', '%Paiements', 'NombrePaimentClient']
    
    X = df[features]
    y = df[targets]
    
    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement des modèles
    models = {}
    for target in targets:
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train[target])
        models[target] = model
    
    # Évaluation des modèles
    st.subheader("Évaluation des modèles")
    evaluation_metrics = {}
    for target, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test[target], y_pred)
        r2 = r2_score(y_test[target], y_pred)
        evaluation_metrics[target] = {'MSE': mse, 'R2': r2}
    
    st.write("Métriques d'évaluation des modèles :", evaluation_metrics)
    
    # Analyse ANOVA
    st.subheader("Analyse ANOVA")
    for target in targets:
        anova_results = []
        for feature in features:
            groups = [y[target][X[feature] == unique_val] for unique_val in X[feature].unique()]
            f_val, p_val = f_oneway(*groups)
            anova_results.append({'Feature': feature, 'F-value': f_val, 'P-value': p_val})
        st.write(f"Résultats de l'ANOVA pour {target}:")
        st.write(pd.DataFrame(anova_results))
