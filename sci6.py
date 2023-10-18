# Importations
import streamlit as st
from sklearn import datasets
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# Fonction principale de l'application Streamlit
def main():
    st.title("Exploration de Data Science avec Pandas Profiling et DataFrame modifiable")
    st.sidebar.title("Navigation")

    # Navigation
    page = st.sidebar.selectbox("Choisissez une page", ["Introduction", "Statistiques Descriptives", "DataFrame Modifiable", "Conclusion"])

    # Chargement du dataset Iris
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

    # Introduction
    if page == "Introduction":
        st.subheader("Introduction")
        st.write("Dans cette application, nous explorerons le dataset Iris avec Pandas Profiling et un DataFrame modifiable.")

    # Statistiques Descriptives
    if page == "Statistiques Descriptives":
        st.subheader("Statistiques Descriptives avec Pandas Profiling")
        report = ProfileReport(df, explorative=True)
        st_profile_report(report)

    # DataFrame Modifiable
    if page == "DataFrame Modifiable":
        st.subheader("DataFrame Modifiable")
        edited_df = st.experimental_data_editor(df)
        st.write("DataFrame après modification:")
        st.dataframe(edited_df)

    # Conclusion
    if page == "Conclusion":
        st.subheader("Conclusion")
        st.write("Nous avons utilisé Pandas Profiling pour l'analyse statistique et st.experimental_data_editor pour la modification du DataFrame.")

# Point d'entrée de l'application
if __name__ == '__main__':
    main()
