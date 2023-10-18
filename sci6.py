# Importations
import streamlit as st
from sklearn import datasets
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# Initialisation de l'état de la session
if 'df' not in st.session_state:
    iris = datasets.load_iris()
    st.session_state.df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

# Fonction principale de l'application Streamlit
def main():
    st.title("Exploration Avancée de Data Science avec Streamlit")
    st.sidebar.title("Navigation et Widgets")

    # Navigation avec les nouveaux widgets
    page = st.sidebar.radio("Choisissez une page", ["Introduction", "Statistiques Descriptives", "DataFrame Modifiable", "Conclusion"])

    # Utilisation de st.caption
    st.caption("Explication: Utilisez le menu latéral pour naviguer entre les pages.")

    # Introduction
    if page == "Introduction":
        st.subheader("Introduction")
        st.write("Dans cette application, nous explorerons le dataset Iris avec de nombreux widgets Streamlit.")
        
        # Utilisation de st.form et st.form_submit_button
        with st.form("form_intro"):
            st.write("Entrez un message pour continuer:")
            user_input = st.text_input("Votre message")
            submitted = st.form_submit_button("Soumettre")
            if submitted:
                st.success(f"Vous avez soumis le message : {user_input}")

    # Statistiques Descriptives
    elif page == "Statistiques Descriptives":
        st.subheader("Statistiques Descriptives avec Pandas Profiling")
        
        # Utilisation de st.download_button
        report = ProfileReport(st.session_state.df, explorative=True)
        st_profile_report(report)

        st.download_button("Téléchargez ce rapport", data=report.to_file(output_file="report.html"), mime="text/html")

    # DataFrame Modifiable
    elif page == "DataFrame Modifiable":
        st.subheader("DataFrame Modifiable")
        
        # Utilisation de st.experimental_data_editor
        edited_df = st.experimental_data_editor(st.session_state.df)
        st.session_state.df = edited_df

        # Utilisation de st.color_picker
        color = st.color_picker("Choisissez une couleur pour le DataFrame", "#00f900")
        st.dataframe(st.session_state.df, color)

    # Conclusion
    elif page == "Conclusion":
        st.subheader("Conclusion")
        
        # Utilisation de st.session_state
        if 'conclusion_text' not in st.session_state:
            st.session_state.conclusion_text = "Nous avons utilisé divers widgets Streamlit pour cette démo."

        # Utilisation de st.text_area
        st.session_state.conclusion_text = st.text_area("Texte de conclusion:", st.session_state.conclusion_text)
        st.write(f"Conclusion : {st.session_state.conclusion_text}")

# Point d'entrée de l'application
if __name__ == '__main__':
    main()
