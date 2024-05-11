import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('model/regression_model_v01.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title('Predicción Salarial de Desarrollador de Software')

    st.write("""### Necesitamos información para predecir el salario""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom of Great Britain and Northern Ireland",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Australia",
        "Netherlands",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("País", countries)
    education = st.selectbox("Nivel de Educación", education)

    experience = st.slider("Años de Experiencia", 0, 50)

    ok = st.button("Calcular Salario")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor_loaded.predict(X)
        st.subheader(f"El salario estimado es ${salary[0]:.2f}")