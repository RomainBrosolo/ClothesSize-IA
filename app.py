import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Charger le modèle
model = tf.keras.models.load_model('model/clothing_size_model.h5')

# Encodeur de taille (manuellement défini pour éviter la sauvegarde)
size_labels = ['XS', 'S', 'M', 'L', 'XL']

st.title("Prédiction de la Taille de Vêtements")

st.write("Entrez les caractéristiques suivantes pour obtenir une prédiction de la taille de vêtements.")

weight = st.number_input("Poids (kg)", min_value=0.0)
age = st.number_input("Âge (années)", min_value=0)
height = st.number_input("Hauteur (cm)", min_value=0.0)

if st.button("Prédire la taille"):
    new_data = np.array([[weight, age, height]])
    
    # Normalisation des données
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)
    
    predicted_size_encoded = model.predict(new_data_scaled)
    predicted_size_encoded = int(round(predicted_size_encoded[0][0]))
    
    predicted_size = size_labels[predicted_size_encoded]
    
    st.success(f"La taille prédite pour un homme de {weight} kg, {age} ans et {height} cm est : {predicted_size}")
