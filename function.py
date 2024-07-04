import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_model(data_path):
    data = pd.read_csv(data_path)

    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)

    # Encodage des labels
    label_encoder = LabelEncoder()
    data['size_encoded'] = label_encoder.fit_transform(data['size'])

    # Sélection des caractéristiques et de la cible
    features = data[['weight', 'age', 'height']]
    target = data['size_encoded']

    # Division du jeu de données
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Création du modèle TensorFlow
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

    # Sauvegarder le modèle
    model.save('clothing_size_model.h5')

    return model

def predict_size(weight, age, height):
    model = tf.keras.models.load_model('clothing_size_model.h5')

    # Simuler la normalisation des données
    scaler = StandardScaler()
    example_data = np.array([[weight, age, height]])
    new_data_scaled = scaler.fit_transform(example_data)
    
    predicted_size_encoded = model.predict(new_data_scaled)
    predicted_size_encoded = int(round(predicted_size_encoded[0][0]))

    # Encodeur de taille (manuellement défini pour éviter la sauvegarde)
    size_labels = ['XS', 'S', 'M', 'L', 'XL']
    predicted_size = size_labels[predicted_size_encoded]

    return predicted_size
