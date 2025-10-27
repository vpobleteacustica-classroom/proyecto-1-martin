

#Funciones del clasificador 

import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def extraer_caracteristicas(dataset_path):
    
    datos = [] 
    generos = os.listdir(dataset_path)

    print("Iniciando extracción...")
    for genero in generos:
        ruta_genero = os.path.join(dataset_path, genero)
        if not os.path.isdir(ruta_genero):
            continue
            
        for archivo in os.listdir(ruta_genero):
            if archivo.endswith('.wav'):
                ruta_archivo = os.path.join(ruta_genero, archivo)
                try:
                    # Cargar audio
                    y, sr = librosa.load(ruta_archivo, mono=True, duration=30)
                    
                    # Extraer características (y tomar el promedio)
                    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
                    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
                    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    
                    # Guardar fila
                    fila = np.hstack((mfccs, chroma, spec_contrast, zcr, tempo, genero))
                    datos.append(fila)
                
                except Exception as e:
                    print(f"Error procesando {ruta_archivo}: {e}")

    

    print("Fin de la extraccion ...")


    # Crear DataFrame
    column_names = [f'mfcc_{i}' for i in range(20)] + \
                   [f'chroma_{i}' for i in range(12)] + \
                   [f'spec_contrast_{i}' for i in range(7)] + \
                   ['zcr', 'tempo', 'genre']
                   
    df = pd.DataFrame(datos, columns=column_names)
    return df

# Función 2: El trabajo de preparación
def preprocesar_datos(df):
    
    
    df = df.dropna() 
    
    # Codificar etiquetas 'y'
    encoder = LabelEncoder()
    df['genre'] = encoder.fit_transform(df['genre'])
    
    # Separar X e y
    X = df.drop('genre', axis=1)
    y = df['genre']
    
    # Escalar 'X'
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, encoder