import librosa as lr
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- FUNCIÓN 1: EXTRACCIÓN DE CARACTERÍSTICAS POR CANCIÓN ---
def extraer_caracteristicas(dataset_path):
    datos = []
    generos = os.listdir(dataset_path)

    # CONFIGURACIÓN
    DURACION_CANCION = 30

    print("Iniciando extracción de características (una fila por canción)...")

    for genero in generos:
        ruta_genero = os.path.join(dataset_path, genero)
        if not os.path.isdir(ruta_genero):
            continue

        print(f"--> Procesando: {genero}")

        for archivo in os.listdir(ruta_genero):
            if archivo.endswith('.wav'):
                ruta_archivo = os.path.join(ruta_genero, archivo)
                try:
                    y, sr = lr.load(ruta_archivo, mono=True, duration=DURACION_CANCION)

                    # EXTRACCIÓN DE CARACTERÍSTICAS
                    mfcc = lr.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                    chroma = lr.feature.chroma_stft(y=y, sr=sr)
                    contrast = lr.feature.spectral_contrast(y=y, sr=sr)
                    zcr = lr.feature.zero_crossing_rate(y)

                    fila = []

                    # Estadísticas
                    fila.extend(np.mean(mfcc.T, axis=0))
                    fila.extend(np.var(mfcc.T, axis=0))
                    fila.extend(np.mean(chroma.T, axis=0))
                    fila.extend(np.var(chroma.T, axis=0))
                    fila.extend(np.mean(contrast.T, axis=0))
                    fila.extend(np.var(contrast.T, axis=0))
                    fila.extend(np.mean(zcr.T, axis=0))
                    fila.extend(np.var(zcr.T, axis=0))

                    # Tempo
                    hop_length = 512
                    oenv = lr.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
                    tempo = lr.beat.tempo(onset_envelope=oenv, sr=sr)[0]
                    fila.append(tempo)

                    # Guardamos el nombre del archivo para identificar la canción
                    fila.append(archivo)

                    # Etiqueta
                    fila.append(genero)

                    datos.append(fila)

                except Exception as e:
                    print(f"Error procesando {ruta_archivo}: {e}")

    # NOMBRES DE COLUMNAS
    columnas = []
    for i in range(20): columnas.append(f'mfcc_mean_{i}')
    for i in range(20): columnas.append(f'mfcc_var_{i}')
    for i in range(12): columnas.append(f'chroma_mean_{i}')
    for i in range(12): columnas.append(f'chroma_var_{i}')
    for i in range(7):  columnas.append(f'contrast_mean_{i}')
    for i in range(7):  columnas.append(f'contrast_var_{i}')
    columnas.append('zcr_mean')
    columnas.append('zcr_var')
    columnas.append('tempo')
    
    columnas.append('filename') # Columna para identificar el archivo
    columnas.append('label') 

    df = pd.DataFrame(datos, columns=columnas)
    return df

# --- FUNCIÓN 2: PREPROCESAMIENTO ESTÁNDAR ---
def preprocesar_datos(df):
    df = df.dropna()

    # 1. Separamos X (Features) e y (Target)
    # La columna 'filename' ya no es necesaria para el split, pero la quitamos
    # para no dársela al modelo.
    X = df.drop(['label', 'filename'], axis=1)
    y = df['label']

    # 2. Forzamos que las features sean numéricas
    X = X.apply(pd.to_numeric, errors='coerce')

    # 3. Codificar etiquetas de texto a números
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 4. SPLIT ESTÁNDAR (Train/Test)
    # Ya no necesitamos GroupShuffleSplit porque tenemos una fila por canción.
    # Usamos stratify para mantener la proporción de géneros en train y test.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 5. Escalar las características (Fit solo en train para evitar fuga de datos)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("Split Estándar Completado.")
    print(f"Canciones para entranamiento: {X_train.shape[0]}")
    print(f"Canciones para prueba: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler, encoder, X.columns