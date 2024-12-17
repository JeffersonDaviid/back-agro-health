import math

import numpy as np
import pandas as pd
from joblib import load
from scipy.interpolate import LinearNDInterpolator
from sklearn.impute import SimpleImputer


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia Haversine entre dos coordenadas geográficas (lat, lon).
    """
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Fórmula de Haversine
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radio de la Tierra en kilómetros
    radius = 6371.0
    return radius * c  # Distancia en kilómetros


def obtener_coordenadas_cercanas(lat, lon, df, k=5):
    """
    Obtiene las k coordenadas más cercanas en el DataFrame df a la coordenada (lat, lon) utilizando distancia Haversine.
    """
    distancias = df.apply(
        lambda row: haversine(lat, lon, row["Latitude"], row["Longitude"]), axis=1
    )
    df["distancia"] = distancias
    return df.nsmallest(k, "distancia")


def rellenar_con_coordenadas_cercanas(df_predecir, df_datos_llenos, required_columns):
    """
    Rellena los datos faltantes en df_predecir usando los datos más cercanos de df_datos_llenos.
    """
    # Excluir las columnas de fecha (year, month, day) del cálculo del promedio
    date_columns = ["year", "month", "day"]
    numeric_columns = [col for col in required_columns if col not in date_columns]

    for idx, row in df_predecir.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]

        # Buscar si la coordenada ya existe en df_datos_llenos
        if not df_datos_llenos[
            (df_datos_llenos["Latitude"] == lat) & (df_datos_llenos["Longitude"] == lon)
        ].empty:
            # Si está, tomar los datos directamente
            datos_cercanos = df_datos_llenos[
                (df_datos_llenos["Latitude"] == lat)
                & (df_datos_llenos["Longitude"] == lon)
            ]
        else:
            # Si no está, buscar las coordenadas más cercanas
            puntos_cercanos = obtener_coordenadas_cercanas(lat, lon, df_datos_llenos)
            # Solo promedia las columnas numéricas, no las de fecha
            datos_cercanos = puntos_cercanos[
                numeric_columns
            ].mean()  # Promediar los datos de las coordenadas más cercanas

        # Rellenar el df_predecir con los valores obtenidos, sin modificar las columnas de fecha
        for col in required_columns:
            if col in date_columns:
                # Si la columna es de fecha, no se promedia, se asigna el valor original
                df_predecir.loc[idx, col] = row[col]
            else:
                # Para las demás columnas, asignar el valor promedio
                df_predecir.loc[idx, col] = (
                    datos_cercanos[col] if col in datos_cercanos else row[col]
                )

    return df_predecir


def predecir_con_interpolacion(
    lat, lon, features, rf_model, scaler, interpolator, df_datos_llenos
):
    """
    Realiza predicción utilizando interpolación para coordenadas cercanas y predicción de Random Forest.
    """
    # Obtener los puntos más cercanos del DataFrame `df_datos_llenos`
    puntos_cercanos = obtener_coordenadas_cercanas(lat, lon, df_datos_llenos)

    # Si encontramos puntos cercanos, promediamos sus valores para la interpolación
    if not puntos_cercanos.empty:
        # Promediar los valores de SoilMoistureNPD de las coordenadas cercanas
        interpolated_value = puntos_cercanos["SoilMoistureNPD"].mean()
    else:
        # Si no hay puntos cercanos, usamos la interpolación espacial
        interpolated_value = interpolator(np.array([[lat, lon]]))[0]

    # Si la interpolación falla, usar el valor promedio general
    if np.isnan(interpolated_value):
        interpolated_value = df_datos_llenos["SoilMoistureNPD"].mean()

    # Crear entrada para el modelo (incluyendo latitud y longitud)
    input_features = np.array([lat, lon] + features).reshape(
        1, -1
    )  # Lat y Lon están aquí
    input_features_scaled = scaler.transform(
        input_features
    )  # Escalar las características

    # Predicción del modelo Random Forest
    rf_prediction = rf_model.predict(input_features_scaled)[0]

    # Combinar la predicción del modelo con la interpolación
    final_prediction = (rf_prediction + interpolated_value) / 2

    return final_prediction


def load_resources(model_path, scaler_path, df_datos_llenos):
    """
    Carga el modelo de Random Forest, el escalador y los datos de interpolación.
    """
    # Cargar el modelo y el escalador
    rf_model = load(model_path)
    scaler = load(scaler_path)

    # Configurar el interpolador espacial para SoilMoisture
    interpolator = LinearNDInterpolator(
        df_datos_llenos[["Latitude", "Longitude"]].values,
        df_datos_llenos["SoilMoistureNPD"].values,
    )

    return rf_model, scaler, interpolator


def interpolar_df(df):
    df["fecha"] = pd.to_datetime(df[["year", "month", "day"]])

    # Ahora puedes ordenar tu DataFrame por la nueva columna 'fecha'
    df = df.sort_values(by=["fecha"])

    # Establecer la columna 'fecha' como índice
    df = df.set_index("fecha")

    # Realizar la interpolación temporal
    df.interpolate(method="time", inplace=True)

    df = df.reset_index()
    df = df.drop(["fecha"], axis=1)

    imputer = SimpleImputer(
        strategy="mean"
    )  # Crea un objeto SimpleImputer con la estrategia 'mean'
    df = pd.DataFrame(
        imputer.fit_transform(df), columns=df.columns
    )  # Aplica la imputación y crea un nuevo DataFrame

    return df


def predecir_RF_model(
    df_predecir,
    df_datos_llenos,
    model_path="src/model/soilmoisture/random_forest_model_sin_pca.joblib",
    scaler_path="src/model/soilmoisture/scaler.joblib",
):
    """
    Predicción de SoilMoisture utilizando Random Forest e interpolación espacial.
    """
    # Cargar recursos
    rf_model, scaler, interpolator = load_resources(
        model_path, scaler_path, df_datos_llenos
    )

    required_columns = [
        "Latitude",
        "Longitude",
        "TBH10r2",
        "TBV10r2",
        "TBH18r2",
        "TBV18r2",
        "TBH23r2",
        "TBV23r2",
        "TBH36r2",
        "TBV36r2",
        "TBH89r2",
        "TBV89r2",
        "VegetationRoughnessNPD",
        "RetrievalQualityFlagNPD",
        "RetrievalQualityFlagSCA",
        "FlagCountAllSamples",
        "FlagCountGoodSamples",
        "FlagCountRain",
        "FlagCountLow2ModerateVWC",
        "FlagCountDenseVWC",
        "year",
        "month",
        "day",
    ]

    # Verificar si todas las columnas requeridas están presentes en df_predecir
    missing_columns = [
        col for col in required_columns if col not in df_datos_llenos.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Faltan las siguientes columnas en df_predecir: {missing_columns}"
        )

    # Rellenar df_predecir con los valores de df_datos_llenos
    df_predecir = rellenar_con_coordenadas_cercanas(
        df_predecir, df_datos_llenos, required_columns
    )

    # Escalar las características
    df_predecir_scaled = scaler.fit_transform(
        df_predecir[required_columns]
    )  # Usar las mismas columnas que en el entrenamiento

    # Realizar las predicciones con el modelo
    df_predecir["SoilMoistureNPD_predicted"] = rf_model.predict(df_predecir_scaled)

    # Realizar la interpolación y combinar con la predicción del modelo
    final_predictions = []
    for idx, row in df_predecir.iterrows():
        lat = row["Latitude"]
        lon = row["Longitude"]
        features = row[required_columns[2:]].tolist()  # Características sin lat/lon
        final_prediction = predecir_con_interpolacion(
            lat, lon, features, rf_model, scaler, interpolator, df_datos_llenos
        )
        final_predictions.append(final_prediction)

    # Añadir las predicciones finales al DataFrame
    df_predecir["moistureData"] = final_predictions

    return df_predecir[
        [
            "Latitude",
            "Longitude",
            "year",
            "month",
            "day",
            "moistureData",
            "VegetationRoughnessNPD",
        ]
    ]
