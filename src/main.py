import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from joblib import load
from sklearn.neighbors import BallTree

from src.models.precipitation_surface_model import InputDataPrecipit
from src.models.soil_moisture_model import InputData
from src.services.soil_moisture_npd_serv import predecir_RF_model
from src.utils.handle_respose import send_error_response

# from tensorflow.keras.models import load_model


app = FastAPI()

SOIL_MOISTURE_MODEL_PATH = "src/model/soilmoisture/random_forest_model_sin_pca.joblib"
SOIL_MOISTURE_SCALER_PATH = "src/model/soilmoisture/scaler.joblib"
PRECIPITATION_SURFACE_PATH = "src/model/precipitationModel/ann_model.h5"
PRECIPITATION_SCALER_X = "src/model/precipitationModel/scaler_X.pkl"
PRECIPITATION_SCALER_Y = "src/model/precipitationModel/scaler_y.pkl"

# Configuracion de CORS
origins = [
    "http://localhost:5173",  # Para permitir solicitudes desde localhost
    "https://mi-app-frontend.com",  # Dominio de produccion permitido
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Permitir cookies y credenciales
    allow_methods=["*"],  # Metodos permitidos (GET, POST, etc.)
    allow_headers=["*"],  # Encabezados permitidos
)

try:
    rf_model = load(SOIL_MOISTURE_MODEL_PATH)
    scaler = load(SOIL_MOISTURE_SCALER_PATH)

    print(f"Modelo y scaler cargados desde {SOIL_MOISTURE_MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    rf_model = None
    scaler = None


print("Verificando archivo scaler_X:", os.path.exists(PRECIPITATION_SCALER_X))
print("Ruta absoluta:", os.path.abspath(PRECIPITATION_SCALER_X))
try:
    ann_model = load_model(PRECIPITATION_SURFACE_PATH, compile=False)
    scaler_X_precip = load(PRECIPITATION_SCALER_X)
    scaler_y_precip = load(PRECIPITATION_SCALER_Y)

    print(f"Escalador: {scaler_X_precip}")
    print(
        f"Modelo de precipitacion y escalers cargados desde {PRECIPITATION_SURFACE_PATH}"
    )
except Exception as e:
    print(f"Error al cargar el modelo de precipitacion: {e}")
    ann_model = None
    scaler_X_precip = None
    scaler_y_precip = None


# Cargamos el archivo CSV y preparamos los datos
csv_file_path = "src/data_actual/01-12-2024 to 16-12-2024.csv"
data_for_soil_moisture = pd.read_csv(csv_file_path)


@app.get("/", response_class=HTMLResponse)
def read_root():
    return "<h1>Bienvenido a Agro Healthy IA</h1>"


@app.post("/predict")
def predict_values(input_data: InputData):
    try:
        print(input_data)

        # transformar los datos de entrada a un DataFrame
        df_input = pd.DataFrame([data.dict() for data in input_data.data])

        print("funciona 1")
        print(df_input)

        df_results = predecir_RF_model(df_input, data_for_soil_moisture)

        print("funciona 2")

        # Convertir el resultado a un diccionario
        results = df_results.to_dict(orient="records")

        print("funciona 3")

        return results
    except Exception as e:
        print(f"Error al predecir los valores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc: HTTPException):
    # Comprobamos si `exc.detail` es un dict, y lo formateamos.
    if isinstance(exc.detail, dict):
        return send_error_response(
            exc.status_code, exc.detail["message"], exc.detail["error"]
        )
    print("Error en el servidor")
    return send_error_response(exc.status_code, exc.detail)


# Ruta para hacer predicciones
@app.post("/predictPrecipit")
def predict_precipitation(input_data: InputDataPrecipit):
    try:
        # Extraer los valores de los campos y construir la matriz de entrada
        nuevos_datos = np.array(
            [
                [
                    data.ConvectivePrecip,
                    data.ProbabilityofPrecip,
                    data.SunglintAngle,
                    data.Temp2Meter,
                    data.TotalColWaterVapor,
                    data.Latitude,
                    data.Longitude,
                ]
                for data in input_data.data
            ]
        )

        # Escalar los datos
        nuevos_datos_scaled = scaler_X_precip.transform(nuevos_datos)

        prediccion_scaled = ann_model.predict(nuevos_datos_scaled)

        prediccion = scaler_y_precip.inverse_transform(prediccion_scaled)

        return {
            "prediccion_escalada": prediccion_scaled.tolist(),
            "prediccion_original": prediccion.tolist(),
            "unidad": "mm/h",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
