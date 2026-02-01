import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
import sys

# Configure logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.logs_configs.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting preprocessor")


def load_data(filepath: str) -> pd.DataFrame:
    """Carga el conjunto de datos desde la ruta especificada."""
    logger.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)


def preprocess_data(
    df: pd.DataFrame, target_column: str = "Risk", save_path: str = None
):
    """
    Preprocesa el conjunto de datos de riesgo crediticio.

    Args:
        df: DataFrame de entrada.
        target_column: Nombre de la columna objetivo.
        save_path: Ruta para guardar el pipeline del preprocesador.

    Returns:
        X_train, X_test, y_train, y_test: Divisiones de datos procesados.
    """
    logger.info("Starting data preprocessing")

    # Eliminar 'Unnamed: 0' si existe (columna de índice residual)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Separación de características (X) y objetivo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Codificación del objetivo: 'good' (bueno) -> 1, 'bad' (malo) -> 0
    y = y.map({"good": 1, "bad": 0}).astype(int)

    # Definición de columnas numéricas y categóricas
    numerical_features = ["Age", "Credit amount", "Duration"]
    categorical_features = [
        "Sex",
        "Job",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose",
    ]

    # Preprocesamiento para datos numéricos: imputación y estandarización
    numerical_transformer = Pipeline(
        steps=[
            # Imputa valores faltantes utilizando la media de la columna
            ("imputer", SimpleImputer(strategy="mean")),
            # Normaliza las características eliminando la media y escalando a la varianza unitaria
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocesamiento para datos categóricos: imputación y one-hot encoding
    categorical_transformer = Pipeline(
        steps=[
            # Imputa valores faltantes con la etiqueta 'unknown'
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            # Convierte variables categóricas en vectores binarios (one-hot)
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Empaquetar preprocesamiento para datos numéricos y categóricos
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Fitting preprocessor")
    # Ajustar y transformar los datos
    X_processed = preprocessor.fit_transform(X)

    # Guardar el preprocesador entrenado si se especifica una ruta
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        logger.info(f"Preprocessor saved to {save_path}")

    logger.info("Splitting data")
    # Dividir los datos en entrenamiento y prueba (80% train, 20% test)
    return train_test_split(X_processed, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Example usage for testing
    dataset_path = os.path.join(
        "datasets",
        "credit_score_dataset",
        "german_credit_risk_v1.0.0_training_23012026.csv",
    )
    if os.path.exists(dataset_path):
        df = load_data(dataset_path)
        X_train, X_test, y_train, y_test = preprocess_data(
            df, save_path="python-projects/credit-score/processing/preprocessor.joblib"
        )
        print("Data shape:", X_train.shape)
    else:
        logger.error(f"Dataset not found at {dataset_path}")
