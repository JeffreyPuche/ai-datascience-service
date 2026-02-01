import torch
import joblib
import yaml
import pandas as pd
import os
import sys
import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.logs_configs.logging_config import setup_logging
from model.model import CreditScoreModel, ModelConfig
from server.schemas import CreditRiskInput

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class Predictor:
    """
    Singleton class for model inference.
    Loads the model weights, configuration, and preprocessor.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        # 1. Revisa si ya existe una instancia guardada en la variable de clase `_instance`
        if not cls._instance:
            # 2. Si NO existe, llama al método de la clase padre (super) para crearla físicamente
            cls._instance = super(Predictor, cls).__new__(cls)
        # Si ya existía, simplemente devuelve la instancia que ya estaba creada
        return cls._instance

    def __init__(self, model_path: str, config_path: str, preprocessor_path: str):
        # Asegurar que la inicialización ocurra solo una vez
        if Predictor._initialized:
            return

        logger.info("Initializing Predictor Singleton")

        # 1. Load configuration from YAML
        logger.info(f"Loading model config from {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # 2. Load preprocessor (joblib)
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        # 3. Determinar el tamaño de entrada a partir del preprocesador
        # Transformamos una entrada ficticia para obtener el número esperado de características para la red neuronal
        dummy_df = pd.DataFrame(
            [
                {
                    "Age": 30,
                    "Sex": "male",
                    "Job": 2,
                    "Housing": "own",
                    "Saving accounts": "NA",
                    "Checking account": "NA",
                    "Credit amount": 1000.0,
                    "Duration": 12,
                    "Purpose": "car",
                }
            ]
        )

        # El preprocesador espera que las columnas coincidan con las de entrenamiento.
        processed_dummy = self.preprocessor.transform(dummy_df)
        input_size = processed_dummy.shape[1]
        logger.info(f"Determined model input size: {input_size}")

        # 4. Inicializar la arquitectura del modelo utilizando la configuración cargada
        model_config = ModelConfig(
            input_size=input_size,
            hidden_layers=yaml_config["hidden_layers"],
            activation_functions=yaml_config["activation_functions"],
            dropout_rate=yaml_config["dropout_rate"],
            learning_rate=yaml_config["learning_rate"],
            epochs=yaml_config["epochs"],
            batch_size=yaml_config["batch_size"],
        )

        self.model = CreditScoreModel(model_config)

        # 5. Load model weights (.pth)
        logger.info(f"Loading model weights from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        # Cargar el estado del modelo y mapear a la CPU
        device = torch.device("cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        Predictor._initialized = True
        logger.info("Predictor successfully initialized and optimized for inference.")

    def inference(self, data: CreditRiskInput) -> dict:
        """
        Performs inference on the provided input data.

        Args:
            data (CreditRiskInput): Input data matching the schema.

        Returns:
            dict: {
                "prediction": "good" | "bad",
                "probability": float (probability of being "good")
            }
        """
        # 1. Convierte un objeto Pydantic en un diccionario usando alias (para que coincida con "Saving accounts" etc)
        input_dict = data.model_dump(by_alias=True)

        # 2. Crear DataFrame para el preprocesador
        df = pd.DataFrame([input_dict])

        # 3. Aplicar el preprocesador
        processed_data = self.preprocessor.transform(df)

        # 4. Convertir a Tensor
        device = next(self.model.parameters()).device
        tensor_data = torch.FloatTensor(processed_data).to(device)

        # 5. Predicción del modelo
        with torch.no_grad():
            # predict_probability returns a tensor with [prob_bad, prob_good]
            probs = self.model.predict_probability(tensor_data)
            good_prob = probs[0][1].item()

            # Predicción binaria
            prediction_idx = torch.argmax(probs, dim=1).item()
            prediction = "good" if prediction_idx == 1 else "bad"

        return {"prediction": prediction, "probability": good_prob}


# --- Instanciación para arquitectura Singleton ---
# Rutas por defecto basadas en la estructura del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_weights_001.pth")
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "config", "models-configs", "model_config_001.yaml"
)
DEFAULT_PREPROCESSOR_PATH = os.path.join(
    PROJECT_ROOT, "processing", "preprocessor.joblib"
)

# Instancia global (ya cargada)
predictor = Predictor(
    model_path=DEFAULT_MODEL_PATH,
    config_path=DEFAULT_CONFIG_PATH,
    preprocessor_path=DEFAULT_PREPROCESSOR_PATH,
)

if __name__ == "__main__":
    # Prueba de la clase y la función de inferencia
    logger.info("\n" + "=" * 50)
    logger.info("TESTING CREDIT SCORE PREDICTOR")
    logger.info("=" * 50)

    # Datos de prueba siguiendo el esquema CreditRiskInput
    sample_input = CreditRiskInput(
        Age=35,
        Sex="male",
        Job="skilled",
        Housing="own",
        Saving_accounts="NA",
        Checking_account="little",
        Credit_amount=9055.0,
        Duration=36,
        Purpose="education",
    )

    logger.info(
        f"\n[1] Input data sample:\n{sample_input.model_dump_json(indent=4, by_alias=True)}"
    )

    # Realizar inferencia
    try:
        result = predictor.inference(sample_input)
        logger.info(f"\n[2] Inference result:\n{result}")

        # Verificar Singleton
        predictor_copy = Predictor(
            DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, DEFAULT_PREPROCESSOR_PATH
        )
        is_singleton = predictor is predictor_copy
        logger.info(f"\n[3] Singleton Verification: {is_singleton}")

        if is_singleton and "prediction" in result:
            logger.info(
                "\nSUCCESS: Predictor class and inference function are working correctly!"
            )
        else:
            logger.info("\nFAILURE: Check implementation logic.")

    except Exception as e:
        logger.info(f"\nERROR during inference: {str(e)}")
        import traceback

        traceback.print_exc()
