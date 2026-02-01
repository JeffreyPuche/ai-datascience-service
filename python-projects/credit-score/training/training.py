import argparse
import sys
import os
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)


# para importar desde directorios vecinos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.logs_configs.logging_config import setup_logging
from model.model import CreditScoreModel, ModelConfig
from processing.preprocessor import load_data, preprocess_data

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting Training")


def load_config(config_path):
    """
    Carga y analiza un archivo de configuración YAML.

    Esta función se utiliza para recuperar parámetros de entrenamiento y configuraciones del modelo
    definidos en archivos YAML externos, permitiendo una experimentación flexible.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(args):
    """
    Función principal de entrenamiento para orquestar el proceso de entrenamiento del modelo.

    Args:
        args (argparse.Namespace): Argumentos de línea de comandos analizados que contienen la ruta de configuración.
    """
    config_path = args.config
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Configuración de MLflow
    mlflow.set_experiment("Credit Score Training")

    with mlflow.start_run(run_name=config_name):
        # Registrar parámetros
        mlflow.log_params(config)
        mlflow.log_param("config_file", config_name)

        # Cargar y preprocesar datos
        # Asumiendo que la ruta del conjunto de datos es relativa a la raíz del proyecto o fija
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "datasets",
            "credit_score_dataset",
            "german_credit_risk_v1.0.0_training_23012026.csv",
        )
        dataset_path = os.path.abspath(dataset_path)  # Normalizar ruta

        logger.info(f"Loading data from {dataset_path}")
        df = load_data(dataset_path)

        preprocessor_path = os.path.join(
            os.path.dirname(__file__), "..", "processing", "preprocessor.joblib"
        )
        X_train, X_test, y_train, y_test = preprocess_data(
            df, save_path=preprocessor_path
        )

        # Convertir a tensores de PyTorch
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # Crear DataLoaders para manejo de lotes (batching)
        batch_size = config.get("batch_size", 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Inicializar el Modelo
        input_size = X_train.shape[1]
        model_config = ModelConfig(
            input_size=input_size,
            output_size=1,
            hidden_layers=config["hidden_layers"],
            activation_functions=config["activation_functions"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=batch_size,
        )

        model = CreditScoreModel(model_config)

        # Función de pérdida y Optimizador
        # Binary Cross Entropy with Logits Loss
        criterion = nn.BCEWithLogitsLoss()
        # AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

        # Bucle de Entrenamiento
        logger.info("Starting training")
        epochs = config["epochs"]
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Reiniciar gradientes
                outputs = model(inputs)  # Pase hacia adelante (Forward pass)
                loss = criterion(outputs, labels)  # Calcular pérdida
                loss.backward()  # Pase hacia atrás (Backward pass)
                optimizer.step()  # Actualizar pesos

                # Acumular pérdida para calcular el promedio de la época
                running_loss += loss.item()

                # Calcular precisión para este lote
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total

            logger.info(
                f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
            )
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

        # Evaluación
        logger.info("Evaluating model")
        # Cambiar el modelo a modo de evaluación
        # Desactiva capas específicas como Dropout y Batch Normalization
        model.eval()
        # Desactivar cálculo de gradientes
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.sigmoid(outputs).numpy()
            preds = (probs > 0.5).astype(int)
            y_true = y_test_tensor.numpy()

            # Métricas
            acc = accuracy_score(y_true, preds)
            roc_auc = roc_auc_score(y_true, probs)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)
            f1 = f1_score(y_true, preds)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)

            logger.info(f"Test Accuracy: {acc:.4f}")
            logger.info(f"Test ROC AUC: {roc_auc:.4f}")

            # Graficar Matriz de Confusión
            cm = confusion_matrix(y_true, preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()

            # Graficar Curva ROC
            fpr, tpr, _ = roc_curve(y_true, probs)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "roc_curve.png")
            plt.close()

            # Graficar Curva Precision-Recall
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, probs)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_curve, precision_curve, label="Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "precision_recall_curve.png")
            plt.close()

            # Reporte de Texto
            report = classification_report(y_true, preds)
            mlflow.log_text(report, "classification_report.txt")

        # Guardar pesos del modelo
        save_dir = os.path.join(os.path.dirname(__file__), "..", "model")
        os.makedirs(save_dir, exist_ok=True)
        # Reemplazar 'model_config' por 'model_weights' en el nombre del archivo
        weights_name = config_name.replace("model_config", "model_weights")
        model_save_path = os.path.join(save_dir, f"{weights_name}.pth")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model weights saved to {model_save_path}")
        mlflow.log_artifact(model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Score Model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    train(args)

# para correr el entrenamiento segir el siguiente comando:
# uv run training/training.py --config config/models-configs/model_config_001.yaml
