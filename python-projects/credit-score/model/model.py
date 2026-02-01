import torch
import torch.nn as nn
import logging
from typing import Literal
import sys
import os
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.logs_configs.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting Model")


# --- 1. CONFIGURACIÓN DEL MODELO ---
@dataclass
class ModelConfig:
    """Clase para definir la arquitectura del modelo"""

    input_size: int
    hidden_layers: list[int]
    activation_functions: list[
        Literal["relu", "leaky_relu", "gelu", "sigmoid", "softmax", "tanh"]
    ]
    output_size: int = 1
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    checkpoint_path: str = "./model/checkpoint"


# --- 2. FACTORIA DE FUNCONES DE ACTIVACIÓN ---
def get_activation(function_name: str) -> nn.Module:
    """Retorna la función de activación"""
    if function_name == "relu":
        return nn.ReLU()
    elif function_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    elif function_name == "gelu":
        return nn.GELU()
    elif function_name == "sigmoid":
        return nn.Sigmoid()
    elif function_name == "softmax":
        return nn.Softmax(dim=1)
    elif function_name == "tanh":
        return nn.Tanh()
    else:
        msg = f"Función de activación no reconocida: {function_name}"
        logger.error(msg)
        raise ValueError(msg)


# --- 3. MODELO ---
class CreditScoreModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.input_size

        # Validar que la longitud de las capas ocultas sea igual a la longitud de las funciones de activación
        if len(config.hidden_layers) != len(config.activation_functions):
            msg = "La longitud de las capas ocultas debe ser igual a la longitud de las funciones de activación"
            logger.error(msg)
            raise ValueError(msg)

        # Construcción de capas ocultas
        for hidden_dim, act_fn in zip(
            config.hidden_layers, config.activation_functions
        ):
            layer = nn.Linear(in_dim, hidden_dim)
            self._init_layer_weights(layer, act_fn)
            layers.append(layer)
            # Normalización media 0 y desviación estándar 1. Estabiliza y acelera la convergencia
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Función de activación
            layers.append(get_activation(act_fn))
            # Regularización para evitar Overfitting
            layers.append(nn.Dropout(config.dropout_rate))
            in_dim = hidden_dim

        # Capa de salida
        output_layer = nn.Linear(in_dim, config.output_size)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        # Empaquetado de todas las capas
        self.model = nn.Sequential(*layers)

    def _init_layer_weights(self, layer, activation_name):
        """Aplica la inicialización óptima según la activación"""

        if activation_name in ["relu", "leaky_relu", "gelu"]:
            # He/Kaiming es mejor para variantes de ReLU
            nonlinearity = "relu" if activation_name == "gelu" else activation_name
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity=nonlinearity
            )
        elif activation_name in ["sigmoid", "tanh", "softmax"]:
            # Xavier/Glorot es mejor para funciones saturadas
            nn.init.xavier_normal_(layer.weight)
        # El bias siempre a cero es un buen estándar inicial
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            x (torch.Tensor): Tensor de entrada de tamaño (batch_size, input_size)
        Returns:
            torch.Tensor: Tensor de salida de tamaño (batch_size, output_size) de probabilidades
        """
        return self.model(x)

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Predicción de probabilidades
        Args:
            x (torch.Tensor): Tensor de entrada
        Returns:
            torch.Tensor: Tensor de probabilidades Good = 1, Bad = 0
        """
        with torch.no_grad():
            logits = self.forward(x)
            good_probability = torch.sigmoid(logits)
            bad_probability = 1 - good_probability
            return torch.cat([bad_probability, good_probability], dim=1)

    def binary_prediction(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Predicción binaria
        Args:
            x (torch.Tensor): Tensor de entrada
            threshold (float): Umbral de decisión
        Returns:
            torch.Tensor: Tensor de predicciones binarias Good = 1, Bad = 0
        """
        with torch.no_grad():
            logits = self.forward(x)
            good_probability = torch.sigmoid(logits)
            return (good_probability >= threshold).to(torch.int)

    def get_model_summary(self) -> str:
        """Obtiene un resumen del modelo"""
        return str(self.model)
