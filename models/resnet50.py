import numpy as np
from typing import Tuple
from keras import layers, models
from keras.applications import ResNet50

class Resnet50:
    def __init__(self, config: dict):
        self.input_shape: Tuple[int, int, int] = config["model"]["input_shape"]
        self.embedding_dim: int = config["model"]["embedding_dim"]
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model.trainable = False
        self.embedding_model = self._build_embedding_model()
        
        print(f"✅ ResNet50 Carregada. Input shape: {self.input_shape}")
        print(f" Embedding Dimensão Final: {self.embedding_model.output_shape[1]}")

    def _build_embedding_model(self):
        return models.Sequential([
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.embedding_dim, activation='relu', name='Embedding_Projection')
        ])

    def get_embeddings(self, images: np.ndarray) -> np.ndarray:
        return self.embedding_model.predict(images, batch_size=64, verbose=1)