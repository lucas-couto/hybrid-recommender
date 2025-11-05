from models import Resnet50, ViT
import tensorflow as tf
import gc

def get_embeddings_models(config, data):
  models = [Resnet50]
  embeddings = {}

  for model in models:
    model = model(config)
    embeddings[model.__class__.__name__] = model.get_embeddings(data)
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()

  return embeddings