import numpy as np
import pandas as pd

def load_elliot_data() -> None:
  embeddings = np.load("outputs/embeddings/resnet50_embeddings.npy", allow_pickle=True)
  items_id = np.load("outputs/embeddings/resnet50_items_id.npy", allow_pickle=True)
  
  df_interactions = pd.read_csv("dataset/interactions.csv")

  item_order = df_interactions['item_id'].unique()
  print(f"Número de itens no Elliot: {len(item_order)}")

  return item_order, items_id, embeddings