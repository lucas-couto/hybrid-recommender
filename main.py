from utils import get_config
from pipeline import (
  browse_all_images, 
  preprocess_images, 
  get_embeddings_models, 
  load_image_data,
  save_and_combine_embeddings,
  load_elliot_data,
  align_image_embeddings,
)

def main():
  config = get_config()
  inputs_shape = config["model"]["input_shape"]
  browse_all_images("dataset/items.csv")
  preprocess_images("images", size=(inputs_shape[0], inputs_shape[1]))
  data = load_image_data("outputs/items.npz")
  
  embeddings = get_embeddings_models(config, data=data["images"])
  save_and_combine_embeddings(
    embeddings_dict=embeddings,
    items_id=data["items_id"],
    config=config
  )

  item_order, items_id, embeddings = load_elliot_data()
  align_image_embeddings(
    item_order=item_order,
    items_id=items_id,
    embeddings=embeddings,
  )
  
if __name__ == "__main__":
  main()