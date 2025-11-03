from utils import get_config
from pipeline import (
  browse_all_images, 
  preprocess_images, 
  get_embeddings_models, 
  load_image_data,
  save_and_combine_embeddings
)

def main():
  config = get_config()
  inputs_shape = config["model"]["input_shape"]
  browse_all_images("dataset/images.csv")
  preprocess_images("images", size=(inputs_shape[0], inputs_shape[1]))
  data = load_image_data("outputs/images.npz")
  embeddings = get_embeddings_models(config, data=data["images"])
  save_and_combine_embeddings(
    embeddings_dict=embeddings,
    image_ids=data["image_ids"],
    config=config
  )
  
if __name__ == "__main__":
  main()