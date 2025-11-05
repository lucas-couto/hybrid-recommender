import os
import pandas as pd
from utils import download_image

def browse_all_images(csv_path: str) -> None:
    if not os.path.exists(csv_path): 
        raise ValueError(f"The CSV file {csv_path} does not exist.")

    df = pd.read_csv(csv_path)

    if 'image_url' not in df.columns:
        raise ValueError("The CSV file must contain an 'image_url' column'.")

    for row in df.itertuples(index=False):
        image_url = getattr(row, 'image_url', None)
        item_id = getattr(row, 'item_id', None)

        dest_dir = os.path.join('images')

        try:
            download_image(item_id, image_url, dest_dir)
        except Exception as e:
            print(f"Falha ao baixar {image_url} para {dest_dir}: {e}")