import os
import requests

def download_image(id: str, url: str, dest_folder: str) -> None:
  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

  try:
    response = requests.get(url)
    response.raise_for_status()

    image_name = os.path.basename(id + "." + url.split(".")[-1])
    image_path = os.path.join(dest_folder, image_name)

    with open(image_path, 'wb') as f:
        f.write(response.content)

    print(f"Image downloaded and saved to {image_path}")

  except requests.exceptions.RequestException as e:
    raise ValueError(f"Failed to download image from {url}. Error: {e}")