import numpy as np
from typing import Dict, Any

def load_image_data(npz_file_path: str = "images.npz") -> Dict[str, Any]:
    try:
        print(f"Iniciando carregamento do arquivo: {npz_file_path}")
        
        data = np.load(npz_file_path, allow_pickle=True)
        
        images_array = data['images']
        ids_array = data['items_id']
        
        print(f"✅ Carregamento concluído. Imagens: {images_array.shape}, IDs: {ids_array.shape}")
        print(f"   Memória (aprox.): {images_array.nbytes / (1024**3):.2f} GB (apenas imagens)")
        
        return {
            'images': images_array,
            'items_id': ids_array
        }
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo NPZ não encontrado em: {npz_file_path}")
        return {'images': np.array([]), 'items_id': np.array([])}
    except Exception as e:
        print(f"❌ Erro inesperado ao carregar NPZ: {e}")
        return {'images': np.array([]), 'items_id': np.array([])}