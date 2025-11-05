import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

def align_image_embeddings(
    item_order: np.ndarray,
    items_id: np.ndarray,
    embeddings: np.ndarray,
    output_path: Union[str, Path] = "dataset/visual_features.npy"
) -> None:
    output_path = Path(output_path)
    print(f"\n--- Iniciando Alinhamento e Reordenação de Embeddings ---")
    
    try:
        df_raw = pd.DataFrame({
            'item_id': items_id,
            'embedding': list(embeddings) 
        }).set_index('item_id')

    except Exception as e:
        print(f"❌ ERRO ao criar o DataFrame de mapeamento: {e}")
        return

    df_sorted = df_raw.reindex(item_order)
    missing_items = df_sorted['embedding'].isna().sum() 

    if missing_items > 0:
        print(f"⚠️ Alerta: {missing_items} itens presentes no interactions.csv não têm embeddings gerados.")
        print("Estes itens serão representados por vetores de zeros (preenchimento padrão).")
        
        embedding_dim = embeddings.shape[1]
        zero_vector = np.zeros(embedding_dim, dtype=np.float32)
        
        zero_vector_list = [zero_vector] * missing_items
        
        nan_indices = df_sorted[df_sorted['embedding'].isna()].index
        
        df_sorted.loc[nan_indices, 'embedding'] = zero_vector_list

    embeddings_final_aligned = np.array(df_sorted['embedding'].tolist())
    np.save(output_path, embeddings_final_aligned.astype(np.float32))

    print(f"  ✅ Reordenação concluída. Shape final: {embeddings_final_aligned.shape}")
    print(f"  💾 Embeddings alinhados com a ordem do ELLIOT salvos em: {output_path.name}")