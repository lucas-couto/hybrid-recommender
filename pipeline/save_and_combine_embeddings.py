import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def save_and_combine_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    items_id: np.ndarray,
    config: Dict[str, Any]
) -> None:
    
    all_embeddings_to_combine = []
    output_dir = Path("outputs/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_models: List[str] = config["model"]["hybrid_models"]
    
    print(f"\n--- Processando e Salvando Embeddings (em .npy) ---")

    for model_name, embedding_array in embeddings_dict.items():
        output_file_embeddings = output_dir / f"{model_name.lower()}_embeddings.npy"
        np.save(output_file_embeddings, embedding_array)
        print(f"  💾 Embeddings de {model_name} salvos em: {output_file_embeddings.name}")
        
        output_file_ids = output_dir / f"{model_name.lower()}_items_id.npy"
        np.save(output_file_ids, items_id)
        print(f"  🆔 IDs de {model_name} salvos em: {output_file_ids.name}")

        if model_name in hybrid_models:
            all_embeddings_to_combine.append(embedding_array)
            print(f"  ➕ {model_name} adicionado à combinação híbrida.")

    # 3. Concatenar e Salvar o Embedding Híbrido (em .npy)
    if all_embeddings_to_combine:
        print("\n--- Gerando e Salvando Embedding Híbrido (em .npy) ---")
        
        # Concatenação dos arrays ao longo do eixo de features (axis=1)
        combined_embeddings = np.concatenate(all_embeddings_to_combine, axis=1)
        
        hybrid_file_base_name = config["hybrid"].get("output_file_base", "hybrid_combined")
        
        # Salvar o Embedding Híbrido
        output_file_hybrid_embeddings = output_dir / f"{hybrid_file_base_name}_embeddings.npy"
        # Mantendo o astype(np.float32) para consistência e eficiência de armazenamento/memória
        np.save(output_file_hybrid_embeddings, combined_embeddings.astype(np.float32))
        
        # Salvar os Image IDs Híbridos (será o mesmo array de IDs)
        output_file_hybrid_ids = output_dir / f"{hybrid_file_base_name}_items_id.npy"
        np.save(output_file_hybrid_ids, items_id)

        initial_dims = [arr.shape[1] for arr in all_embeddings_to_combine]
        print(f"  ✅ Combinação concluída. Dimensões: {initial_dims} -> {combined_embeddings.shape[1]}")
        print(f"  💾 Embedding Híbrido salvo em: {output_file_hybrid_embeddings.name}")
        print(f"  🆔 IDs Híbridos salvos em: {output_file_hybrid_ids.name}")
    else:
        print("\n⚠️ Nenhum modelo especificado para combinação híbrida. Apenas salvamento individual concluído.")