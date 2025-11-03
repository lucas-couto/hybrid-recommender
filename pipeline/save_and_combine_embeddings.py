import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def save_and_combine_embeddings(
    embeddings_dict: Dict[str, np.ndarray],
    image_ids: np.ndarray,
    config: Dict[str, Any]
) -> None:
    
    all_embeddings_to_combine = []
    output_dir = Path("outputs/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    hybrid_models: List[str] = config["model"]["hybrid_models"]
    
    print(f"\n--- Processando e Salvando Embeddings ---")

    for model_name, embedding_array in embeddings_dict.items():
        output_file_individual = output_dir / f"{model_name.lower()}.npz"
        
        np.savez_compressed(
            output_file_individual,
            image_ids=image_ids,
            embeddings=embedding_array
        )
        print(f"  💾 Embeddings de {model_name} salvos em: {output_file_individual.name}")
        
        if model_name in hybrid_models:
            all_embeddings_to_combine.append(embedding_array)
            print(f"  ➕ {model_name} adicionado à combinação híbrida.")

    # 3. Concatenar e Salvar o Embedding Híbrido
    if all_embeddings_to_combine:
        print("\n--- Gerando Embedding Híbrido ---")
        
        # Concatenação dos arrays ao longo do eixo de features (axis=1)
        combined_embeddings = np.concatenate(all_embeddings_to_combine, axis=1)
        
        hybrid_file_name = config["hybrid"].get("output_file", "hybrid_combined_embeddings.npz")
        output_file_hybrid = output_dir / hybrid_file_name
        
        np.savez_compressed(
            output_file_hybrid,
            image_ids=image_ids,
            embeddings=combined_embeddings.astype(np.float32)
        )
        
        initial_dims = [arr.shape[1] for arr in all_embeddings_to_combine]
        print(f"  ✅ Combinação concluída. Dimensões: {initial_dims} -> {combined_embeddings.shape[1]}")
        print(f"  💾 Embedding Híbrido salvo em: {output_file_hybrid.name}")
    else:
        print("\n⚠️ Nenhum modelo especificado para combinação híbrida. Apenas salvamento individual concluído.")
