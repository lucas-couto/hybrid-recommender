import os
import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, List
from PIL import UnidentifiedImageError
from keras.preprocessing.image import load_img

warnings.filterwarnings('ignore', category=UserWarning, module='PIL')

def preprocess_images(
    src_dir: str = "images",
    output_file: str = "outputs/items.npz",
    size: Tuple[int, int] = (128, 128),
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> None:
    print(f"Iniciando pré-processamento de {src_dir} -> {output_file}...")

    src = Path(src_dir)
    dst_path = Path(output_file)
    if not src.exists():
        raise ValueError(f"Diretório fonte não existe: {src}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Configuração de Dtype: float16 para valores normalizados (0.0 - 1.0)
    np_dtype = np.float16
    initial_dtype = np.uint8

    # Buffers
    imgs: List[np.ndarray] = []
    # Agora armazenará apenas a parte do ID (sem extensão)
    items_id: List[str] = [] 

    total_processed = 0
    skipped_count = 0

    for root, _, files in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)

        for fname in files:
            if fname.startswith(".") or not fname.lower().endswith(extensions):
                continue

            src_path = root_path / fname
            
            # Caminho relativo completo (e.g., subpasta/B07DJ2XG61.jpg)
            rel_path = rel_root / fname 

            # ===== EXTRAÇÃO DO ID DA IMAGEM =====
            # 1. Usa Path(rel_path).name para obter apenas o nome do arquivo (B07DJ2XG61.jpg)
            # 2. Usa .stem para remover a extensão (.jpg)
            image_id = Path(rel_path.name).stem # Ex: 'B07DJ2XG61'

            try:
                # 1. Carregamento e Redimensionamento
                pil_img = load_img(src_path, target_size=size, color_mode="rgb")
                arr = np.asarray(pil_img, dtype=initial_dtype)

                # 2. Normalização Incondicional
                arr = (arr.astype(np.float32) / 255.0).astype(np_dtype)

                # 3. Adicionar aos buffers
                imgs.append(arr)
                # SALVA APENAS O ID EXTRAÍDO
                items_id.append(image_id) 
                total_processed += 1
                
                # Feedback de progresso
                if total_processed % 1000 == 0:
                    print(f"Processadas {total_processed} imagens...")

            except (UnidentifiedImageError, OSError) as e:
                skipped_count += 1
                print(f"[skip] Imagem inválida/erro de IO: {src_path}: {e}")
            except Exception as e:
                skipped_count += 1
                print(f"[skip] Erro inesperado: {src_path}: {e}")

    if not imgs:
        print("\n❌ Nenhuma imagem válida encontrada para processamento.")
        return

    # 4. Empilhar e Salvar no NPZ
    print("\nEmpilhando arrays...")
    images_array = np.stack(imgs, axis=0)
    ids_array = np.array(items_id, dtype=object) # IDs como array de strings/object

    # Salvamento
    np.savez_compressed(
        dst_path,
        images=images_array,
        items_id=ids_array
    )

    print("\n✅ Concluído.")
    print(f"Total salvas: {total_processed} | Puladas: {skipped_count}")
    print(f"Shape final (Imagens): {images_array.shape} | Dtype: {images_array.dtype}")
    print(f"Arquivo NPZ salvo: {dst_path.resolve()}")