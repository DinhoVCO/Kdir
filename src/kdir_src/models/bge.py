import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_bge_large(device='cuda'):
    """
    Carga el modelo BGE-large y su dimensión de embedding.
    """
    model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    embedding_dimension = model.get_sentence_embedding_dimension()
    return model, embedding_dimension

def get_bge_large_embeddings(model, sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True):
    """
    Obtiene los embeddings de las oraciones en batches.

    Args:
        model: El modelo SentenceTransformer cargado.
        sentences: Una lista de cadenas de texto (oraciones).
        batch_size: El número de oraciones a procesar en cada batch.
        normalize_embeddings: Si normalizar los embeddings a longitud 1.
                              Recomendado para la mayoría de las tareas de similitud.
        show_progress_bar: Si mostrar la barra de progreso de tqdm.
    Returns:
        Un numpy array de embeddings.
    """
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=False
    )
    return embeddings