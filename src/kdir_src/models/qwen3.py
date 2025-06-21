from sentence_transformers import SentenceTransformer

def load_qwen3_06B(device='cuda'):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
    embedding_dimension = model.get_sentence_embedding_dimension()
    return model, embedding_dimension


def get_qwen3_embeddings(model, sentences, batch_size=32, normalize_embeddings=True):
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
        convert_to_tensor=False
    )
    return embeddings