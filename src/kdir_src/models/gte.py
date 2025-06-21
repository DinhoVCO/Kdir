from sentence_transformers import SentenceTransformer

def load_gte_large(device='cuda'):
    model = SentenceTransformer("thenlper/gte-large", device=device)
    embedding_dimension = model.get_sentence_embedding_dimension()
    return model, embedding_dimension


def get_gte_embeddings(model, sentences, batch_size=32, show_progress_bar=True, normalize_embeddings=True):
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=False
    )
    return embeddings