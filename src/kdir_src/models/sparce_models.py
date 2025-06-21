from fastembed import SparseTextEmbedding, SparseEmbedding

def load_bm25():
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return model
    
def load_splade():
    model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return model

def get_sparse_embeddings(model, sentences, batch_size=32):
    sparse_embeddings_list: list[SparseEmbedding] = list(
        model.embed(
            sentences, 
            batch_size=batch_size
        )
    ) 
    return sparse_embeddings_list