from kdir_src.models.bge import load_bge_large, get_bge_large_embeddings 
from kdir_src.models.contriever import load_contriever, load_contriever_ft, get_contriever_embeddings_batched
from kdir_src.models.dpr import load_question_dpr, load_context_dpr, get_dpr_embeddings_batched
from kdir_src.models.sparce_models import load_bm25, load_splade, get_sparse_embeddings
from kdir_src.models.qwen3 import load_qwen3_06B, get_qwen3_embeddings
from kdir_src.models.gte import load_gte_large, get_gte_embeddings

def load_models():
    print('Loading contriever...')
    contriever = load_contriever()
    print('Loading contriever FT...')
    contriever_ft = load_contriever_ft()
    print('Loading question dpr...')
    question_dpr = load_question_dpr()
    print('Loading context dpr...')
    context_dpr = load_context_dpr()
    print('Loading bge large...')
    bge_large = load_bge_large()
    #print('Loading qwen3 06B...')
    #qwen3_06B = load_qwen3_06B()
    print('Loading gte large...')
    gte_large = load_gte_large()
    print('Loading bm25...')
    bm25 = load_bm25()
    #print('Loading splade...')
    #splade = load_splade()
    models = {
        'contriever': contriever,
        'contriever_ft': contriever_ft,
        'question_dpr': question_dpr,
        'context_dpr': context_dpr,
        'bge_large': bge_large,
        #'qwen3_06B': qwen3_06B,
        'gte_large': gte_large,
        'bm25': bm25,
        #'splade': splade
    }
    return models

def get_docs_embeddings(models, sentences, batch_size=32):
    embeddings=[]
    contriever_embeddings = get_contriever_embeddings_batched(models['contriever'][0],models['contriever'][1], sentences, batch_size)
    embeddings.append(contriever_embeddings)
    contriever_ft_embeddings = get_contriever_embeddings_batched(models['contriever_ft'][0],models['contriever_ft'][1], sentences, batch_size)
    embeddings.append(contriever_ft_embeddings)
    dpr_embeddings = get_dpr_embeddings_batched(models['context_dpr'][0],models['context_dpr'][1], sentences, batch_size)
    embeddings.append(dpr_embeddings)
    bge_embeddings = get_bge_large_embeddings(models['bge_large'][0], sentences, batch_size)
    embeddings.append(bge_embeddings)
    #qwen3_06B_embeddings = get_qwen3_embeddings(models['qwen3_06B'][0], sentences, 64)
    #embeddings.append(qwen3_06B_embeddings)
    gte_large_embeddings = get_gte_embeddings(models['gte_large'][0], sentences, batch_size)
    embeddings.append(gte_large_embeddings)
    print("Procesando BM25 embeddings")
    bm25_embeddings = get_sparse_embeddings(models['bm25'], sentences, 128)
    embeddings.append(bm25_embeddings)
    #print("Procesando SPLADE embeddings")
    #splade_embeddings = get_sparse_embeddings(models['splade'], sentences, 128)
    #embeddings.append(splade_embeddings)
    print("Embeddings finalizado")
    return embeddings


def get_query_embeddings(models, sentences, batch_size=32, show_progress_bar=True):
    embeddings=[]
    contriever_embeddings = get_contriever_embeddings_batched(models['contriever'][0],models['contriever'][1], sentences, batch_size, show_progress_bar)
    embeddings.append(contriever_embeddings)
    contriever_ft_embeddings = get_contriever_embeddings_batched(models['contriever_ft'][0],models['contriever_ft'][1], sentences, batch_size, show_progress_bar)
    embeddings.append(contriever_ft_embeddings)
    dpr_embeddings = get_dpr_embeddings_batched(models['question_dpr'][0],models['question_dpr'][1], sentences, batch_size, show_progress_bar)
    embeddings.append(dpr_embeddings)
    bge_embeddings = get_bge_large_embeddings(models['bge_large'][0], sentences, batch_size, show_progress_bar)
    embeddings.append(bge_embeddings)
    #qwen3_06B_embeddings = get_qwen3_embeddings(models['qwen3_06B'][0], sentences, 64)
    #embeddings.append(qwen3_06B_embeddings)
    gte_large_embeddings = get_gte_embeddings(models['gte_large'][0], sentences, batch_size, show_progress_bar)
    embeddings.append(gte_large_embeddings)
    if(show_progress_bar):
        print("Procesando BM25 embeddings")
    bm25_embeddings = get_sparse_embeddings(models['bm25'], sentences, 128)
    embeddings.append(bm25_embeddings)
    #print("Procesando SPLADE embeddings")
    #splade_embeddings = get_sparse_embeddings(models['splade'], sentences, 128)
    #embeddings.append(splade_embeddings)
    if(show_progress_bar):
        print("Embeddings finalizado")
    return embeddings