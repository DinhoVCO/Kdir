from tqdm import tqdm
from kdir_src.utils.qdrant import recuperar_documentos, recuperar_documentos_sparsos
from kdir_src.models.all_models import load_models, get_query_embeddings
from kdir_src.dataset.beir import get_sentences_queries
import numpy as np
import torch
import json
import os

def format_result(rel_docs):
    docs_inference={}
    for doc in rel_docs:
        doc_id = doc.payload['doc_id']
        score = doc.score
        docs_inference[doc_id] = score
    return docs_inference

def get_results(dataset_name, path='../results/',batch_size=32, top_k=10):
    os.makedirs(path, exist_ok=True)
    models = load_models()
    sentences, queries_ids = get_sentences_queries(dataset_name)
    all_sentence_embeddings = get_query_embeddings(models, sentences, batch_size)
    collection_name= f"kdir_{dataset_name}"
    vectors =["contriever","contriever_ft","dpr","bge_large","gte_large","sparse_bm25"]
    i=0
    for model_embeddings in all_sentence_embeddings:
        results = get_inference_results(model_embeddings, queries_ids, collection_name, vectors[i], top_k)
        with open(path+f"results_{vectors[i]}.json", "w") as f:
            json.dump(results, f, indent=2)
        i+=1
    
def get_inference_results(model_embeddings, queries_ids, collection_name, vector_name, top_k=10):
    results = {}
    for i, query_embedding in tqdm(enumerate(model_embeddings), desc=f"obteniendo resultados {vector_name}"):
        processed_query_embedding = query_embedding 
        if isinstance(query_embedding, torch.Tensor):
            processed_query_embedding = query_embedding.tolist()
        elif isinstance(query_embedding, np.ndarray):
             processed_query_embedding = query_embedding.tolist()
        if(vector_name!="sparse_bm25"):
            rel_docs = recuperar_documentos(collection_name, processed_query_embedding, vector_name, top_k)
        else:
            rel_docs = recuperar_documentos_sparsos(collection_name, processed_query_embedding, vector_name, top_k)
        qid=queries_ids[i]
        results[qid] = format_result(rel_docs)
    return results