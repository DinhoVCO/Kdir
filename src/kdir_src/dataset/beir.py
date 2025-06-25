from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm
import random
import os

def get_beir_dataset(dataset_name):
    dataset = dataset_name
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "../datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def get_beir_train_dataset(dataset_name):
    dataset = dataset_name
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "../datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    return corpus, queries, qrels


def get_sentences_queries_arguana(queries):
    queries_ids=[]
    sentences = []
    all_query_items = list(queries.items())
    for query_id, query in tqdm(all_query_items, desc="Procesando queries"):
        queries_ids.append(query_id)
        sentences.append(query)
    return sentences, queries_ids


def get_sentences_corpus_arguana(corpus):
    sentences = []
    payloads = []
    all_doc_items = list(corpus.items())
    for doc_id, contenido in tqdm(all_doc_items, desc="Procesando corpus"):
        payload = {
            'doc_id': doc_id,
            'title': contenido['title'],
            'text': contenido['text']
        }
        payloads.append(payload)
        sentences.append(contenido['text'])
    return sentences, payloads


def get_sentences_corpus(dataset_name):
    corpus, queries, qrels = get_beir_dataset(dataset_name)
    if(dataset_name=='arguana'):
        return get_sentences_corpus_arguana(corpus)
    return get_sentences_corpus_arguana(corpus)

def get_sentences_queries(dataset_name):
    corpus, queries, qrels = get_beir_dataset(dataset_name)
    if(dataset_name=='arguana'):
        return get_sentences_queries_arguana(queries)
    if(dataset_name=='nfcorpus'):
        return get_sentences_queries_arguana(queries)
    return get_sentences_queries_arguana(queries)

def build_beir_random_examples(corpus, queries, qrels, num_examples=4, seed=None):
    """
    Construye ejemplos aleatorios de Query y Passage a partir de los datos de BEIR.
    Para cada consulta, toma solo el primer documento relevante.

    Args:
        corpus (dict): Diccionario de documentos.
        queries (dict): Diccionario de consultas.
        qrels (dict): Diccionario de relevancias.
        num_examples (int): El número de ejemplos que se desea generar.
        seed (int, optional): Semilla para la reproducibilidad de la aleatoriedad.
                              Por defecto es None (sin semilla).

    Returns:
        list: Una lista de diccionarios, donde cada diccionario contiene
              'query' y 'passage' relevantes.
    """
    if seed is not None:
        random.seed(seed)

    examples = []
    
    # Obtener una lista de todos los query_ids que tienen al menos un documento relevante
    candidate_qids = [qid for qid, rel_docs in qrels.items() if any(score > 0 for score in rel_docs.values())]
    
    # Si hay menos consultas candidatas que el número de ejemplos deseados,
    # tomamos todas las consultas candidatas.
    num_to_sample = min(num_examples, len(candidate_qids))
    
    # Seleccionar aleatoriamente N query_ids
    # Usamos random.sample para obtener una muestra sin reemplazo
    selected_qids = random.sample(candidate_qids, num_to_sample)
    
    for qid in selected_qids:
        query_text = queries.get(qid)
        
        if query_text:
            # Encontrar el primer documento relevante para esta consulta
            first_relevant_doc_id = None
            for doc_id, score in qrels[qid].items():
                if score > 0:
                    first_relevant_doc_id = doc_id
                    break # Tomar solo el primero y salir

            if first_relevant_doc_id:
                passage_data = corpus.get(first_relevant_doc_id)
                if passage_data:
                    passage_text = ""
                    if "title" in passage_data and passage_data["title"]:
                        passage_text += passage_data["title"] + " "
                    passage_text += passage_data["text"]
                    
                    examples.append({
                        "query": query_text,
                        "passage": passage_text
                    })
    return examples

