import numpy as np
from kdir_src.models.all_models import get_query_embeddings
from qdrant_client.models import SparseVector
from kdir_src.utils.qdrant import recuperar_documentos, recuperar_documentos_sparsos
from kdir_src.dataset.beir import get_sentences_queries
from kdir_src.utils.inference import get_inference_results
from kdir_src.dataset.beir import get_sentences_queries, build_beir_random_examples, get_beir_train_dataset
from tqdm import tqdm
import os
import json


def generar_prompt_few_shot(ejemplos, consulta_final):
    """
    Crea un prompt few-shot a partir de ejemplos dados y una consulta final.

    Args:
        ejemplos (list of dict): Una lista de diccionarios, donde cada diccionario
                                 tiene las claves 'query' y 'passage'.
        consulta_final (str): La consulta final que quieres que el modelo responda.

    Returns:
        str: El prompt few-shot formateado.
    """
    partes_prompt = []
    for ejemplo in ejemplos:
        partes_prompt.append(f"Query: {ejemplo['query']}\nPassage: {ejemplo['passage']}\n\n")

    # Agregamos la instrucción inicial si no está ya implícita en los ejemplos
    prompt_completo = "Write a passage that answers the given query: \n\n" + "".join(partes_prompt)

    # Agregamos la consulta final esperando una respuesta
    prompt_completo += f"Query: {consulta_final}\nPassage:"

    return prompt_completo




class Query2Doc:
    def __init__(self, datset_name, generator, encoder_models):
        self.generator = generator
        self.encoder_models = encoder_models
        self.dataset_name = datset_name

    def generate(self, query, max_retries=5, retry_delay_seconds=1):
        """
        Generates hypothetical documents for a given query, with retry logic for API errors.

        Args:
            query (str): The input query.
            max_retries (int): Maximum number of retries if the API call fails.
            retry_delay_seconds (int): Delay in seconds between retries.

        Returns:
            list: A list of generated documents (strings).
            
        Raises:
            Exception: If generation fails after all retries.
        """        
        for attempt in range(max_retries):
            try:
                # Llama a tu modelo generativo (LLM real) aquí
                # Asegúrate de que self.generator.generate(prompt) devuelva una lista de strings.
                hypothesis_documents = self.generator.generate(query) 
                # print(f"  --> Attempt {attempt + 1}: Generated for query: '{query}'") # Opcional para depuración
                return hypothesis_documents
            except Exception as e: # Captura excepciones generales. Puedes ser más específico si conoces los errores de tu API.
                print(f"Error generating for query '{query}' (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay_seconds} seconds...")
                    time.sleep(retry_delay_seconds)
                else:
                    print(f"Max retries reached for query '{query}'. Giving up.")
                    raise # Vuelve a lanzar la excepción si se agotaron los reintentos


    def get_and_save_passages(self, path='../doc_gen/query2doc/'):
        os.makedirs(path, exist_ok=True)
        corpus_train, queries_train, qrels_train =get_beir_train_dataset(self.dataset_name)
        sentences, queries_ids = get_sentences_queries(self.dataset_name)
        output_filepath = os.path.join(path, f"generated_documents_{self.dataset_name}.jsonl")
         # --- Step 1: Read existing generated query IDs ---
        existing_query_ids = set()
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    try:
                        entry = json.loads(line)
                        if "query_id" in entry:
                            existing_query_ids.add(entry["query_id"])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {output_filepath}: {line.strip()}")
                        continue
        
        print(f"Found {len(existing_query_ids)} previously generated documents in '{output_filepath}'.")

        # --- Step 2: Open file in append mode and iterate through queries ---
        # Open in 'a' (append) mode so new lines are added without overwriting
        # If the file didn't exist, 'a' will create it.
        with open(output_filepath, 'a', encoding='utf-8') as f_write:
            for i, query in tqdm(enumerate(sentences), desc='Processing queries (generate or skip)'):
                current_query_id = queries_ids[i]

                if current_query_id in existing_query_ids:
                    # If the query ID already exists, skip generation
                    # print(f"Skipping query ID '{current_query_id}' as it's already generated.")
                    continue

                examples = build_beir_random_examples(corpus_train, queries_train, qrels_train, 4)
                prompt_final = generar_prompt_few_shot(examples, query)
                
                # If not generated, proceed with generation
                hypothesis_documents = self.generate(prompt_final)
                
                # Create the JSON object for the current line
                entry = {
                    "query_id": current_query_id,
                    "query": query,
                    "generated_documents": hypothesis_documents
                }
                
                # Convert the object to a JSON string and write the line, followed by a newline
                f_write.write(json.dumps(entry, ensure_ascii=False) + '\n')
                # The write is performed immediately after generating each set of documents.
        
        print(f"Document generation complete. Results saved in: {output_filepath}")
            
            
            
            
    
   
        
            
        
        