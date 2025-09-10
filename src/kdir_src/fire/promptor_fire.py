class Promptor_fire:
    def __init__(self, task: str):
        self.task = task
    
    def get_prompt(self, query: str, reference_docs: list):
        if self.task == 'scifact':
            return self.build_multi_prompts(query, reference_docs)
        elif self.task == 'arguana':
            return self.build_multi_prompts(query, reference_docs, )
        elif self.task == 'nfcorpus':
            return self.build_multi_prompts(query, reference_docs, )
        elif self.task == 'fiqa':
            return self.build_multi_prompts(query, reference_docs, )
        elif self.task == 'scidocs':
            return self.build_multi_prompts(query, reference_docs, )
        else:
            raise ValueError('Task not supported')

    def build_multi_prompts(self, query, reference_docs):
        pseudo_doc = self.build_prompt(query, reference_docs, BASE_DOC_PROMPT, prompt_doc_names)
        pseudo_query = self.build_prompt(query, reference_docs, BASE_QUERY_PROMPT, prompt_query_names)
        pseudo_answer = self.build_prompt(query, reference_docs, BASE_QA_PROMPT, prompt_qa_names)
        return [pseudo_doc, pseudo_query, pseudo_answer]

    
    def build_prompt(self, query: str, reference_docs: list, PROMPT_TEMPLATE, prompt_names) -> str:
        def formatear_doc(arg):
            new_doc = f"{arg.payload['title']}\n\n{arg.payload['text']}\n\n"
            return new_doc
        task_prompt_names=prompt_names[self.task]
        if reference_docs:
            formatted_candidates = "\n".join(
                [f"{i+1}. {formatear_doc(arg)}" for i, arg in enumerate(reference_docs)]
            )
        else:
            formatted_candidates = "None provided."
        final_prompt = PROMPT_TEMPLATE.format(
            query=query,
            reference_docs=formatted_candidates,
            task = task_prompt_names['task'],
            query_name =task_prompt_names['query_name'],
            source_doc_name =task_prompt_names['source_doc_name'],
            gen_doc_name =task_prompt_names['gen_doc_name'],
        )
        return final_prompt

        
## Document
###################

prompt_doc_names = {
    'arguana':{
        'task': "Write a compelling and well-founded counter-argument for the following argument.",
        'query_name': 'Argument',
        'source_doc_name': 'Reference Arguments',
        'gen_doc_name': 'Counter-Argument'
    },
    'nfcorpus':{
        'task': " Write a single, new, and comprehensive medical article passage that provides a definitive answer to the user's query.",
        'query_name': 'Query',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Document'
    },
    'scifact':{
        'task': " Write a single, new, and comprehensive scientific paper passage to support/refute the claim.",
        'query_name': 'Claim',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Document'
    },
    'fiqa':{
        'task': " Write a single, new, and comprehensive financial article passage to answer the question.",
        'query_name': 'Question',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Document'
    },
    'scidocs':{
        'task': " Write a single, new, and comprehensive scientific document passage on the topic.",
        'query_name': 'Topic',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Document'
    },
}

BASE_DOC_PROMPT = """
Task: {task}

- Use the '{source_doc_name}' provided below as inspiration or incorporate their ideas if they are relevant and helpful.
- **Closely match the style of the '{source_doc_name}'. This includes their approximate length (e.g., a single concise sentence), tone, and overall structure.**
- The {gen_doc_name} should be logical and maintain a respectful tone.
- Your response must ONLY be the text of the final {gen_doc_name}, with no introduction or explanation.

{query_name}: {query}

{source_doc_name}:
{reference_docs}

{gen_doc_name}:

"""
## Query
###################


prompt_query_names = {
    'arguana': {
        'task': "Generate a new, insightful Argument using the reference arguments for context.",
        'query_name': 'Original Argument',
        'source_doc_name': 'Reference Arguments',
        'gen_doc_name': 'Generated Argument'
    },
    'nfcorpus': {
        'task': "Based on the user's query and the provided medical documents, generate a more specific or a follow-up medical query.",
        'query_name': 'Original Query',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Query'
    },
    'scifact': {
        'task': "Given the scientific claim and source documents, formulate a new, verifiable query that could help further validate or challenge the claim.",
        'query_name': 'Original Claim',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Query'
    },
    'fiqa': {
        'task': "From the financial question and source articles, create a new, more detailed question that delves deeper into the topic.",
        'query_name': 'Original Question',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Query'
    },
    'scidocs': {
        'task': "Using the topic and source scientific documents, generate a new, more focused research query or topic.",
        'query_name': 'Original Topic',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Generated Query'
    },
}

BASE_QUERY_PROMPT = """
Task: {task}

- Use the '{source_doc_name}' provided below for context and inspiration.
- The new query should be logical, well-formed, and directly related to the original inputs.
- **Your response must ONLY be the text of the final {gen_doc_name}, with no introduction or explanation.**

{query_name}: {query}

{source_doc_name}:
{reference_docs}

{gen_doc_name}:

"""


## Anwer
###################

prompt_qa_names = {
    'arguana': {
        'task': "Extract the core claim and the key premises or evidence used to support it from the Argument",
        'query_name': 'Argument',
        'source_doc_name': 'Reference Arguments',
        'gen_doc_name': 'Claim'
    },
    'nfcorpus': {
        'task': "Provide a direct and definitive answer to the user's query.",
        'query_name': 'Query',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Answer'
    },
    'scifact': {
        'task': "Directly support or refute the claim with a clear explanation.",
        'query_name': 'Claim',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Explanation'
    },
    'fiqa': {
        'task': "Provide a direct and concise answer to the question.",
        'query_name': 'Question',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Answer'
    },
    'scidocs': {
        'task': "Provide a direct and concise summary on the topic.",
        'query_name': 'Topic',
        'source_doc_name': 'Source Documents',
        'gen_doc_name': 'Summary'
    },
}

BASE_QA_PROMPT = """
Task: {task}

- The documents '{source_doc_name}' are provided for your reference. Be aware that they might not hold all the relevant information to resolve the task.
- The {gen_doc_name} should be logical and maintain a respectful tone.
- Your response must ONLY be the text of the final {gen_doc_name}, with no introduction or explanation.

{query_name}: {query}

{source_doc_name}:
{reference_docs}

{gen_doc_name}:

"""

        