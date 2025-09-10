import os
import json
import openai
import time

SSL_CERT_FILE2 = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = ""
OPENAI_API_VERSION = "2024-12-01-preview" #"2023-08-01-preview"
os.environ['SSL_CERT_FILE'] = SSL_CERT_FILE2
client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)

class Generator:
    def __init__(self, model_name):
        self.model_name = model_name

class LiteLLMGenerator(Generator):
    def __init__(self, model_name, n=1, max_tokens=512, temperature=0.7, top_p=1, random_seed=None, safe_prompt=False, stop=None, wait_till_success=False):
        super().__init__(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        self.top_p = top_p
        self.random_seed = random_seed
        self.safe_prompt = safe_prompt
        self.stop = stop if stop is not None else []
        self.wait_till_success = wait_till_success
        self._client_init()

    @staticmethod
    def parse_response(response):
        to_return = []
        for choice in response.choices:
            text = choice.message.content
            to_return.append(text)
        return to_return

    def _client_init(self):
        self.client = client

    def generate(self, prompt):
        #"gpt-4o-petrobras", "gpt-35-turbo-petrobras"
        get_results = False
        while not get_results:
            try:
                result = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    #n=self.n,
                    #top_p=self.top_p,
                )
                #respuesta = chat_completion.choices[0].message.content.strip()
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    print(f"Error during generation, retrying in 1 second: {e}")
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)