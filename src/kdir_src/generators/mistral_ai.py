import time
from mistralai import Mistral

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

class MistralGenerator(Generator):
    def __init__(self, model_name, api_key, n=1, max_tokens=512, temperature=0.7, top_p=1, random_seed=None, safe_prompt=False, stop=None, wait_till_success=False):
        super().__init__(model_name, api_key)
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
        self.client = Mistral(api_key=self.api_key)

    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = self.client.chat.complete(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }, 
                    ],
                    max_tokens=self.max_tokens,
                    n=self.n,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    random_seed=self.random_seed,
                    safe_prompt=self.safe_prompt,
                    stop=self.stop if self.stop else [],
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    print(f"Error during generation, retrying in 1 second: {e}")
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)