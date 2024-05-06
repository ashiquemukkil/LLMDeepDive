import os
import requests

from typing import Optional,List,Iterator,Any

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_URL = "https://sesame-panzanella-rzl32icxd728lczq.salad.cloud/" 
SALAD_HEADER ={"Salad-Api-Key":"c598a341-4139-4e45-92e6-2880fbd61425"}
MODEL = "llama2"

# def get_openai_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message.content


def get_llama_chat_completion(messages,model="llama2",**kwargs):
    try:
        data = {
            'model': model,
            'messages': messages,
            'stream': False
        }
        data.update(**kwargs)

        response = requests.post(OLLAMA_URL+"api/chat", json=data,headers=SALAD_HEADER
                                )
        if response.status_code == 200:
            return response.json().get("message")#.get("message")
        print(f"response ",response.content )
        return response.status_code
    except Exception as e:
        print(f"Error with llama hosting {e}")
        return False

def get_llama_completion(prompt,model="llama2",**kwargs):
    try:
        data = {
            'model': model,
            'prompt': prompt,
            'stream': False,
        }
        data.update(**kwargs)

        response = requests.post(OLLAMA_URL+"api/generate", json=data,headers=SALAD_HEADER
                                )
        if response.status_code == 200:
            return response.json().get("response")#.get("message")
        print(f"response ",response.content )
        return response.status_code
    except Exception as e:
        print(f"Error with llama hosting {e}")
        return False


class SaladChatOllama(ChatOllama):
    base_url: str = OLLAMA_URL.rstrip("/")
    headers:dict = SALAD_HEADER

class SaladOllamaEmbeddings(OllamaEmbeddings):
    base_url: str = OLLAMA_URL.rstrip("/")

    def _process_emb_response(self, input: str) -> List[float]:
        """Process a response from the API.

        Args:
            response: The response from the API.

        Returns:
            The response as a dictionary.
        """
        headers=SALAD_HEADER

        try:
            res = requests.post(
                f"{self.base_url}/api/embeddings",
                headers=headers,
                json={"model": self.model, "prompt": input, **self._default_params},
            )
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if res.status_code != 200:
            raise ValueError(
                "Error raised by inference API HTTP code: %s, %s"
                % (res.status_code, res.text)
            )
        try:
            t = res.json()
            return t["embedding"]
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised by inference API: {e}.\nResponse: {res.text}"
            )
    
        
    
def pull_llama2():
    data = {
        'model': "llama2",
    }
    response = requests.post(OLLAMA_URL+"api/pull", json=data,headers=SALAD_HEADER)
    if response.status_code == 200:
        return response.content
    print(response.content)
    return response.status_code

if __name__ == "__main__":
    print(get_llama_completion("hi"))
