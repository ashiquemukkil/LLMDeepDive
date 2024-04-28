import os
import openai
from openai import OpenAI
import requests

OLLAMA_URL = "https://sesame-panzanella-rzl32icxd728lczq.salad.cloud/" 
SALAD_HEADER ={"Salad-Api-Key":"c598a341-4139-4e45-92e6-2880fbd61425"}

os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def get_openai_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def get_llama_completion(prompt,model="llama2"):
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
    }
    response = requests.post(OLLAMA_URL, json=data)
    if response.status_code == 200:
        return response.json().get("response","--ERROR--")
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
        return response.status_code
    except Exception as e:
        print(f"Error with llama hosting {e}")
        return False


if __name__ == "__main__":
    a = get_llama_completion("hi")