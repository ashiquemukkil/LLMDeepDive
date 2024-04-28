import os
import openai
from openai import OpenAI
import requests

OLLAMA_URL = "http://localhost:11434/api/generate" 

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


if __name__ == "__main__":
    download_llama()