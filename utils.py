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

def download_llama():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id =  "NousResearch/Llama-2-7b-hf"
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


if __name__ == "__main__":
    download_llama()