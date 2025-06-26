import os
from dotenv import load_dotenv
from autogen import config_list_from_json

load_dotenv()

GORQ_API_KEY = os.getenv("GROQ_API_KEY")

def configure():
    return [{
        "model": "llama-3.1-8b-instant",
        "api_key": GORQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1"
    }]