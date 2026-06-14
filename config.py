import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")