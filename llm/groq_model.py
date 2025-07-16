import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
