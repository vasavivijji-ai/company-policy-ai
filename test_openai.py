import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5-mini",
    input="Explain RAG in one simple sentence."
)

print(response.output_text)
