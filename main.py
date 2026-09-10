import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "My AI API is working!"
    }


@app.get("/about")
def about():
    return {
        "project": "Company Policy AI Assistant",
        "technology": "FastAPI + OpenAI",
        "purpose": "Answer questions using an LLM"
    }


@app.get("/ask")
def ask_question(question: str):

    response = client.responses.create(
    model="gpt-5-mini",
    instructions="Answer questions clearly and simply for a beginner learning AI Engineering.",
    input=question
)
    

    answer = response.output_text

    return {
        "question": question,
        "answer": answer
    }
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Company Policy AI Assistant"
    }
@app.get("/model")
def model_info():
    return {
        "model": "gpt-5-mini"
    }        