import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from rag_engine import retrieve_context


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv(
    "OPENAI_API_KEY"
)


# ==========================================
# CREATE OPENAI CLIENT
# ==========================================

client = OpenAI(
    api_key=api_key
)


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Company Policy AI Assistant",
    description="RAG-based AI assistant for company policies",
    version="1.0"
)


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Company Policy AI Assistant is running!"
    }


# ==========================================
# ABOUT
# ==========================================

@app.get("/about")
def about():

    return {
        "project": "Company Policy AI Assistant",
        "technology": "Python + FastAPI + RAG + FAISS + OpenAI",
        "purpose": "Answer questions using company policy documents"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "Company Policy AI Assistant"
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model")
def model_info():

    return {
        "model": "gpt-5-mini",
        "architecture": "RAG + FAISS + OpenAI"
    }


# ==========================================
# ASK QUESTION
# ==========================================

@app.get("/ask")
def ask_question(
    question: str
):

    # ======================================
    # RETRIEVE RELEVANT CONTEXT
    # ======================================

    context = retrieve_context(
        question
    )


    # ======================================
    # GUARDRAIL
    # ======================================

    if context is None:

        return {
            "question": question,
            "answer":
                "I couldn't find relevant information "
                "in the company policy."
        }


    # ======================================
    # LLM INSTRUCTIONS
    # ======================================

    instructions = f"""
You are the Company Policy AI Assistant.

Answer the user's question using ONLY
the company policy context provided below.

Rules:

1. Use only the information in the context.

2. Do not use general knowledge.

3. Do not make up information.

4. Do not add information that is not supported
by the provided context.

5. If the answer is not available in the context,
say:

"I couldn't find that information in the company policy."

6. Keep the answer short and direct.

Company Policy Context:

{context}
"""


    # ======================================
    # CALL OPENAI
    # ======================================

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=question
    )


    # ======================================
    # GET ANSWER
    # ======================================

    answer = response.output_text


    # ======================================
    # RETURN JSON
    # ======================================

    return {
        "question": question,
        "answer": answer
    }