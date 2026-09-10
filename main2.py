from fastapi import FastAPI

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
        "technology": "FastAPI",
        "purpose": "Answer company policy questions"
    }


@app.get("/ask")
def ask_question(question: str):
    return {
        "question": question,
        "answer": "Employees receive 15 vacation days every year."
    }
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }    