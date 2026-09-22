# 🤖 Company Policy AI Assistant

An AI-powered application that allows employees to ask questions about company policies and receive concise, policy-based answers.

## 🚀 Project Overview

The Company Policy AI Assistant is an AI application built using Python, FastAPI, Streamlit, OpenAI, Docker, and automated LLM evaluation.

The application is designed to answer employee questions using company-provided policy information while reducing unsupported or unrelated responses through basic guardrails.

## 🏗️ Architecture

```text
User
  ↓
Streamlit UI
  ↓
FastAPI Backend
  ↓
Guardrails
  ↓
Company Policy Context
  ↓
OpenAI LLM
  ↓
AI-generated Answer
```

## ✨ Features

* Ask questions about company policies
* AI-generated responses using an LLM
* FastAPI REST API
* Streamlit user interface
* Basic AI guardrails
* Handles unsupported policy questions
* Automated evaluation using test cases
* PASS/FAIL evaluation
* Keyword-based response validation
* Dockerized application
* Cloud deployment support

## 🛠️ Technologies

* Python
* FastAPI
* Streamlit
* OpenAI API
* Docker
* Git & GitHub
* Requests
* python-dotenv
* LLM Evaluation
* AI Guardrails

## 💬 Example Questions

### Vacation

**Question:**

How many vacation days do employees receive?

**Answer:**

Employees receive 15 vacation days per year.

### Insurance

**Question:**

Do employees receive dental and vision insurance?

**Answer:**

The company provides dental and vision insurance benefits to employees.

### Unsupported Policy

**Question:**

What is the maternity leave policy?

**Answer:**

I couldn't find that information in the company policy.

### Unrelated Question

**Question:**

What is the capital of France?

**Answer:**

I can only answer questions about company policies.

## 🧪 AI Evaluation

The project includes an automated evaluation script using `evaluation.py`.

The evaluation process:

```text
Test Question
      ↓
FastAPI
      ↓
OpenAI
      ↓
Actual Answer
      ↓
Keyword Validation
      ↓
PASS / FAIL
```

The evaluation script also calculates:

* Total tests
* Passed tests
* Failed tests
* Accuracy

## 🛡️ Guardrails

The application includes basic guardrails to:

* Restrict unrelated questions
* Prevent unsupported policy information
* Reduce hallucinated responses
* Return a clear message when information is unavailable

## 🐳 Docker

The FastAPI backend is containerized using Docker.

Example:

```bash
docker build -t company-policy-ai .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 company-policy-ai
```

## 🔐 Environment Variables

The OpenAI API key is stored securely in a `.env` file.

Example:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git using `.gitignore`.

## ▶️ Run Locally

### 1. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start FastAPI

```bash
uvicorn main:app --reload
```

### 4. Start Streamlit

Open another terminal:

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit:

```text
http://localhost:8501
```

FastAPI:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```text
company-policy-ai/
│
├── app.py
├── main.py
├── evaluation.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
├── .dockerignore
└── venv/
```

## 🎯 Learning Outcomes

Through this project, I practiced:

* Building AI-powered applications
* Working with LLM APIs
* Building REST APIs with FastAPI
* Creating AI interfaces with Streamlit
* Implementing basic AI guardrails
* Evaluating LLM responses
* Containerizing applications with Docker
* Using Git and GitHub
* Deploying an AI backend to the cloud

## 🔮 Future Improvements

* Connect the application to a full RAG pipeline
* Add document upload functionality
* Use vector databases for semantic retrieval
* Add conversation history
* Improve LLM evaluation
* Add stronger hallucination detection
* Add authentication
* Improve production monitoring and logging

## 👩‍💻 Project Goal

The goal of this project is to demonstrate practical AI Engineering skills by building, evaluating, containerizing, and deploying an AI-powered application.
