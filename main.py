import os

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI


# Load environment variables from .env
load_dotenv()


# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")


# Create OpenAI client
client = OpenAI(api_key=api_key)


# Create FastAPI application
app = FastAPI()


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "My AI API is working!"
    }


# About endpoint
@app.get("/about")
def about():
    return {
        "project": "Company Policy AI Assistant",
        "technology": "FastAPI + OpenAI",
        "purpose": "Answer questions using company policy"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Company Policy AI Assistant"
    }


# Model information endpoint
@app.get("/model")
def model_info():
    return {
        "model": "gpt-5-mini"
    }


# Ask endpoint
@app.get("/ask")
def ask_question(question: str):

    # Company policy
    company_policy = """
    Company Policy:

    1. Vacation:
    Employees receive 15 vacation days per year.

    2. Remote Work:
    Employees may work remotely according to company guidelines.

    3. Health Insurance:
    The company provides health insurance benefits to employees.

    4. Dental and Vision:
    The company provides dental and vision insurance benefits.

    5. Sick Leave:
    Employees receive sick leave according to company policy.

    6. Vacation Requests:
    Employees should request vacation in advance according to company guidelines.

    7. Federal Holidays:
    The company observes designated federal holidays.

    8. Company Laptop:
    Eligible employees may receive a company laptop for work purposes.
    """


    # -----------------------------
    # GUARDRAIL
    # -----------------------------

    policy_keywords = [
        "vacation",
        "remote",
        "work",
        "insurance",
        "dental",
        "vision",
        "sick",
        "holiday",
        "laptop",
        "company",
        "employee",
        "leave"
    ]


    # Convert question to lowercase
    question_lower = question.lower()


    # Check whether question is related to company policy
    is_policy_question = any(
        keyword in question_lower
        for keyword in policy_keywords
    )


    # Stop unrelated questions
    if not is_policy_question:
        return {
            "question": question,
            "answer": "I can only answer questions about company policies."
        }


    # -----------------------------
    # AI INSTRUCTIONS
    # -----------------------------

    instructions = f"""
    You are the Company Policy AI Assistant.

    Answer the user's question using ONLY the company policy
    provided below.

    Rules:

    1. Use only the information in the company policy.

    2. Do not use general knowledge.

    3. Do not provide information about other companies.

    4. Do not provide laws or policies from other countries.

    5. Do not make up information.

    6. If the policy contains information that answers the question,
       provide that information clearly.

    7. If the policy contains partial information,
       provide only the information that is available.

    8. If the policy contains absolutely no information
       related to the question, say:

       "I couldn't find that information in the company policy."

    9. Keep the answer short and direct.

    Company Policy:

    {company_policy}
    """


    # -----------------------------
    # CALL OPENAI
    # -----------------------------

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=question
    )


    # Get AI answer
    answer = response.output_text


    # Return response
    return {
        "question": question,
        "answer": answer
    }