import os

from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss


# ==========================================
# STEP 1: LOAD OPENAI API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
)


# ==========================================
# STEP 2: LOAD COMPANY POLICY
# ==========================================

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# ==========================================
# STEP 3: SPLIT DOCUMENT INTO CHUNKS
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print("Total chunks:", len(chunks))


# ==========================================
# STEP 4: CREATE EMBEDDINGS
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = model.encode(
    chunk_texts
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# STEP 5: CREATE FAISS VECTOR DATABASE
# ==========================================

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    embedding_dimension
)

index.add(
    embeddings.astype("float32")
)

print(
    "Vectors in FAISS:",
    index.ntotal
)


# ==========================================
# STEP 6: RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_relevant_documents(
    question,
    top_k=3,
    max_distance=0.8
):

    # Convert question into embedding
    question_embedding = model.encode(
        [question]
    )

    # Search FAISS
    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        # Guardrail:
        # Only accept sufficiently relevant chunks
        if distance <= max_distance:

            results.append({
                "document": chunks[index_number],
                "distance": float(distance)
            })

    return results


# ==========================================
# STEP 7: CREATE CONTEXT
# ==========================================

def create_context(results):

    context = "\n\n".join(
        result["document"].page_content
        for result in results
    )

    return context


# ==========================================
# STEP 8: ASK THE LLM
# ==========================================

def generate_answer(
    question,
    context
):

    instructions = f"""
You are the Company Policy AI Assistant.

Answer the user's question using ONLY
the company policy context provided below.

Rules:

1. Use only the information in the context.

2. Do not use general knowledge.

3. Do not make up information.

4. If the answer is not available in the context,
say:

"I couldn't find that information in the company policy."

5. Keep the answer short and direct.

Company Policy Context:

{context}
"""

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=question
    )

    return response.output_text


# ==========================================
# STEP 9: TEST QUESTIONS
# ==========================================

test_questions = [

    "How many vacation days do employees receive?",

    "Do employees receive dental and vision insurance?",

    "What is the maternity leave policy?",

    "What is the capital of France?"

]


# ==========================================
# STEP 10: RUN RAG + GUARDRAIL + LLM
# ==========================================

for question in test_questions:

    print("\n")
    print("=" * 50)
    print("QUESTION")
    print("=" * 50)

    print(question)


    # --------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------

    results = retrieve_relevant_documents(
        question
    )


    # --------------------------------------
    # GUARDRAIL
    # --------------------------------------

    if not results:

        print("\nNo relevant information found.")

        print("\nFINAL ANSWER:")

        print(
            "I couldn't find relevant information "
            "in the company policy."
        )

        continue


    # --------------------------------------
    # Create context
    # --------------------------------------

    context = create_context(
        results
    )


    print("\n")
    print("=" * 50)
    print("RETRIEVED CONTEXT")
    print("=" * 50)

    print(context)


    # --------------------------------------
    # Send context + question to LLM
    # --------------------------------------

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------------
    # Display final answer
    # --------------------------------------

    print("\n")
    print("=" * 50)
    print("FINAL AI ANSWER")
    print("=" * 50)

    print(answer)