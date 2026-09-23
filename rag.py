from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import os

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()


# Get OpenAI API key
api_key = os.getenv(
    "OPENAI_API_KEY"
)


# Create OpenAI client
client = OpenAI(
    api_key=api_key
)


# ==========================================
# 1. LOAD DOCUMENT
# ==========================================

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# ==========================================
# 2. CHUNK DOCUMENT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print(
    "Total chunks:",
    len(chunks)
)


# ==========================================
# 3. CREATE EMBEDDINGS
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


# ==========================================
# 4. CREATE FAISS INDEX
# ==========================================

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    embedding_dimension
)

index.add(
    embeddings.astype("float32")
)

print(
    "FAISS index created."
)


# ==========================================
# 5. RETRIEVAL FUNCTION
# ==========================================

def retrieve_documents(
    question,
    top_k=3
):

    question_embedding = model.encode(
        [question]
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    retrieved_chunks = []

    for index_number in indices[0]:

        retrieved_chunks.append(
            chunks[index_number]
        )

    return retrieved_chunks


# ==========================================
# 6. CREATE CONTEXT
# ==========================================

def create_context(
    documents
):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# ==========================================
# 7. USER QUESTION
# ==========================================

question = (
    "How many vacation days do employees receive?"
)


# ==========================================
# 8. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

retrieved_documents = retrieve_documents(
    question,
    top_k=3
)


# ==========================================
# 9. CREATE CONTEXT
# ==========================================

context = create_context(
    retrieved_documents
)


# ==========================================
# 10. SEND CONTEXT TO LLM
# ==========================================

instructions = f"""
You are the Company Policy AI Assistant.

Answer the user's question using ONLY
the provided company policy context.

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


# ==========================================
# 11. GET FINAL ANSWER
# ==========================================

answer = response.output_text


print("\n==============================")
print("QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

print(context)


print("\n==============================")
print("AI ANSWER")
print("==============================")

print(answer)