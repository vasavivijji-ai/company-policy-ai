import os

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# -----------------------------------
# 1. Load company policy document
# -----------------------------------

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# -----------------------------------
# 2. Split document into chunks
# -----------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print("Total chunks:", len(chunks))


# -----------------------------------
# 3. Create OpenAI embeddings
# -----------------------------------

def create_embeddings(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )


chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = create_embeddings(
    chunk_texts
)

print(
    "Embedding shape:",
    embeddings.shape
)


# -----------------------------------
# 4. Create FAISS vector index
# -----------------------------------

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    embedding_dimension
)

index.add(embeddings)

print(
    "Vectors in FAISS:",
    index.ntotal
)


# -----------------------------------
# 5. Retrieve relevant documents
# -----------------------------------

def retrieve_relevant_documents(
    question,
    top_k=3,
    max_distance=0.8
):

    question_embedding = create_embeddings(
        [question]
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if distance <= max_distance:

            results.append({
                "document": chunks[index_number],
                "distance": float(distance)
            })

    return results


# -----------------------------------
# 6. Create context
# -----------------------------------

def create_context(results):

    context = "\n\n".join(
        result["document"].page_content
        for result in results
    )

    return context


# -----------------------------------
# 7. Public function for FastAPI
# -----------------------------------

def retrieve_context(question):

    results = retrieve_relevant_documents(
        question
    )

    if not results:
        return None

    context = create_context(
        results
    )

    return context