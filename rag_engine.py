from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss


# ==========================================
# STEP 1: LOAD DOCUMENT
# ==========================================

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# ==========================================
# STEP 2: SPLIT DOCUMENT
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)


print("RAG Engine")
print("Total chunks:", len(chunks))


# ==========================================
# STEP 3: CREATE EMBEDDINGS
# ==========================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = embedding_model.encode(
    chunk_texts
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# STEP 4: CREATE FAISS INDEX
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
# STEP 5: RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def retrieve_relevant_documents(
    question,
    top_k=3,
    max_distance=0.8
):

    # Convert question to embedding
    question_embedding = embedding_model.encode(
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

        if distance <= max_distance:

            results.append({
                "document": chunks[index_number],
                "distance": float(distance)
            })

    return results


# ==========================================
# STEP 6: CREATE CONTEXT
# ==========================================

def create_context(results):

    context = "\n\n".join(
        result["document"].page_content
        for result in results
    )

    return context


# ==========================================
# STEP 7: COMPLETE RAG RETRIEVAL
# ==========================================

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