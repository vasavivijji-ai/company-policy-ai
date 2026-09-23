from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss


# ==========================================
# 1. LOAD DOCUMENT
# ==========================================

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# ==========================================
# 2. IMPROVED CHUNKING
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
# 3. ADD METADATA
# ==========================================

for chunk in chunks:

    chunk.metadata["document_type"] = (
        "company_policy"
    )

    chunk.metadata["source_name"] = (
        "company_policy.txt"
    )


# ==========================================
# 4. CREATE EMBEDDINGS
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
# 5. CREATE FAISS INDEX
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
# 6. RETRIEVAL
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

    results = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        results.append({
            "document": chunks[index_number],
            "distance": float(distance)
        })

    return results


# ==========================================
# 7. RELEVANCE FILTER
# ==========================================

def retrieve_relevant_documents(
    question,
    top_k=3,
    max_distance=0.8
):

    results = retrieve_documents(
        question,
        top_k
    )

    relevant_results = []

    for result in results:

        if result["distance"] <= max_distance:

            relevant_results.append(
                result
            )

    return relevant_results


# ==========================================
# 8. CREATE CONTEXT
# ==========================================

def create_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            result["document"].page_content
        )

    return "\n\n".join(
        context_parts
    )


# ==========================================
# 9. TEST QUESTION
# ==========================================

question = (
    "How many vacation days do employees receive?"
)


# ==========================================
# 10. RETRIEVE
# ==========================================

results = retrieve_relevant_documents(
    question,
    top_k=3
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n==============================")
print("QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("RETRIEVED RESULTS")
print("==============================")


for i, result in enumerate(results):

    print(
        f"\n--- RESULT {i + 1} ---"
    )

    print(
        "Distance:",
        result["distance"]
    )

    print(
        "Content:",
        result["document"].page_content
    )

    print(
        "Metadata:",
        result["document"].metadata
    )


# ==========================================
# 12. CREATE CONTEXT
# ==========================================

if not results:

    print(
        "\nI couldn't find relevant information "
        "in the company policy."
    )

else:

    context = create_context(
        results
    )

    print("\n==============================")
    print("CONTEXT")
    print("==============================")

    print(context)