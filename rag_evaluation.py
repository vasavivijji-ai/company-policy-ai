from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss


# ==========================================
# STEP 1: LOAD COMPANY POLICY
# ==========================================

loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# ==========================================
# STEP 2: SPLIT DOCUMENT INTO CHUNKS
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
# STEP 3: CREATE EMBEDDINGS
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
# STEP 5: RETRIEVAL FUNCTION
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

        # Keep only relevant chunks
        if distance <= max_distance:

            results.append({
                "document": chunks[index_number],
                "distance": float(distance)
            })

    return results


# ==========================================
# STEP 6: EVALUATION TEST CASES
# ==========================================

test_cases = [

    {
        "question":
            "How many vacation days do employees receive?",

        "expected_keywords":
            ["vacation", "15"]
    },

    {
        "question":
            "Do employees receive dental and vision insurance?",

        "expected_keywords":
            ["dental", "vision"]
    },

    {
        "question":
            "Can employees work remotely?",

        "expected_keywords":
            ["remote", "work"]
    },

    {
        "question":
            "What is the maternity leave policy?",

        "expected_keywords":
            []
    },

    {
        "question":
            "What is the capital of France?",

        "expected_keywords":
            []
    }

]


# ==========================================
# STEP 7: EVALUATION
# ==========================================

passed_count = 0
failed_count = 0


print("\n")
print("=" * 60)
print("RAG EVALUATION")
print("=" * 60)


for test in test_cases:

    question = test["question"]

    expected_keywords = test[
        "expected_keywords"
    ]


    # Retrieve documents
    results = retrieve_relevant_documents(
        question
    )


    # Combine retrieved content
    retrieved_text = "\n".join(
        result["document"].page_content
        for result in results
    )


    retrieved_text_lower = (
        retrieved_text.lower()
    )


    # ======================================
    # EVALUATION LOGIC
    # ======================================

    if expected_keywords:

        passed = all(
            keyword.lower()
            in retrieved_text_lower
            for keyword in expected_keywords
        )

    else:

        # For questions where no policy
        # information should be retrieved,
        # we expect no relevant chunks.

        passed = len(results) == 0


    # ======================================
    # DISPLAY RESULT
    # ======================================

    print("\n")
    print("-" * 60)

    print(
        "Question:",
        question
    )

    print(
        "Expected keywords:",
        expected_keywords
    )

    print(
        "Retrieved chunks:",
        len(results)
    )


    if results:

        print(
            "Best distance:",
            results[0]["distance"]
        )


    print(
        "Retrieved content:"
    )

    if retrieved_text:

        print(retrieved_text)

    else:

        print(
            "No relevant content retrieved."
        )


    # ======================================
    # PASS / FAIL
    # ======================================

    if passed:

        print(
            "\nResult: PASS ✅"
        )

        passed_count += 1

    else:

        print(
            "\nResult: FAIL ❌"
        )

        failed_count += 1


# ==========================================
# STEP 8: FINAL EVALUATION SUMMARY
# ==========================================

total_tests = len(test_cases)

accuracy = (
    passed_count / total_tests
) * 100


print("\n")
print("=" * 60)
print("RAG EVALUATION RESULTS")
print("=" * 60)

print(
    "Total tests:",
    total_tests
)

print(
    "Passed:",
    passed_count
)

print(
    "Failed:",
    failed_count
)

print(
    "Retrieval accuracy:",
    f"{accuracy:.0f}%"
)

print("=" * 60)