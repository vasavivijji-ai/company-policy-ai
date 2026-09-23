from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the company policy document
loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)


# Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Convert each chunk into an embedding
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = model.encode(
    chunk_texts
)


# Display results
print("Total chunks:", len(chunks))

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "First chunk:"
)

print(
    chunk_texts[0]
)

print(
    "\nFirst chunk embedding:"
)

print(
    embeddings[0]
)
# Create a user question
question = "How many vacation days do employees receive?"


# Convert the question into an embedding
question_embedding = model.encode(
    question
)


# Display the question embedding
print("\nQuestion:")
print(question)

print("\nQuestion embedding:")
print(question_embedding)

print("\nQuestion embedding shape:")
print(question_embedding.shape)
# Compare the question with the first chunk
similarity = cosine_similarity(
    [question_embedding],
    [embeddings[0]]
)

print("\nSimilarity with first chunk:")
print(similarity[0][0])
# Compare the question with all chunks
similarities = cosine_similarity(
    [question_embedding],
    embeddings
)[0]


print("\nSimilarity scores:")

for i, score in enumerate(similarities):

    print(
        f"Chunk {i + 1}: {score:.4f}"
    )
    # Find the most relevant chunk
best_index = similarities.argmax()

best_chunk = chunks[best_index]

best_score = similarities[best_index]


print("\n==============================")
print("MOST RELEVANT CHUNK")
print("==============================")

print("Chunk number:", best_index + 1)

print("Similarity score:", best_score)

print("\nChunk content:")
print(best_chunk.page_content)