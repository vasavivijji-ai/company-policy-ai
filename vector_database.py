from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss


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


# Convert chunks into embeddings
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = model.encode(
    chunk_texts
)


# Get embedding dimension
embedding_dimension = embeddings.shape[1]

print("Number of chunks:", len(chunks))

print(
    "Embedding dimension:",
    embedding_dimension
)


# Create FAISS index
index = faiss.IndexFlatL2(
    embedding_dimension
)


# Add embeddings to FAISS
index.add(
    embeddings.astype("float32")
)


print("FAISS index created.")

print(
    "Number of vectors in index:",
    index.ntotal
)


# User question
question = "Can employees work remotely?"

# Create question embedding
question_embedding = model.encode(
    [question]
)


# Search FAISS
distances, indices = index.search(
    question_embedding.astype("float32"),
    1
)


# Get the best result
best_index = indices[0][0]

best_chunk = chunks[best_index]


# Display result
print("\n==============================")
print("USER QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("MOST RELEVANT POLICY")
print("==============================")

print(best_chunk.page_content)


print("\n==============================")
print("DISTANCE")
print("==============================")

print(distances[0][0])