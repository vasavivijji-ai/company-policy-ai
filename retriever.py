from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss


# Load the company policy
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


# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Create embeddings
chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

embeddings = model.encode(
    chunk_texts
)


# Get embedding dimension
embedding_dimension = embeddings.shape[1]


# Create FAISS index
index = faiss.IndexFlatL2(
    embedding_dimension
)


# Add embeddings to FAISS
index.add(
    embeddings.astype("float32")
)


print("FAISS index created.")
print("Total vectors:", index.ntotal)


# Retrieval function
def retrieve_documents(question, top_k=3):

    # Convert question into embedding
    question_embedding = model.encode(
        [question]
    )

    # Search FAISS
    distances, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    # Get retrieved chunks
    retrieved_chunks = []

    for index_number in indices[0]:

        retrieved_chunks.append(
            chunks[index_number]
        )

    return retrieved_chunks


# Convert retrieved documents into text
def create_context(documents):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return context


# Test question
question = "How many vacation days do employees receive?"


# Retrieve relevant documents
results = retrieve_documents(
    question,
    top_k=3
)


# Create context
context = create_context(
    results
)


# Display results
print("\n==============================")
print("USER QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================")

print(context)