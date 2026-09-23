from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load the company policy document
loader = TextLoader(
    "documents/company_policy.txt"
)

documents = loader.load()


# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)


# Split the document
chunks = text_splitter.split_documents(
    documents
)


# Print total number of chunks
print("Total chunks:", len(chunks))


# Inspect each chunk
for i, chunk in enumerate(chunks):

    print(f"\n--- CHUNK {i + 1} ---")

    print("Characters:", len(chunk.page_content))

    print("Content:")
    print(chunk.page_content)

    print("Metadata:")
    print(chunk.metadata)