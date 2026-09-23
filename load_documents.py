from langchain_community.document_loaders import TextLoader


# Load the company policy document
loader = TextLoader(
    "documents/company_policy.txt"
)


# Read the document
documents = loader.load()


# Get the first document
document = documents[0]


# Print the document text
print("DOCUMENT CONTENT:")
print(document.page_content)


# Print document metadata
print("\nDOCUMENT METADATA:")
print(document.metadata)