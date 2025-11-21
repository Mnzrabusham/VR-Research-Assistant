"""
VR Research Assistant with Fine-Tuned Model (chatgpt 3.5-turbo) for better VR-specific responses
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Read fine-tuned model ID
USE_FINETUNED = True  # Set to False to use base model

if USE_FINETUNED:
    try:
        with open("finetuned_model_id.txt", "r") as f:
            MODEL_NAME = f.read().strip()
        print(f"Using fine-tuned model: {MODEL_NAME}\n")
    except FileNotFoundError:
        print("Fine-tuned model not found, using base model")
        MODEL_NAME = "gpt-3.5-turbo"
else:
    MODEL_NAME = "gpt-3.5-turbo"
    print(f"Using base model: {MODEL_NAME}\n")

# Load PDFs
papers_dir = "papers/"
documents = []

print("Loading papers...")
for pdf_file in os.listdir(papers_dir):
    if pdf_file.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(papers_dir, pdf_file))
        documents.extend(loader.load())

# Split into chunks
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Create vector database
print("Creating vector database...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt template
prompt = ChatPromptTemplate.from_template("""You are an expert VR research assistant specializing in attention guidance, spatial perception, and immersive technologies.

Answer the question based on the following research context. Be specific and cite relevant findings when possible.

Context:
{context}

Question: {input}

Answer:""")

# Create chains with fine-tuned model
print(f"Initializing LLM ({MODEL_NAME})...")
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Ask questions
print("=" * 80)
print("VR Research Assistant (Fine-Tuned)")
print("=" * 80)

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == 'quit':
        print("\nGoodbye!")
        break
    
    print("Searching and generating answer...\n")
    result = retrieval_chain.invoke({"input": question})
    
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: {len(result['context'])} documents retrieved")
    
    # Optionally show source snippets
    show_sources = input("\nShow source snippets? (y/n): ")
    if show_sources.lower() == 'y':
        for i, doc in enumerate(result['context'], 1):
            print(f"\n--- Source {i} ---")
            print(f"File: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
