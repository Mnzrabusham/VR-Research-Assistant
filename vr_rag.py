from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# 1. Load PDFs
papers_dir = "papers/"
documents = []
for pdf_file in os.listdir(papers_dir):
    if pdf_file.endswith('.pdf'):
        loader = PyPDFLoader(os.path.join(papers_dir, pdf_file))
        documents.extend(loader.load())

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create vector database
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Create prompt template
prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:

{context}

Question: {input}""")

# 6. Create chains
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 7. Ask questions
while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == 'quit':
        break
    
    result = retrieval_chain.invoke({"input": question})
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['context'])} documents")