# RAG is mostly used to provide fresh content, RAG has two phases — an Indexing Phase (done once, offline) and a Retrieval Phase (done at query time, live).

# Phase-1 Raw Docs -> Chunks(splits docs into manageable pieces (e.g. 500 tokens per chunk)) -> Embeddings -> Vector Store
# Phase-2 User Query -> Embedded Query -> Vector Search -> Top_k_chunks -> Prompt + LLM -> Result

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Loading the Document
loader = TextLoader("company_policy.txt")
documents = loader.load()

# prompt template
template = """You are a helpful assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Splitting into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, chunk_overlap = 70, separators=["\n\n", "\n", " "]
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")
print("=" * 30)

# Embeddings
embedding = HuggingFaceEmbeddings()

# vector store
vector_store = Chroma.from_documents(documents=chunks,embedding=embedding,persist_directory="./data")

# load existing vector store
# vector_store = Chroma(persist_directory="./data", embedding_function=embedding)

# build the RAG chain
model = ChatOllama(model="llama3.2:3b")
qa_chain = RetrievalQA.from_chain_type(
    llm = model,
    chain_type = "stuff",
    retriever = vector_store.as_retriever(search_type="similarity",
                                          search_kwargs={"k":3}),
    return_source_documents = True,
    chain_type_kwargs = {"prompt": prompt}
)

run = True
while run:
    question = input("Ask any Question have in your mind (quit/exit to close): ")
    if question.lower() == "quit" or question.lower() =="exit":
        break
    output = qa_chain.invoke({"query": question})
    answer = output.get("result")
    source_docs = output.get("source_documents", [])
    print("Answer: ",answer)
    print("sources: ",source_docs[0].page_content if source_docs else "No Source Found")