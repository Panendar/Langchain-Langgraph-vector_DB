from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb

doc_ids = ["doc_1","doc_2","doc_3","doc_4"]

docs = [
    "LangGraph is a graph-based workflow library built on LangChain.",
    "LangChain enables building LLM-powered chains and pipelines.",
    "LangGraph supports conditional routing, loops, and state management.",
    "Nodes in LangGraph are Python functions that transform state.",
]


client = chromadb.PersistentClient(path="./data")
collection = client.get_or_create_collection("qna_workflow")
collection.upsert(ids=doc_ids, documents=docs)

def search_chroma(question) -> List[str]:
    result = collection.query(query_texts=[question], n_results=3)
    documents_nested = result.get("documents", [])
    if documents_nested:
        docs_for_query = documents_nested[0]
        return [d for d in docs_for_query if isinstance(d, str)]
    return []

class WorkflowState(TypedDict):
    question:str
    documents:List[str]
    generation:Optional[str]
    attempts:int

model = ChatOllama(model="gemma:2b")
output = StrOutputParser()

# Building Node
def analyze(state:WorkflowState) -> WorkflowState:
    prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer.

Rewrite the question into a short search query.

Rules:
- Return ONLY the search query.
- Do NOT explain anything.
- Do NOT include sentences like "Here is the query".
- Output must be a single line.

Question: {question}
    """)
    chain = prompt | model | output
    analyzed_question = chain.invoke({"question": state["question"]})
    analyzed_question = analyzed_question.split("\n")[-1]
    analyzed_question = analyzed_question.replace("*", "").strip()
    print(f"[analyze] Original: {state['question']}")
    print(f"[analyze] Analyzed Query: {analyzed_question}")
    print("=" * 25)

    state["question"] = analyzed_question
    return state


def retrieve(state:WorkflowState) -> WorkflowState:
    results = search_chroma(state["question"])
    print(f"Retrieved Documents: {results}")
    return {**state, "documents":results, "attempts": state["attempts"] + 1}


def generate(state:WorkflowState) -> WorkflowState:
    context = "\n".join(state["documents"]) if state["documents"] else "No Context Found."
    prompt = f"Context:\n{context}\n\nAnswer this: {state['question']}"
    answer = model.invoke(prompt).content.strip()
    return {**state, "generation":answer}

def should_retry(state:WorkflowState) -> str:
    if not state["documents"] and state["attempts"] < 3:
        return "retrieve"
    return "generate"

# Building Graph
graph = StateGraph(WorkflowState)

graph.add_node("analyze", analyze)
graph.add_node("retrieve",retrieve)
graph.add_node("generate",generate)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "retrieve")
graph.add_conditional_edges("retrieve", should_retry, {
    "generate":"generate",
    "retrieve": "retrieve"
})
# graph.add_edge("retrieval", "generate")
graph.add_edge("generate", END)

app = graph.compile()

result = app.invoke({
    "question": "What is LangGraph and how does it differ from LangChain?",
    "documents": [],
    "generation": None,
    "attempts": 0
})
print("\nRetrieved Documents:")
for doc in result["documents"]:
    print("-", doc)

print("\nFinal Answer:\n")
print(result["generation"])