from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1.Define State 
# State is the shared memory of your workflow — all the data nodes pass between each other as the graph executes.
class AgentState(TypedDict):
    question:str
    answer:str

# 2. Define Nodes (functions)
def analyze(state:AgentState) -> AgentState:
    print(f"Analyzing: {state['question']}")
    return state   # pass state to next node

def generate_answer(state:AgentState) -> AgentState:
    state["answer"] = f"Here's what I know about: {state['question']}"
    return state

# 3. Build Graph
graph = StateGraph(AgentState)
graph.add_node("analyze",analyze)
graph.add_node("generate", generate_answer)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "generate")
graph.add_edge("generate", END)

app = graph.compile()
result = app.invoke({"question": "What is LangGraph?", "answer": ""})
print(result["answer"])




# state with typedict

from typing import TypedDict, List
from langchain_core.messages import BaseMessage

# Define your state schema clearly
class RAGState(TypedDict):
    question: str           # Original user question
    documents: List[str]    # Retrieved context docs
    generation: str         # Final LLM answer
    iterations: int         # Loop counter (for safety)

# Node 1: Retrieve documents
def retrieve(state:RAGState) -> RAGState:
    # Simulate retrieval
    docs = [f"Relevant doc about: {state['question']}"]
    return {**state, "documents": docs, "iterations":state["iterations"] +1}

#Node 2: Generator answer
def generator(state:RAGState) -> RAGState:
    context = "\n".join(state["documents"])
    answer = f"Based on context: {context}\nAnswer:..."
    return {**state, "generation":answer}

# State flows through — each node reads AND writes!
initial_states: RAGState = {
    "question": "WHat is vector search?",
    "documents": [],
    "generation":"",
    "iterations":0
}


# MULTI STEP RAG WORKFLOW
class WorkflowState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    grade: str
    iterations: int

MAX_ITER = 3

def retrieve(state: WorkflowState) -> WorkflowState:
    docs = ["Doc1: LangGraph is .....", "Doc2: Nodes are..."]
    return {**state, "documents":docs}

def grade_documents(state:WorkflowState) -> WorkflowState:
    # SImulate grading __ in real life, use LLM to check relevance
    grade = "useful" if state["iterations"] < 2 else "not useful"
    return {**state, "grade": grade}

def generator(state: WorkflowState) -> WorkflowState:
    context = "\n".join(state["documents"])
    answer = f"Answer based on docs: {context[:100]}..."
    return {**state, "geneartion": answer}

def route_after_grade(state: WorkflowState) -> WorkflowState:
    context = "\n".join(state["documents"])
    answer = f"Answer based on docs: {context[:100]}..."
    return {**state, "generation": answer}

def route_after_grade(state: WorkflowState) -> str:
    if state["grade"] == "useful" or state["iterations"] >= MAX_ITER:
        return "generate"
    return "retrieve"  # loop back!

# Build the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generator)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", route_after_grade, {
    "generate": "generate",
    "retrieve": "retrieve"
})
workflow.add_edge("generate", END)

app = workflow.compile()
result = app.invoke({
    "question": "What is LangGraph?",
    "documents": [], "generation": "",
    "grade": "", "iterations": 0
})
print(result["generation"])



# VISUALIZING YOUR GRAPH
# from IPython.display import Image, display

# # Compile your graph (app = workflow.compile())

# # Option 1: ASCII representation
# print(app.get_graph().draw_ascii())

# # Option 2: Mermaid diagram (renders in notebooks)
# print(app.get_graph().draw_mermaid())

# # Option 3: PNG image (requires graphviz)
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception:
#     # Fallback if graphviz not installed
#     print("Install graphviz for PNG rendering")

# # Enable LangSmith tracing (set env vars)
# import os
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "your-key-here"
# # Now every invoke() is automatically traced!

# # Debug with interrupt
# app_debug = workflow.compile(interrupt_before=["generate"])
# state = app_debug.invoke(initial_state)
# # Execution pauses BEFORE "generate" — inspect state here!
# print("State before generation:", state)



# ⏸️ What is Human-in-the-Loop?
# The ability to pause a LangGraph workflow at a specific node, show the state to a human, get their approval or input, and then resume execution.

# 🔑 Key Concept: Checkpointers
# LangGraph uses checkpointers (e.g., MemorySaver) to save state — enabling pause and resume even across sessions.

# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory,
#                        interrupt_before=["generate"])
# 📖 Resources to Read
# LangGraph Docs: Human-in-the-Loop
# Search: "LangGraph interrupt_before example"
# Try: Build a workflow that asks "Should I send this email?" before the send step
# 🎯 Mini Challenge
# Extend the RAG workflow from today to pause before generating — print the state to "review" it, then manually call app.invoke(None, config) to resume.