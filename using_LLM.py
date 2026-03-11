from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator helps to translate from {input_language} to {output_language}, Return only the text in {output_language}, no explanations or additional information."),
    ("user", "{text}")
])

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# result = chain.invoke({
#     "input_language": "English",
#     "output_language": "Telugu",
#     "text": "Hello, How are you doing today?"
# })
# print(result)

results = chain.batch([
    {"input_language": "English", "output_language": "Telugu", "text": "Hello"},
    {"input_language": "English", "output_language": "German", "text": "Hello"},
    {"input_language": "English", "output_language": "Hindi", "text": "Hello"}
])
for r in results:
    print(r)





# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an assistant who helps with the regular activities, in a friendly manner."),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", "{text}")
# ])

# llm = ChatOllama(model="gemma:2b", temperature= 0.8)

# store = {}  # In-memory store for chat history acts as a database

# output_parser = StrOutputParser()

# chain = prompt | llm | output_parser

# # Function to provide the correct history object based on session ID

# def get_history(session_id: str) -> RunnableWithMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return RunnableWithMessageHistory(store[session_id])

# # chain with the history support
# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     get_history,
#     input_message_key = "input",
#     history_message_key = "chat_history"
# )

# input = input("Enter your message: ")

# run = True
# while run:
#     session_id = "default_session"  # In real applications, this would be dynamic per user/session
#     result = chain_with_history.invoke({
#         "text": input,
#         "session_id": session_id
#     })
#     print("Ollama Response:", result)
#     input = input("Enter your message (or type 'exit' to quit): ")
#     if input.lower() == 'exit':
#         run = False
