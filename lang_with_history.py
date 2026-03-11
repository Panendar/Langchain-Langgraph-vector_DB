from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# setting up llm model
llm = ChatOllama(model="gemma:2b", temperature= 0.7)

# setting up prompt template with chat_history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant who helps with the regular activities, in a friendly manner."),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human", "{text}")
])

# setting up store to keep chat histories
store = {}

# output and chain set-up
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# function to get session history based on session id 
def get_session_history(session_id: str) -> ChatMessageHistory:       # for the others to understand saying that provide --
    if session_id not in store:                                   # --the session_id a string and will return the chatMessageHistory as output
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# wrap the chain with the vx.1 history manager
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "text",
    history_messages_key = "chat_history"
)

# configuration of the session_id's 
session_id = "user_123"
config = {"configurable" : {"session_id" : session_id}}

# interaction should be passed with the config

# for the same config
c_r1 = chain_with_history.invoke(
    {"text" : "Hello, My name is Panendar"},
    config = config
)
print(f"AI's response: {c_r1}")      # AI's response: Hello! It's a pleasure to meet you, Panendar! How can I assist you today?

# c_r2 = chain_with_history.invoke(
#     {"text" : "what's my name?"},
#     config = config
# )
# print(f"AI's response: {c_r2}")         # AI's response: Your name is Panendar. How may I assist you today?

# # for another user by creating different config
# session_id_2 = "user_456"
# config_2 = {"configurable" : {"session_id" : session_id_2}}

# c2_r1 = chain_with_history.invoke(
#     {"text" : "hi, I am Alice."},
#     config = config_2
# )
# print(f"AI's response: {c2_r1}")      # AI's response: Hello Alice! It's great to meet you. How can I help you today?


# c2_r2 = chain_with_history.invoke(
#     {"text" : "can you remind me my name?"},
#     config = config_2
# )
# print(f"AI's response: {c2_r2}")      # AI's response: Your name is Alice. How can I assist you further today?