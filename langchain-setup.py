import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("PPLX_API_KEY")
if not api_key:
    print("Warning: PPLX_API_KEY not found in environment variables.")

from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the prompt Template - be very specific to avoid web search explanations
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate the text from {input_language} to {output_language}. Return ONLY the translated text, no explanations or additional information."),
    ("user", "{text}")
])

# Langchain automatically uses the API key from the environment variable
perplexity = ChatPerplexity(model="sonar-pro", temperature=0.3)

# initialize the output parser
output_parser = StrOutputParser()

# Build the LCEL chain using the (|) pipe operator
chain = prompt | perplexity | output_parser

# Invoke the chain
result = chain.invoke({
    "input_language" : "English",
    "output_language": "Spanish",
    "text" : "Hello , How is it going?"
})

print(result)