import chromadb

client = chromadb.PersistentClient(path="./data")

collection = client.get_or_create_collection("task_doc")

doc_id = ["task_1", "task_2", "task_3", "task_4", "task_5"]

docs = [
    "Python is a programming language",
    "Cats and dogs are popular pets",
    "Machine learning uses statistics",
    "Cooking pasta requires boiling water",
    "Deep learning is a subset of AI"
]

collection.add(ids=doc_id, documents=docs)

# data = collection.get(include=["documents"])
# print(data)

query = "artificial intelligence and data"
result = collection.query(query_texts=query, n_results=2)

for doc, dist in zip(result['documents'][0], result['distances'][0]):
    print(f"Doc: {doc}")
    print(f"Similarity score: {1- dist:.2f}\n")