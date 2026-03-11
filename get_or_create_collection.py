import chromadb

# client = chromadb.PersistentClient(path="./data")
# collection = client.get_or_create_collection(name="my_docs")

# documents = [
#     "aaa",
#     "bbb"
# ]
# collection.add(documents=documents, ids=["1", "2"])

# collection = client.get_collection("my_docs")
# print(collection.get(include=["embeddings"]))

client = chromadb.PersistentClient(path="./data")
collection = client.get_or_create_collection(name="students")

student_ids = ["1", "2"]
student_docs = ["Pani wants to become an FS-AI engineer","Ashmit goes for Higher Studies"]
metadata = [
    {"year": "B.Tech 3rd year"},
    {"year": "B.Tech 3rd year"}
]

collection.add(ids=student_ids, documents=student_docs, metadatas=metadata)

# data = collection.get(include=["documents", "metadatas", "embeddings"])
# print(data)

results = collection.query(query_texts=["AI career"],n_results=1)
print(results)


# print(collection.get())