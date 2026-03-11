import chromadb

client = chromadb.PersistentClient(path="./data")
collection =  client.get_collection("my_docs")

result = collection.query(query_texts=["aaa"],n_results=2)
print(result)