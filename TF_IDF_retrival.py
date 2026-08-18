# documents = [
#     "the cat eats fish",
#     "the dog eats meat",
#     "the bird eats seeds",
# ]

# TF-IDF : 
# IDF stands for Inverse Document Frequency measures how rare a term is across the document collection
# the common formula is:- IDF(t) = log(N/df(t))

# N = total number of documents,     df(t) = number of documents containing term t
# for our example the df(cat) = 1 appears in one doc[doc_1] df(eats) = 3
# IDF(cat) ~ 1 and IDF(eats) 0 from this we can tell that 

# TF-IDF(t,d) = TF(t,d) * IDF(t)

def tokenize(text):
    return text.lower().split()

def doc_frequency(term,documents):
    count = 0
    for doc in documents:
        tokens = tokenize(doc)

        if term in tokens:
            count +=1
    return count

# print(doc_frequency("eats", documents))

# calculate IDF

import math

def idf(term, documents):
    n =len(documents)
    df = doc_frequency(term,documents)

    ans = round(math.log(n/df), 3)
    return ans


def tf(term,document):
    tokens = tokenize(document)

    count = tokens.count(term)
    total_terms  = len(tokens)
    ans = round(count/total_terms, 3)
    return ans

def tf_idf(term, document, documents):
    term_tf = tf(term,document)
    term_idf = idf(term, documents)

    return round(term_tf * term_idf, 3)

def build_vocab(documents):
    vocabulary = set()
    for doc in documents:
        tokens = tokenize(doc)
        vocabulary.update(tokens)

    return sorted(vocabulary)

# vocabulary = build_vocab(documents)

def tf_idf_vector(document, documents, vocabulary):
    vector = []
    for term in vocabulary:
        score = tf_idf(term, document, documents)
        vector.append(score)
    return vector

# print(tf_idf_vector(
#     "cat eats fish",
#     documents,
#     vocabulary
# ))

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a,b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a*a for a in vector_a))
    magnitude_b = math.sqrt(sum(b *b for b in vector_b))

    if magnitude_a ==0 or magnitude_b ==0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def search(query, documents, top_k=3):
    vocabulary = build_vocab(documents + [query])
    query_vector = tf_idf_vector(query,documents, vocabulary)
    results = []

    for index, document in enumerate(documents):
        document_vector = tf_idf_vector(document, documents, vocabulary)
        score = cosine_similarity(query_vector, document_vector)
        results.append((index,document, score))

    results.sort(key=lambda x : x[2], reverse=True)

    return results[:top_k]



# Test
documents = [
    "cat eats fish",
    "cat likes fish",
    "dog eats meat"
]

results = search(
    "cat fish",
    documents,
    top_k=2
)

for index, document, score in results:
    print(score, document)