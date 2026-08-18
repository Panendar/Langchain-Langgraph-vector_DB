documents = [
    "cat eats fish",
    "cat likes fish",
    "dog eats meat",
    "cat likes cat fish",
]

def tokenize(text):
    return text.lower().split()

#  print(tokenize(documents[0]))

def count_term(term,doc):
    tokens = tokenize(doc)
    return tokens.count(term)

#  print(count_term('cat', documents[0]))

# we know that TF is the term frequency => the no. of times the token appears / document_length(we normalize sometimes)

# TF(t,d)=    number of terms in d/ 
#           frequency of term t in d


def tf(term, doc):
    tokens = tokenize(doc)
    count = tokens.count(term)

    total_terms = len(tokens)

    return count / total_terms

# result = tf('cat', documents[-1])        # print(round(tf('cat',documents[-1]),3))
# print(f"{result:.3f}")


def build_vocabulary(documents):
    vocab = set()

    for doc in documents:
        tokens = tokenize(doc)
        vocab.update(tokens)
    return sorted(vocab)

vocabulary = build_vocabulary(documents)
# print(vocabulary)

def tf_vector(document, vocabulary):
    tokens = tokenize(document)
    total_terms = len(tokens)

    vector = []
    for term in vocabulary:
        count = tokens.count(term)
        score = count / total_terms
        vector.append(round(score,3))
    return vector

query = "cat fish"
print(tf_vector(query,vocabulary))
for doc in documents:
    print(tf_vector(doc, vocabulary))