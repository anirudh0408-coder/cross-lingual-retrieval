import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from data import spanish_docs

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

index = faiss.read_index("spanish_index.faiss")

with open("doc_ids.pkl", "rb") as f:
    doc_ids = pickle.load(f)

query = "How does artificial intelligence help medicine?"

query_embedding = model.encode([query]).astype("float32")

k = 5
distances, indices = index.search(query_embedding, k)

print("\nEnglish Query:", query)
print("\nTop 5 Spanish Results:\n")

for i in range(k):
    doc_id = doc_ids[indices[0][i]]
    print(f"Doc ID: {doc_id}")
    print(f"Text: {spanish_docs[doc_id]}")
    print(f"Similarity Score: {distances[0][i]}")
    print("-----------------------")
s