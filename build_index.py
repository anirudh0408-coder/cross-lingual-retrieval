import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data import spanish_docs
import pickle

print("Loading multilingual model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

doc_ids = list(spanish_docs.keys())
texts = list(spanish_docs.values())

print("Encoding Spanish documents...")
embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "spanish_index.faiss")

with open("doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)

print("FAISS index built successfully!")

