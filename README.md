\# Cross-Lingual Embedding Retrieval System



This project implements a cross-lingual semantic search system where queries in English retrieve relevant documents written in Spanish using multilingual sentence embeddings and FAISS vector search.



\## Technologies Used

\- Python

\- Sentence Transformers

\- FAISS

\- NumPy



\## Objective

To allow users to search Spanish documents using English queries by mapping both languages into the same semantic embedding space.



\## Project Workflow

1\. Spanish documents are embedded using a multilingual Sentence Transformer.

2\. These embeddings are indexed using FAISS for fast similarity search.

3\. An English query is embedded using the same model.

4\. FAISS returns the top 5 most similar Spanish documents.



\## How to Run



\### 1. Install dependencies

pip install sentence-transformers faiss-cpu numpy



\### 2. Build the index

python build\_index.py



\### 3. Run a search

python search.py



\## Sample Query

How does artificial intelligence help medicine?



\## Sample Output

"La inteligencia artificial está transformando la medicina."



This shows the system correctly retrieves relevant Spanish documents using English input.



\## Author

Anirudh



