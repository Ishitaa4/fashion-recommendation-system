import pickle
import numpy as np
import faiss
import os

EMB_FILE = "embeddings_efficientnet.pkl"
INDEX_FILE = "faiss_hnsw.index"

M = 32              # graph connectivity
EF_CONSTRUCTION = 200

print("Loading embeddings...")
with open(EMB_FILE, "rb") as f:
    X = pickle.load(f)

X = np.array(X).astype("float32")
N, D = X.shape
print("Shape:", X.shape)

print("Building FAISS HNSW index...")
index = faiss.IndexHNSWFlat(D, M)
index.hnsw.efConstruction = EF_CONSTRUCTION
index.add(X)

index.hnsw.efSearch = 64

print("Saving index...")
faiss.write_index(index, INDEX_FILE)

print("DONE.")
print("Index vectors:", index.ntotal)
