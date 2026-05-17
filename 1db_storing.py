# db_storing.py
# Loads the embeddings into into two parallel ChromaDB collections.
# Role: turns text into a searchable vector index for the retrieval stage.
# Input: embeddings_bge_enriched.npy, embeddings_e5_enriched.npy,
# retrieval_metadata_enriched.csv
# Output: ChromaDB collections -> retrieval.py

# Two separate collections to separate BGE and E5 vectors. Mixing them would
# create issues with similarity.
import numpy as np
import pandas as pd
import chromadb

config_a_bge = np.load("embeddings_bge_enriched.npy")
config_b_e5 = np.load("embeddings_e5_enriched.npy")
metadata = pd.read_csv("retrieval_metadata_enriched.csv")
print(f"Config A: {len(config_a_bge)}")
print(f"Config B: {len(config_b_e5)}")
print(f"CSV rows: {len(metadata)}")

chromaDB = chromadb.PersistentClient(path="/content")

config_a_bge_collection = chromaDB.get_or_create_collection(name="config_a_bge")
config_b_e5_collection = chromaDB.get_or_create_collection(name="config_b_e5")

# Batch size 5000 to stay under CHroma's limits
def database_adder(database, vectors, metadata):
  configuration = vectors.tolist()
  for batch_start in range(0, len(configuration), 5000):
    batch_end = min(batch_start+5000, len(configuration))
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    for vector in range(batch_start, batch_end):
      ids.append(str(metadata["id"][vector]))
      embeddings.append(configuration[vector])
      documents.append(str(metadata["content"][vector]))
      metadatas.append({
          "title":str(metadata["title"][vector]),
          "category":str(metadata["category"][vector]),
          "subcategory":str(metadata["subcategory"][vector]),
          "tags":str(metadata["tags"][vector]),
          "difficulty":str(metadata["difficulty"][vector]),
          "likes":int(metadata["likes"][vector]),
          "upvotes":int(metadata["upvotes"][vector]),
      })

    database.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
  )

database_adder(config_a_bge_collection, config_a_bge, metadata)
database_adder(config_b_e5_collection, config_b_e5, metadata)
