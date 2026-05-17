# retrieval.py
# This file embeds a user query with the same model that built each collection and
# runs cosine similarity search over ChromaDB to return top-k candidates.
# Role: performs vector search given user input ranks semantically close prompts to be fed in the reranker.
# Input:  retrieval_metadata_enriched.csv (from preprocessing.py),
#         embeddings_bge_enriched.npy and embeddings_e5_enriched.npy (from preprocessing.py, used for sanity checks),
#         ChromaDB collections config_a_bge and config_b_e5 (from db_storing.py),
#         seeded random sample from metadata used as the query
# Output: candidates_for_reranker.json containing query, top_k, source_id,
#         and ranked candidate lists with metadata for Config A and Config B,
#         that will be fed in reranking.py

# Each query is encoded with the same model that built its target collection.
import chromadb
import numpy as np
import pandas as pd
import time
import random
import json
from sentence_transformers import SentenceTransformer

# Load metadata and embedding arrays so this file can run independently of preprocessing.py / embedding.py.
# These are produced by preprocessing.py and consumed by embedding.py; we reload them here for the sanity checks at the end.
metadata = pd.read_csv("retrieval_metadata_enriched.csv")
config_a_bge = np.load("embeddings_bge_enriched.npy")
config_b_e5 = np.load("embeddings_e5_enriched.npy")

bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
e5_model = SentenceTransformer("intfloat/e5-base-v2")


chromaDB = chromadb.PersistentClient(path="/content")
config_a_chromadb = chromaDB.get_collection(name="config_a_bge")
config_b_chromadb = chromaDB.get_collection(name="config_b_e5")

print(f"ChromaDB Config A Loading Check: {config_a_chromadb.count()} prompts")
print(f"ChromaDB Config B Loading Check: {config_b_chromadb.count()} prompts")

# query_prefix exists so E5 can receive its required "query: " prefix
# without forcing the BGE path to use it. Also, Chroma returns 1 - cosine
# distance, such that we get a similarity score in [0, 1].


def retrieve_chromadb(query: str, collection, embedding_model, top_k: int = 20, query_prefix=""):
    query_vector = embedding_model.encode(
        query_prefix + query,
        convert_to_numpy=True
    ).tolist()

    # Searching ChromaDB
    start = time.time()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    latency = time.time() - start

    # ChromaDB returns lists-of-lists (one per query), so we take index [0]
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ChromaDB distance = 1 - cosine_similarity, so we convert back
    formatted = []
    for rank, (pid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        formatted.append({
            "rank":rank,
            "id":pid,
            "similarity":round(1 - dist, 4),
            "distance":round(dist, 4),
            "title":meta.get("title", ""),
            "category":meta.get("category", ""),
            "subcategory":meta.get("subcategory", ""),
            "tags":meta.get("tags", ""),
            "difficulty":meta.get("difficulty", ""),
            "likes":meta.get("likes", 0),
            "upvotes":meta.get("upvotes", 0),
            "content":doc,
        })

    return formatted, latency


def print_results(results, query, label):
    print(f"Query: \"{query}\"")
    for r in results:
        print(f"\n  #{r['rank']}  [{r['id']}]  similarity={r['similarity']}")
        print(f"  Title      : {r['title']}")
        print(f"  Category   : {r['category']} > {r['subcategory']}")
        print(f"  Tags       : {r['tags']}")
        print(f"  Difficulty : {r['difficulty']} | Likes: {r['likes']} | Upvotes: {r['upvotes']}")
        print(f"  Content    : {r['content'][:180]}...")


# Here we sample a real prompt and use it as the query, to ensure that we have a semantically relevant query 
# within the context of the dataset. Next, we drop that same prompt from results, otherwise the top hit would be
# the source itself with similarity 1. Thus, top_k + 1 ensures we still get top_k after exclusion.
# As random seed value we picked the date of birth of one of the member of our group born in the 14/06/2005.

# Reproducible single sampled query with random seed
random.seed(14062005)
source_id = random.choice(list(metadata["id"]))
source_row = metadata[metadata["id"] == source_id].iloc[0]
QUERY = str(source_row["content"])
SAMPLED_QUERY = QUERY
SAMPLED_SOURCE_ID = source_id
TOP_K = 20

results_a, _ = retrieve_chromadb(QUERY, config_a_chromadb, bge_model, top_k=TOP_K + 1)
results_b, _ = retrieve_chromadb(QUERY, config_b_chromadb, e5_model, top_k=TOP_K + 1, query_prefix="query: ")

# Dropping the source prompt itself from the results
results_a = [r for r in results_a if r["id"] != source_id][:TOP_K]
results_b = [r for r in results_b if r["id"] != source_id][:TOP_K]

print(f"Sampled source prompt: [{source_id}] — {source_row['title']}")
print(f"Category: {source_row['category']} > {source_row['subcategory']}\n")

print_results(results_a[:5], QUERY, "Config A (BGE)")
print_results(results_b[:5], QUERY, "Config B (E5)")


# We perform an A/B comparison across 5 queries that were selected in the same way as in the previous
# lines of code. This is done to compare how differently BGE and E5 rank the same corpus.

def trim(text, n):
    text = str(text)
    return text if len(text) <= n else text[:n] + "..."

random.seed(14062005)
sampled_ids = random.sample(list(metadata["id"]), 5)

test_queries = []
for sid in sampled_ids:
    source_row = metadata[metadata["id"] == sid].iloc[0]
    test_queries.append((sid, str(source_row["content"])))

print("Running A/B comparison across sampled held-out queries...\n")

for source_id, query in test_queries:
    # again we retrieve top 6 so that after excluding the source we still have 5
    res_a, _ = retrieve_chromadb(query, config_a_chromadb, bge_model, top_k=6)
    res_b, _ = retrieve_chromadb(query, config_b_chromadb, e5_model, top_k=6, query_prefix="query: ")

    # dropping the source prompt itself 
    res_a = [r for r in res_a if r["id"] != source_id][:5]
    res_b = [r for r in res_b if r["id"] != source_id][:5]

    ids_a = [r["id"] for r in res_a]
    ids_b = [r["id"] for r in res_b]
    overlap = len(set(ids_a) & set(ids_b))

    # shortening preview of the query so the printout is more readable
    print(f"\nSOURCE: [{source_id}]  QUERY: \"{trim(query, 80)}\"")
    for i in range(5):
        print(f"  #{i+1}   A: [{res_a[i]['id']}] sim={res_a[i]['similarity']} | {trim(res_a[i]['title'], 80)}")
        print(f"            content: {trim(res_a[i]['content'], 150)}")
        print(f"       B: [{res_b[i]['id']}] sim={res_b[i]['similarity']} | {trim(res_b[i]['title'], 80)}")
        print(f"            content: {trim(res_b[i]['content'], 150)}")
    print(f"  Shared prompts in top 5: {overlap}/5")


def get_candidates_for_reranker(query: str, top_k: int = 50):
    results_a, _ = retrieve_chromadb(query, config_a_chromadb, bge_model, top_k=top_k)
    results_b, _ = retrieve_chromadb(query, config_b_chromadb, e5_model, top_k=top_k, query_prefix="query: ")
    output = {
        "query": query,
        "top_k": top_k,
        "config_a_candidates": results_a,
        "config_b_candidates": results_b,
    }
    return output
# We take an example query to test our rerankers
QUERY = "help me write a professional email to a client"
candidates = get_candidates_for_reranker(QUERY, top_k=50)
with open("/content/candidates_for_reranker.json", "w") as f:
    json.dump(candidates, f, indent=2)
print(f"Query: {QUERY}")
print(f"Exported {len(candidates['config_a_candidates'])} candidates (Config A)")
print(f"Exported {len(candidates['config_b_candidates'])} candidates (Config B)")
print("Saved to: /content/candidates_for_reranker.json")


# Sanity checks so that the vector counts and metadata row counts match before reranking
print(config_a_bge.shape)
print(config_b_e5.shape)
print(metadata.shape)

print("Chroma A:", config_a_chromadb.count())
print("Chroma B:", config_b_chromadb.count())

candidates.keys()

print("\nExample candidate from Config A:")
print(candidates["config_a_candidates"][49])
print("\nExample candidate from Config B:")
print(candidates["config_b_candidates"][49])
