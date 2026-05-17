# Generated from app_clean_newversion2.ipynb
# Standalone Step 4 runner.
# This file starts from the JSON exported at the end of Step 3.
# Expected JSON structure:
# {
#   "query": "...",
#   "top_k": 50,
#   "config_a_candidates": [...],
#   "config_b_candidates": [...]
# }

import json
from pathlib import Path

def load_step3_candidates(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Step 3 output file not found: {path} - please run Step 3 first and provide the correct path.")
    

    with path.open("r", encoding="utf-8") as f:
        step3_output = json.load(f)

    query = step3_output["query"]
    candidates = {"config_a_candidates": step3_output["config_a_candidates"],
        "config_b_candidates": step3_output["config_b_candidates"]}

    print(f"Loaded Step 3 candidates from: {path}")
    print(f"Query: {query}")
    print(f"Config A candidates: {len(candidates['config_a_candidates'])}")
    print(f"Config B candidates: {len(candidates['config_b_candidates'])}")

    return query, candidates

path="/content/candidates_for_reranker.json"
QUERY, candidates = load_step3_candidates(path)


# Reranker Step 4 up to Metadata-Aware Reranking
# Cells 62 through 77

# %% [markdown]
# ## RERANKER (PART 4)

# %% [markdown]
# Before building the reranker, let's do a sanity check

# %%
if "config_a_bge" in globals() and "config_b_e5" in globals() and "metadata" in globals():
    print(config_a_bge.shape)
    print(config_b_e5.shape)
    print(metadata.shape)
else:
    print("Standalone mode: using candidates loaded from Step 3 JSON.")

# %%
if "config_a_chromadb" in globals() and "config_b_chromadb" in globals():
    print("Chroma A:", config_a_chromadb.count())
    print("Chroma B:", config_b_chromadb.count())
else:
    print("Standalone mode: ChromaDB collections are not needed for Step 4.")

# %%
candidates.keys()

# %%
#print example candidate
print("\nExample candidate from Config A:")
print(candidates["config_a_candidates"][min(49, len(candidates["config_a_candidates"]) - 1)])
print("\nExample candidate from Config B:")
print(candidates["config_b_candidates"][min(49, len(candidates["config_b_candidates"]) - 1)])

# %% [markdown]
# After performing vector retrieval we need to fine our search by performing a reranking over the best results that we got so that they can be ordered over certain criterias

# %%
#as a model we will choose cross-encoder/ms-marco-MiniLM-L-6-v2, which is fine-tuned for relevance ranking tasks and should perform well in distinguishing subtle differences in prompt relevance.
from sentence_transformers import SentenceTransformer, util, CrossEncoder
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
print(f"Reranker model loaded: {RERANKER_MODEL_NAME}")

# %% [markdown]
# In order to improve the reranker and make sure that it might perform also when the prompt is vague or unclear we want to incorporate in a structured textual representation also other attributes, like title, category and tags, so that it would improve our retrieval. This ensures also a coherence with "Config B" where said attributes where already taken into account for embedding.
#

# %% [markdown]
# Now time for the reranker function, that takes the "rough" candidates resulting from the vector search and reorders them using a stronger model, returning only the top k results.
#

# %%
def rerank_candidates(query, candidates, top_k=10):
    query = query.strip() #remove leading/trailing whitespace
    candidate_texts = [candidate.get("content") for candidate in candidates] #retrieve the text representation for each candidate
    pairs = [[query, ct] for ct in candidate_texts] #prepare the input pairs for the cross-encoder (query + candidate text)
    scores = reranker_model.predict(pairs) #cross-encoder returns relevance scores for each pair (higher = more relevant)
    reranked = [{"candidate": c, "rerank_score": float(s)} for c, s in zip(candidates, scores)] #attach scores to candidates
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k] #sort candidates by rerank_score and return top_k


# %% [markdown]
# Testing on candidates from Config A:

# %%
reranked_a = rerank_candidates(QUERY, candidates["config_a_candidates"], top_k=5)

print("\nTop 5 reranked candidates from Config A:")
for i, item in enumerate(reranked_a, start=1):
    c = item["candidate"]
    score = item["rerank_score"]
    print(f"\n {i}  ID: {c['id']}  Rerank Score: {score:.4f}")
    print(f" Title: {c['title']}")
    print(f"content:{c['content']}")

# %% [markdown]
# Now on Candidates from B

# %%
# And now on the candidates from Config B
reranked_b = rerank_candidates(QUERY, candidates["config_b_candidates"], top_k=5)


print("\nTop 5 reranked candidates from Config B:")
for i, item in enumerate(reranked_b, start=1):
    c = item["candidate"]
    score = item["rerank_score"]
    print(f"\n {i}  ID: {c['id']}  Rerank Score: {score:.4f}")
    print(f" Title: {c['title']}")
    print(f"content:{c['content']}")


# %% [markdown]
# Now we want to analyze the reranker performance compared to the vector search for the top 10 among the 50 best ones found by vector search itself. As evaluation metrics we use NDCG, RR and P@10.

