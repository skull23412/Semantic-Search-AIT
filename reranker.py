#reranker.py
#builds a function which reranks the candidates retrieved from the vector search using a stronger model, and then tests it on the candidates from both Config A and Config B,
#printing the top 5 reranked results for each configuration. The function takes the original query and the list of candidates,
#computes relevance scores using a cross-encoder model, and returns the top k reranked candidates based on those scores.
#Role: to refine the original retrieval done by vector search
#input: candidates_for_reranker_json
#output: Reranked_output_config_A.json , Reranked_output_config_B.json
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



if "config_a_bge" in globals() and "config_b_e5" in globals() and "metadata" in globals():
    print(config_a_bge.shape)
    print(config_b_e5.shape)
    print(metadata.shape)
else:
    print("Standalone mode: using candidates loaded from Step 3 JSON.")


if "config_a_chromadb" in globals() and "config_b_chromadb" in globals():
    print("Chroma A:", config_a_chromadb.count())
    print("Chroma B:", config_b_chromadb.count())
else:
    print("Standalone mode: ChromaDB collections are not needed for Step 4.")


candidates.keys()


print("\nExample candidate from Config A:")
print(candidates["config_a_candidates"][min(49, len(candidates["config_a_candidates"]) - 1)])
print("\nExample candidate from Config B:")
print(candidates["config_b_candidates"][min(49, len(candidates["config_b_candidates"]) - 1)])


# After performing vector retrieval we need to fine our search by performing a reranking over the best results that we got so that they can be ordered over certain criterias


#as a model we will choose cross-encoder/ms-marco-MiniLM-L-6-v2, which is fine-tuned for relevance ranking tasks and should perform well in distinguishing subtle differences in prompt relevance.
from sentence_transformers import SentenceTransformer, util, CrossEncoder
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
print(f"Reranker model loaded: {RERANKER_MODEL_NAME}")



def rerank_candidates(query, candidates, top_k=10):
    query = query.strip() #remove leading/trailing whitespace
    candidate_texts = [candidate.get("content") for candidate in candidates] #retrieve the text representation for each candidate
    pairs = [[query, ct] for ct in candidate_texts] #prepare the input pairs for the cross-encoder (query + candidate text)
    scores = reranker_model.predict(pairs) #cross-encoder returns relevance scores for each pair (higher = more relevant)
    reranked = [{"candidate": c, "rerank_score": float(s)} for c, s in zip(candidates, scores)] #attach scores to candidates
    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k] #sort candidates by rerank_score and return top_k




reranked_a = rerank_candidates(QUERY, candidates["config_a_candidates"], top_k=5)

print("\nTop 5 reranked candidates from Config A:")
for i, item in enumerate(reranked_a, start=1):
    c = item["candidate"]
    score = item["rerank_score"]
    print(f"\n {i}  ID: {c['id']}  Rerank Score: {score:.4f}")
    print(f" Title: {c['title']}")
    print(f"content:{c['content']}")



reranked_b = rerank_candidates(QUERY, candidates["config_b_candidates"], top_k=5)


print("\nTop 5 reranked candidates from Config B:")
for i, item in enumerate(reranked_b, start=1):
    c = item["candidate"]
    score = item["rerank_score"]
    print(f"\n {i}  ID: {c['id']}  Rerank Score: {score:.4f}")
    print(f" Title: {c['title']}")
    print(f"content:{c['content']}")



with open("Reranked_output_config_A.json", "w", encoding="utf-8") as f:
    json.dump(reranked_a, f, ensure_ascii=False, indent=4)

with open("Reranked_output_config_B.json", "w", encoding="utf-8") as f:
    json.dump(reranked_b, f, ensure_ascii=False, indent=4)