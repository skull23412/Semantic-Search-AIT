#reranker_metadata_enriched.py
#an improved version of the original reranker which incorporates metadata into the reranking process. It builds a function that combines relevance scores 
# from a cross-encoder model with metadata-based features such as popularity (likes and upvotes) and keyword matching using TF-IDF. The final reranking is based on a weighted combination of these factors, 
# allowing for a more comprehensive evaluation of candidate relevance. 
#Role: to refine the original retrieval done by vector search and by the original reranker
#input: candidates_for_reranker.json, retireval_metadata_enriched.csv
#output: reranked_metadata_enriched_config_A.json , reranked_metadata_enriched_config_B.json

import json
from pathlib import Path
from reranker import  reranker_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#first of all let's load the candidates that we got from the retrieval step, which are stored in a json file
CANDIDATES_PATH = "candidates_for_reranker.json"
METADATA_PATH =  "retrieval_metadata_enriched.csv"
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

path = CANDIDATES_PATH
metadata = pd.read_csv(METADATA_PATH)
QUERY, candidates = load_step3_candidates(path)


#normalization function to scale scores between 0 and 1, which is important for combining different types of scores in a meaningful way.
def minmax_normalize(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return series.apply(lambda x: 0.0)

    return (series - min_val) / (max_val - min_val)
# this function builds a textual representation of the textual metadata of a candidate, which will be used for the metadata-based relevance scoring. 
def build_metadata_text(candidate):
    return " ".join([str(candidate.get("title", "")),
        str(candidate.get("category", "")),
        str(candidate.get("subcategory",  "")),
        str(candidate.get("tags", "")),
        str(candidate.get("difficulty", ""))])
#weighted reranker function that finds similarity scores for both content and metadata, normalizes them, and combines them using specified weights.
def weighted_reranker(query, candidates, metadata_weight=0.4, content_weight=0.6):
    query = query.strip() 
    content_pairs= [[query, str(c.get("content", ""))] for c in candidates] #prepare input pairs for content relevance scoring
    metadata_pairs = [[query, build_metadata_text(c)] for c in candidates] #prepare input pairs for metadata relevance scoring
    content_scores = minmax_normalize(reranker_model.predict(content_pairs)) #we compute content relevance scores and we normalize them
    metadata_scores = minmax_normalize(reranker_model.predict(metadata_pairs))#we compute metadata relevance scores and we normalize them
    combined_scores = metadata_weight * metadata_scores + content_weight * content_scores
    weighted_reranker_scores=[]
    for candidate, combined_score in zip(candidates, combined_scores):
        weighted_reranker_scores.append({"candidate": candidate, "weighted_rerank_score": float(combined_score)})
    return sorted(weighted_reranker_scores, key=lambda x: x["weighted_rerank_score"], reverse=True)




# we fit a TF-IDF vectorizer on the embedding_text field of the metadata, which contains a textual representation of the prompt content and its associated metadata
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
corpus_tfidf = tfidf_vectorizer.fit_transform(metadata["embedding_text"].astype(str).tolist())
corpus_texts = metadata["embedding_text"].astype(str).tolist()

# this function identifies the most relevant keywords for each candidate based on TF-IDF scores, which can provide insights into why certain candidates are more relevant to the query.
def tf_idf_keyword_identifier(query, candidates, top_n=10):
    candidate_texts = [" ".join([build_metadata_text(c),str(c.get("content", ""))])for c in candidates] #combine content and metadata for keyword extraction

    tfidf_matrix = tfidf_vectorizer.transform(candidate_texts) #we transform the candidate texts into TF-IDF vectors using the fitted vectorizer
    query_vector = tfidf_vectorizer.transform([query]) #we also transform the query into a TF-IDF vector using the same vectorizer to see keyword overlap between the query and candidates
    cosine_similarities = cosine_similarity(query_vector,tfidf_matrix).flatten() #we compute cosine similarity between the query vector and each candidate vector to get a relevance score based on keyword overlap
    feature_names = tfidf_vectorizer.get_feature_names_out() #we retrieve the feature names (keywords) from the vectorizer to identify which keywords are most relevant for a given candidate

    candidate_keywords = []
    keyword_details = []
    for i, candidate in enumerate(candidates): #for each candidate, we identify the top N keywords based on their TF-IDF scores and their contribution to the cosine similarity with the query. We also store details about these keywords for later analysis.
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]

        keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]

        candidate_keywords.append({"id": candidate.get("id", ""), "score": float(cosine_similarities[i]), "keywords": keywords})
        keyword_details.append({"id": candidate.get("id", ""), "title": candidate.get("title", ""), "tfidf_score": float(cosine_similarities[i]), "matched_keywords": keywords})
    return cosine_similarities.tolist(), candidate_keywords


def metadata_aware_reranker(query, candidates,top_k=10, alpha=0.75, beta=0.10, gamma=0.15): #finally we combine the weighted relevance scores with a popularity score (based on likes and upvotes) and a keyword relevance score (based on TF-IDF).
    weighted_reranker_results=weighted_reranker(query, candidates, metadata_weight=0.4, content_weight=0.6)
    reranked_scores=pd.Series([candidate["weighted_rerank_score"] for candidate in weighted_reranker_results])
    popularity_scores=pd.Series([(float(candidate["candidate"].get("likes", 0))) + (float(candidate["candidate"].get("upvotes", 0))) for candidate in weighted_reranker_results])
    weighted_candidates=[item["candidate"] for item in weighted_reranker_results]
    keyword_scores, keywords=tf_idf_keyword_identifier(query, weighted_candidates, top_n=10)

    final_scores=alpha*minmax_normalize(reranked_scores) + beta*minmax_normalize(popularity_scores) + gamma*minmax_normalize(pd.Series(keyword_scores))
    for i, item in enumerate(weighted_reranker_results):
        item["final_score"] = float(final_scores.iloc[i])
        item["tfidf_keywords"] = keywords[i]["keywords"]

    return sorted(weighted_reranker_results, key=lambda x: x["final_score"], reverse=True)[:top_k]


print("\nTop 5 candidates from Config A with metadata-aware reranking:")

final_reranked_a = metadata_aware_reranker( QUERY, candidates["config_a_candidates"], top_k=5, alpha=0.75, beta=0.10, gamma=0.15)

for i, item in enumerate(final_reranked_a, start=1):
    c = item["candidate"]
    score = item["final_score"]

    print(f"\n{i}  ID: {c['id']}  Final Score: {score:.4f}")
    print(f"Title: {c['title']}")
    print(f"Content: {c['content']}")
    print(f"TF-IDF keywords: {item['tfidf_keywords']}")
    print(f"number of likes: {c['likes']} ")
    print(f"number of upvotes: {c['upvotes']}")
    print(f"Dataset tags: {c['tags']}")

print("\nTop 5 candidates from Config B with metadata-aware reranking:")

final_reranked_b = metadata_aware_reranker( QUERY, candidates["config_b_candidates"], top_k=5, alpha=0.75, beta=0.10, gamma=0.15)

for i, item in enumerate(final_reranked_b, start=1):
    c = item["candidate"]
    score = item["final_score"]

    print(f"\n{i}  ID: {c['id']}  Final Score: {score:.4f}")
    print(f"Title: {c['title']}")
    print(f"Content: {c['content']}")
    print(f"TF-IDF keywords: {item['tfidf_keywords']}")
    print(f"number of likes: {c['likes']} ")
    print(f"number of upvotes: {c['upvotes']}")
    print(f"Dataset tags: {c['tags']}")

#we dump the full reranked lists for both configs into separate json files as the final output
with open("reranked_metadata_enriched_config_A.json", "w", encoding="utf-8") as f:
    json.dump(final_reranked_a, f, ensure_ascii=False, indent=4)

with open("reranked_metadata_enriched_config_B.json", "w", encoding="utf-8") as f:
    json.dump(final_reranked_b, f, ensure_ascii=False, indent=4)
