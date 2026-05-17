# preprocessing.py
# Builds an enriched natural language representation per prompt by
# combining content, category, subcategory, tags, and title into one string
# and tokenizes it, generating dense vectors with BGE and E5.
# Role: produces the text that will be fed to the embedding models.
# Input:  df (pandas DataFrame of raw prompt records)
# Output: embeddings_bge_enriched.npy, embeddings_e5_enriched.npy,
# retrieval_metadata_enriched.csv

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

df = pd.read_json("/content/dataset.json")

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Cap tags at 4 to keep the enriched string compact and avoid long tag lists
# dominating the embedding input over the actual prompt content.
def format_tags(tags, max_tags=4):
    cleaned_tags = [clean_text(tag) for tag in tags if clean_text(tag)]
    cleaned_tags = cleaned_tags[:max_tags]

    if len(cleaned_tags) == 0:
        return ""
    elif len(cleaned_tags) == 1:
        return cleaned_tags[0]
    else:
        return ", ".join(cleaned_tags[:-1]) + f", and {cleaned_tags[-1]}"

# Verify the main semantic fields are not empty.
for col in ["content", "category", "subcategory", "title"]:
    print(col, (df[col].astype(str).str.strip() == "").sum())

print("empty tag lists:", df["tags"].apply(lambda x: len(x) == 0).sum())

# Template formats each record as a sentence rather than a concatenation, so
# embedding models receive natural text.
# content remains primary while category, subcategory, tags and title follow.
# Tag sentence is skipped if there are no tags.
def build_embedding_text(row):
    tags = format_tags(row["tags"], max_tags=4)

    sentences = [
        f"The task is: {clean_text(row['content'])}.",
        f"This prompt belongs to the {clean_text(row['category'])} category and the {clean_text(row['subcategory'])} subcategory."
    ]

    if tags:
        sentences.append(f"It is related to {tags}.")

    sentences.append(f'The prompt title is "{clean_text(row["title"])}".')

    return " ".join(sentences)


df["embedding_text"] = df.apply(build_embedding_text, axis=1)

for i in [0, 10, 25, 100, 250]:
    print("ID:", df.loc[i, "id"])
    print(df.loc[i, "embedding_text"])

# Sanity check before tokenization: combining fields could push inputs past
# maximum model sequence length and cause truncation.
df["embedding_text_len_words"] = df["embedding_text"].str.split().str.len()

print("ENRICHED INPUT LENGTH STATS")
print(df[["embedding_text_len_words"]].describe())

print("\n WORD LENGTH PERCENTILES")
print(df["embedding_text_len_words"].quantile([0.90, 0.95, 0.99]))

# MiniLM as a lightweight baseline. Checking truncation rate here
# tells us if its token limit is acceptable for the enriched text.
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model max sequence length:", model.max_seq_length)

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=True))

df["embedding_text_tokens_minilm"] = df["embedding_text"].apply(count_tokens)

print("TOKEN LENGTH STATS - MiniLM")
print(df[["embedding_text_tokens_minilm"]].describe())

above_limit = (df["embedding_text_tokens_minilm"] > model.max_seq_length).sum()
print("\nInputs above MiniLM limit:", above_limit)
print("Percentage above MiniLM limit:", above_limit / len(df) * 100)

# E5 is trained with query-passage prefixes, "passage:" is required in the
# corpus so that the model correctly stores vectors.
# BGE needs no prefix, so it uses the raw text.
df["embedding_text_e5"] = "passage: " + df["embedding_text"]

def count_token_lengths(model_name, text_column):
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    readable_model_name = (
        model_name
        .split("/")[-1]
        .replace("-", "_")
        .replace(".", "_")
    )

    token_length_column = f"tokens_{readable_model_name}"

    def count_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=True))

    df[token_length_column] = df[text_column].apply(count_tokens)


    texts_above_limit = (
        df[token_length_column] > model.max_seq_length
    ).sum()

    percentage_above_limit = texts_above_limit / len(df) * 100

    print("Model:", model_name)
    print("Model max sequence length:", model.max_seq_length)
    print(df[[token_length_column]].describe())
    print("\nInputs above model limit:", texts_above_limit)
    print("Percentage above model limit:", percentage_above_limit)

    return token_length_column


e5_token_column = count_token_lengths(
    model_name="intfloat/e5-base-v2",
    text_column="embedding_text_e5"
)

bge_token_column = count_token_lengths(
    model_name="BAAI/bge-base-en-v1.5",
    text_column="embedding_text"
)

bge_model_name = "BAAI/bge-base-en-v1.5"

bge_model = SentenceTransformer(bge_model_name)

embeddings_bge = bge_model.encode(
    df["embedding_text"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("BGE embeddings shape:", embeddings_bge.shape)

# "passage:" prefix for corpus encoding. Queries will later use "query:"
df["embedding_text_e5"] = "passage: " + df["embedding_text"]

e5_model_name = "intfloat/e5-base-v2"

e5_model = SentenceTransformer(e5_model_name)

embeddings_e5 = e5_model.encode(
    df["embedding_text_e5"].tolist(),
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("E5 embeddings shape:", embeddings_e5.shape)

# Save embeddings and metadata for later retrieval
np.save("embeddings_bge_enriched.npy", embeddings_bge)
np.save("embeddings_e5_enriched.npy", embeddings_e5)

retrieval_metadata_cols = [
    "id",
    "likes",
    "upvotes",
    "title",
    "content",
    "category",
    "subcategory",
    "tags",
    "difficulty",
    "language",
    "target_model",
    "embedding_text"
]

df[retrieval_metadata_cols].to_csv("retrieval_metadata_enriched.csv", index=False)
