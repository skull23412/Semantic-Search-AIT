# -*- coding: utf-8 -*-
"""
 EDA_LEAF.py
Performs the exploratory data analysis of the LEAF prompt dataset.
The analysis is used to understand the dataset structure, quality, and semantic characteristics
before making embedding-related design choices.
Role: supports the design of the semantic retrieval pipeline by identifying which fields
should be used for the  embedding representationand which fields should be treated as metadata.
Input: LEAF-promptkaban-dataset/dataset.json
Output: descriptive statistics and EDA results used to guide preprocessing and embedding design.

"""

import json
import pandas as pd
import numpy as np
from collections import Counter
import re

INPUT_FILE = r"LEAF-promptkaban-dataset\dataset.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total records:", len(data))
print("Data structure type:", type(data))
print("First record keys:", list(data[0].keys()))



##Dataset overview


df = pd.DataFrame(data)
print("Dataset shape:", df.shape)
df.head(3)
print("DATA TYPES:")
print(df.dtypes)
print()

print("MISSING VALUES:")
print(df.isna().sum().sort_values(ascending=False))



#Core distributions for embeddings
# This section examines the main categorical distributions of the dataset.
# The goal is to understand which fields are most relevant for the embedding strategy,
# especially language, category, subcategory, difficulty level, and placeholder presence.

print("LANGUAGE DISTRIBUTION:")
print(df["language"].value_counts())
print()

print("TOP 20 CATEGORIES:")
print(df["category"].value_counts().head(20))
print()

print("TOP 20 SUBCATEGORIES:")
print(df["subcategory"].value_counts().head(20))
print()

print("DIFFICULTY DISTRIBUTION:")
print(df["difficulty"].value_counts())
print()

print("HAS_PLACEHOLDERS DISTRIBUTION:")
print(df["has_placeholders"].value_counts())


#Text length analysis

# This block computes the length of title and content,
# both in characters and in words. The goal is to compare how much semantic
# information each field carries and to verify whether content should be treated
# as the main textual component of the embedding representation.

df["title_len_chars"] = df["title"].fillna("").astype(str).str.len()
df["content_len_chars"] = df["content"].fillna("").astype(str).str.len()

df["title_len_words"] = df["title"].fillna("").astype(str).str.split().str.len()
df["content_len_words"] = df["content"].fillna("").astype(str).str.split().str.len()

print("TEXT LENGTH STATS:")
print(df[[
    "title_len_chars", "content_len_chars",
    "title_len_words", "content_len_words"
]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))


#Title quality / noise check
# This section evaluates whether titles are sufficiently informative for embedding.
# We consider as less informative titles that are very short, mostly written in uppercase,
# or contain informal expressions.


def is_very_short(text, threshold=4):
    return len(str(text).split()) <= threshold

def has_all_caps(text):
    s = str(text)
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.7

def has_informal_style(text):
    s = str(text).lower()
    patterns = [
        r"\bidk\b", r"\bwats\b", r"\bpls\b", r"\bwut\b", r"\bim\b",
        r"\bcuz\b", r"\btho\b", r"\bgimme\b", r"\babt\b", r"\blol\b"
    ]
    return any(re.search(p, s) for p in patterns)

df["title_very_short"] = df["title"].apply(is_very_short)
df["title_all_caps_style"] = df["title"].apply(has_all_caps)
df["title_informal_style"] = df["title"].apply(has_informal_style)

print("TITLE NOISE CHECK:")
print("Very short titles (%):", round(df["title_very_short"].mean() * 100, 2))
print("All-caps style titles (%):", round(df["title_all_caps_style"].mean() * 100, 2))
print("Informal-style titles (%):", round(df["title_informal_style"].mean() * 100, 2))

# Main finding:
# A large share of titles are very short, while all-caps and informal styles are less frequent.
# This suggests that titles are useful as short contextual cues, but they should not be treated
# as the main source of semantic information.

#Placeholder analysis
# We counts how many placeholders appear in each prompt and then we identify
# the most frequent placeholder types across the dataset.
# It is important to understand whether placeholders are rare noise or meaningful
# structural elements that should be preserved during preprocessing.

df["n_placeholders"] = df["placeholders"].apply(lambda x: len(x) if isinstance(x, list) else 0)

print("PLACEHOLDER STATS:")
print(df["n_placeholders"].describe())
print()

all_placeholders = Counter()
for row in df["placeholders"]:
    if isinstance(row, list):
        all_placeholders.update(row)

print("TOP 30 PLACEHOLDERS:")
for ph, cnt in all_placeholders.most_common(30):
    print(f"{ph}: {cnt}")


#Tag analysis
# This section counts how many tags are associated with each prompt and identifies
# the most frequent tags in the dataset.
# The goal is to assess whether tags provide useful semantic descriptors that can
# complement the main prompt content in the embedding representation.

df["n_tags"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

print("TAG COUNT STATS:")
print(df["n_tags"].describe())
print()

all_tags = Counter()
for row in df["tags"]:
    if isinstance(row, list):
        all_tags.update(row)

print("TOP 50 TAGS:")
for tag, cnt in all_tags.most_common(50):
    print(f"{tag}: {cnt}")


#Analysis of TOP 5 shortes and longest promts

print("5 SHORTEST PROMPTS BY WORD COUNT:")
shortest = df.sort_values("content_len_words").head(5)
for _, row in shortest.iterrows():
    print("\nID:", row["id"])
    print("Title:", row["title"])
    print("Content:", row["content"])
    print("Category:", row["category"], "| Subcategory:", row["subcategory"])


print("\n 5 LONGEST PROMPTS BY WORD COUNT:")
longest = df.sort_values("content_len_words", ascending=False).head(5)
for _, row in longest.iterrows():
    print("\nID:", row["id"])
    print("Title:", row["title"])
    print("Content preview:", row["content"][:1200])
    print("Category:", row["category"], "| Subcategory:", row["subcategory"])


#Shortest prompts by word count:


#The shortest prompts illustrate a critical challenge for the embedding stage: some records contain extremely 
#limited lexical information, such as one-word or near one-word requests. In these cases, 
#semantic meaning is only partially expressed in the content field and is more likely to emerge from 
#the combination of title, category, subcategory, and tags. These examples show why an embedding strategy 
#based on content alone may underperform on a non-trivial subset of the datase

#Longest prompts by word count:

#The longest prompts resemble full user problem descriptions rather than compact prompt templates. 
#They contain narrative context, uncertainty, and implicit goals, which provide rich semantic material for dense representations. 
#These cases are likely to benefit strongly from modern sentence-level or text-level embeddings, 
#since the main challenge is not lexical sparsity but preserving topical and functional coherence across a longer sequence


#Divios of semantic and semantic type of data
# This section separates the variables that directly contribute to prompt meaning
# from those that describe popularity, usage, technical properties, or filtering information.
# The reason behind this division is deciding which fields should be used in the embedding representation
# and which fields should be preserved for later filtering, interpretation, or reranking.

semantic_fields = [
    "title", "content", "category", "subcategory", "tags", "placeholders", "language"
]

metadata_fields = [
    "author_reputation", "version", "fork_count", "likes", "upvotes", "downvotes",
    "views", "uses", "created_at", "has_placeholders", "difficulty", "target_model"
]

print("SEMANTIC FIELDS:")
for c in semantic_fields:
    print("-", c)

print("\n METADATA / RANKING / FILTERING FIELDS:")
for c in metadata_fields:
    print("-", c)