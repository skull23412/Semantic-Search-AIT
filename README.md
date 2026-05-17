# **Source Code Guide**

This folder contains the source code used for the project **Titolo**.

The goal of the project is to build a semantic search pipeline for the LEAF prompt dataset, from which is possibile to retrive promts according to their semantic similarity and thier quality with a user query.
The code starts from the raw `dataset.json` file, provided by the company and follows the main stages of the pipeline: exploratory data analysis, preprocessing, embedding generation, vector database creation, retrieval, evaluation, and reranking.

This README explains how the code is organized, which input file is required, and in which order the scripts should be executed to reproduce the results.

## src_file Structure

The `src` folder contains the following files:

-`README.md`: explains how to run the code and reproduce the project pipeline

-`requirements.txt`: lists the external Python libraries required to run the code

-`eda_leaf.py`: performs exploratory data analysis on the LEAF prompt dataset

-`preprocessing.py`: produces the text that will be fed to the embedding models

-`embedding.py`: turns text into a searchable vector index for the retrieval stage

-`retrieval.py`:performs vector search given user input ranks semantically close prompts to be fed in the reranker

-`reranker.py`:

-`reranker_metadata_enriched.py`:

??

-`embeddings_bge_enriched.npy`: numerical embedding of the dataset performed using BGE

-`embeddings_e5_enriched.npy`: numerical embedding of the dataset performed using E5

-`retrivial_metadata_enriched.csv`:Stores semantic and numerical metadata used for ChromaDB insertion and reranking.

??

## Environment

The project was run using:Python 3.12.13

To install the external libraries used in the project, run:

```bash
pip install -r requirements.txt
```
The project also uses the following Python Standard Library modules:

-`json`

-`re`

-`collections`

-`time`

-`random`

-`pathlib`


## Input Data

The initial input file is:

```text
dataset.json
```

Before running the scripts, make sure that the dataset path used in the code matches the location of the file on your machine.

## Data Flow

The project starts using the raw dataset stored in `dataset.json`.

During the execution of the pipeline, some scripts generate intermediate files that are used as input by later scripts.
Therefore, it is important to execute files in the specified order to correctly reproduce the right workflow.

## Execution Order

Run the scripts in the following order.

| Step | Script | Main input | Main output | Purpose |
|---|---|---|---|---|
| 1 | `eda_leaf.py` | `dataset.json` | Descriptive statistics and EDA results | Explores the dataset structure, completeness, textual fields, tags, placeholders, and metadata. The results are used to guide preprocessing and embedding design. |
| 2 | `preprocessing.py` | `dataset.json` | `embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`retrieval_metadata_enriched.csv` | Builds the enriched textual representation, generates BGE and E5 embeddings, and saves the metadata used for ChromaDB insertion and reranking. |
| 3 | `embedding.py` | `embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`retrieval_metadata_enriched.csv` | `config_a_bge_collection`<br>`config_b_e5_collection` | Creates the local ChromaDB collections by storing each embedding vector together with its corresponding metadata. |
| 4 | `retrieval.py` | `retrieval_metadata_enriched.csv`<br>`embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`config_a_bge_collection`<br>`config_b_e5_collection` | `candidates_for_reranker.json` | Runs semantic retrieval using Config A and Config B, returning the top candidates to be passed to the reranking stage. |
| 5 | `reranker.py` | `candidates_for_reranker.json`<br>`retrieval_metadata_enriched.csv` | `reranked_output_config_A.json`<br>`reranked_output_config_B.json` | Applies the cross-encoder reranker to reorder the retrieved candidates based mainly on semantic relevance. |
| 6 | `reranker_metadata_enriched.py` | `candidates_for_reranker.json`<br>`retrieval_metadata_enriched.csv` | `reranked_metadata_enriched_config_A.json`<br>`reranked_metadata_enriched_config_B.json` | Applies the metadata-aware reranking strategy, combining semantic relevance with additional metadata-based signals. |

## Execution Order

Run the scripts in the following order:

1.`eda_leaf.py`

2.`preprocessing.py`

3.`embedding.py`

4.`retrieval.py`

5.`reranker.py`

6.`reranker_metadata_enriched.py`
