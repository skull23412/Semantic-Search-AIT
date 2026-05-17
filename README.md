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

-`reranker.py`:to refine the original retrieval done by vector search

-`reranker_metadata_enriched.py`:to refine the original retrieval done by vector search and by the original reranker

-`streamlit_demo.py`:


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

### Execution Order

Run the scripts in the following order:

1.`eda_leaf.py`

2.`preprocessing.py`

3.`embedding.py`

4.`retrieval.py`

5.`reranker.py`

6.`reranker_metadata_enriched.py`

| Step | Script | Main input | Main output |
|---|---|---|---|
| 1 | `eda_leaf.py` | `dataset.json` | Descriptive statistics and EDA results |
| 2 | `preprocessing.py` | `dataset.json` | `embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`retrieval_metadata_enriched.csv` |
| 3 | `embedding.py` | `embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`retrieval_metadata_enriched.csv` | `config_a_bge_collection`<br>`config_b_e5_collection` |
| 4 | `retrieval.py` | `retrieval_metadata_enriched.csv`<br>`embeddings_bge_enriched.npy`<br>`embeddings_e5_enriched.npy`<br>`config_a_bge_collection`<br>`config_b_e5_collection` | `candidates_for_reranker.json` |
| 5 | `reranker.py` | `candidates_for_reranker.json`<br>`retrieval_metadata_enriched.csv` | `reranked_output_config_A.json`<br>`reranked_output_config_B.json` |
| 6 | `reranker_metadata_enriched.py` | `candidates_for_reranker.json`<br>`retrieval_metadata_enriched.csv` | `reranked_metadata_enriched_config_A.json`<br>`reranked_metadata_enriched_config_B.json` |

**Description of intermediate files:**

-`embeddings_bge_enriched.npy`: BGE numerical embeddings 

-`embeddings_e5_enriched.npy`: E5 numerical embeddings  

-`retrieval_metadata_enriched.csv`: metadata linked to each prompt and used for ChromaDB and reranking  

-`config_a_bge_collection`: local ChromaDB collection with BGE vectors and metadata  

-`config_b_e5_collection`: local ChromaDB collection with E5 vectors and metadata

-`candidates_for_reranker.json`: ranked retrieval candidates with metadata for Config A and Config B, used as input for reranking.

-`Reranked_output_config_A.json`: The top 5 prompts for a given query after the reranking process using conf A

-`Reranked_output_config_B.json`: The top 5 prompts for a given query after the reranking process using conf B

-`reranked_metadata_enriched_config_A.json`: The top 5 prompts for a given query after the reranking process using conf A considering also metadata

-`reranked_metadata_enriched_config_B.json`:The top 5 prompts for a given query after the reranking process using conf B considering also metadata

Each script should be executed only after the previous one has successfully completed, because some scripts use intermediate files generated in earlier stages.
Additionally it must be considered that some intermediate files listed above are generated automatically during the execution of the pipeline and therefore are not included in the `src` zip folder.
For this reason before executing each of the file update the file paths in the scripts if necessary, so that they match the local location of the dataset and the output folders on your machine.
