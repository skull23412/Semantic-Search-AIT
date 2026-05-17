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

-

-

-

-

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

## Input Data

The initial input file is:

```text
LEAF-promptkaban-dataset/dataset.json
```

## Data Flow

The project starts using the raw dataset stored in `LEAF-promptkaban-dataset/dataset.json`.

During the execution of the pipeline, some scripts generate intermediate files that are used as input by later scripts.
Therefore, it is important to execute files in the specified order to correctly reproduce the right workflow.





## Execution Order

Run the scripts in the following order:

1.`eda_leaf.py`

2.`preprocessing.py`

3.`embedding.py`

4.
