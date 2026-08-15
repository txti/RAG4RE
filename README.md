# RAG4RE

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Transformers](https://img.shields.io/badge/Transformers-4.38.2-yellow)](https://github.com/huggingface/transformers)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-2.2.2-orange)](https://www.sbert.net/)
[![GitHub last commit](https://img.shields.io/github/last-commit/sefeoglu/RAG4RE)](https://github.com/sefeoglu/RAG4RE/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/sefeoglu/RAG4RE)](https://github.com/sefeoglu/RAG4RE/issues)
[![GitHub repo size](https://img.shields.io/github/repo-size/sefeoglu/RAG4RE)](https://github.com/sefeoglu/RAG4RE)

Implementation for the paper: **Retrieval-Augmented Generation-Based Relation Extraction**.

The project provides an end-to-end pipeline for relation extraction with and without retrieval augmentation across datasets such as TACRED, TACREV, Re-TACRED, and SemEval.

## Citation

```bibtex
@article{doi:10.1177/22104968251385519,
  author = {Sefika Efeoglu and Adrian Paschke},
  title = {Retrieval-Augmented Generation-Based Relation Extraction},
  journal = {Semantic Web},
  volume = {16},
  number = {5},
  pages = {22104968251385519},
  year = {2025},
  doi = {10.1177/22104968251385519},
  url = {https://doi.org/10.1177/22104968251385519}
}
```

## Dataset Notes

- TACRED is licensed by LDC and must be obtained from [LDC2018T24](https://catalog.ldc.upenn.edu/LDC2018T24).
- TACREV is constructed from TACRED using [DFKI-NLP/tacrev](https://github.com/DFKI-NLP/tacrev).
- Re-TACRED is derived from TACRED using [gstoica27/Re-TACRED](https://github.com/gstoica27/Re-TACRED).
- SemEval 2010 Task 8 is available on [Hugging Face](https://huggingface.co/datasets/sem_eval_2010_task_8).

Because TACRED is restricted, prompts/raw outputs that expose original text are not directly redistributed.

## Project Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── data/
├── results/
└── src/
    ├── config.ini
    ├── main.py
    ├── utils.py
    ├── data_augmentation/
    │   ├── embeddings/
    │   └── prompt_generation/
    ├── data_preparation/
    ├── evaluation/
    │   └── results_analysis.py
    ├── generation_module/
    │   └── generation.py
    └── retrieval/
        ├── refinement.py
        └── retriever.py
```

## Setup

1. Install as a package (editable mode for development).

```bash
pip install -e .
```

Optional: if you prefer plain requirements installation instead of packaging:

```bash
pip install -r requirements.txt
```

2. Review and update experiment settings in `src/config.ini`.

- Paths in the config are project-relative.
- Choose dataset, prompt type (`simple` or `rag`), and model.

## Run

1. Generate sentence embeddings.

```bash
rag4re --config src/config.ini embed
```

2. Compute retrieval similarity index.

```bash
rag4re --config src/config.ini similarity
```

3. Run generation pipeline.

```bash
rag4re --config src/config.ini pipeline
```

4. Run evaluation.

```bash
rag4re --config src/config.ini evaluate
```

5. Run the complete workflow.

```bash
rag4re --config src/config.ini all
```

## Environment

The experiments were run on NVIDIA GeForce GTX 1080 Ti GPUs (4 x 12GB) with large CPU memory availability.

