"""Command line interface for the RAG4RE package."""

import argparse

from data_augmentation.embeddings.sentence_embeddings import run_from_config as run_embeddings_from_config
from data_augmentation.embeddings.sentence_sim import run_from_config as run_similarity_from_config
from evaluation.results_analysis import run_from_config as run_evaluation_from_config
from retrieval.retriever import benchmark_data_augmentation_call


def build_parser():
    parser = argparse.ArgumentParser(description="RAG4RE package CLI")
    parser.add_argument(
        "--config",
        default="src/config.ini",
        help="Path to config file.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("pipeline", help="Run retrieval + generation pipeline")
    subparsers.add_parser("embed", help="Compute sentence embeddings")
    subparsers.add_parser("similarity", help="Compute sentence similarity index")
    subparsers.add_parser("evaluate", help="Compute metrics and error analysis")
    subparsers.add_parser("all", help="Run embed, similarity, pipeline, and evaluate")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    command = args.command or "pipeline"
    config_path = args.config

    if command == "embed":
        run_embeddings_from_config(config_path)
        return

    if command == "similarity":
        run_similarity_from_config(config_path)
        return

    if command == "pipeline":
        benchmark_data_augmentation_call(config_path)
        return

    if command == "evaluate":
        run_evaluation_from_config(config_path)
        return

    if command == "all":
        run_embeddings_from_config(config_path)
        run_similarity_from_config(config_path)
        benchmark_data_augmentation_call(config_path)
        run_evaluation_from_config(config_path)
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
