import os
import sys
import argparse

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from retrieval.retriever import benchmark_data_augmentation_call

def main():
    parser = argparse.ArgumentParser(description="Run RAG4RE benchmark pipeline.")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config file. Relative paths resolve from the src directory.",
    )
    args = parser.parse_args()

    config_file_path = args.config
    benchmark_data_augmentation_call(config_file_path)
if __name__ == "__main__":
    main()