""" This script is used to compute the sentence embeddings for the sentences in the dataset."""
"""Created by: Sefika"""
import os
import sys
import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import configparser

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value):
    path_obj = Path(path_value).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def resolve_config_path(config_file_path):
    config_path = Path(config_file_path).expanduser()
    if config_path.is_absolute():
        return config_path

    src_relative = SRC_ROOT / config_path
    if src_relative.exists():
        return src_relative

    return PROJECT_ROOT / config_path

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def compute_sentence(data):
    """Compute the sentence embeddings for the sentences in the dataset
    Args:
        data (list): list of sentences
    Returns:
        list: list of sentence embeddings
    """
    sent_embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("The embeddings will be compted for {0} sentences".format(len(data)))

    for i, line in enumerate(data):
        sent = " ".join(line['tokens'])
        clean_sent = clean_sentence(sent)
        embeddings = model.encode(clean_sent)
        sent_embeddings.append(embeddings)
        print("Processed sentence: ", i)

    print("The embeddings were completed for {0} sentences".format(len(sent_embeddings)))

    return sent_embeddings

def clean_sentence(sent):
    """Clean the sentence from the entity tags"""
    sent = sent.replace("<e1>", "")
    sent = sent.replace("</e1>", "")
    sent = sent.replace("<e2>", "")
    sent = sent.replace("</e2>", "")

    return sent

def write_embeddings(embeddings, output_file):
    np.save(output_file, embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sentence embeddings.")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config file. Relative paths resolve from the src directory.",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_path = resolve_config_path(args.config)
    config.read(config_path)

    input_file = resolve_path(config["EMBEDDING"]["input_embedding_path"])
    output_file = resolve_path(config["EMBEDDING"]["output_embedding_path"])
    data = read_json(str(input_file))
    embeddings = compute_sentence(data)
    write_embeddings(embeddings, str(output_file))