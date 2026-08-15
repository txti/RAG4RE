import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from numpy.linalg import norm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]


import configparser


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
        


def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings

    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings

    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """

    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_sentence = " ".join(train_line['tokens'])
                
            context =  train_sentence
            train_similarities.append({"train":train_index, "simscore": sim, "sentence":context})

        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    """Compute Consine similarity between test and train embeddings for semeval dataset
    Args:
        test_data (list): list of sentences
        train_data (list): list of sentences
        train_embeddings (list): list of sentence embeddings
        test_embeddings (list): list of sentence embeddings
    Returns:
        list: list of similarity scores along with similar sentence and train data index
    """
    
    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_similarities.append({"train":train_index, "simscore":sim, "sentence":train_line})
        
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset="semeval"):
    """Compute similarity between test and train embeddings"""

    test_data = read_json(test_file)
    train_data = read_json(train_file)

    train_embeddings = np.load(train_emb)
    test_embeddings = np.load(test_emb)

    if dataset == "semeval":
        similarities = semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings)
    else:
        similarities = compute_similarity(test_data, train_data, train_embeddings, test_embeddings)

    write_json(output_sim_path, similarities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute train/test sentence similarity scores.")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config file. Relative paths resolve from the src directory.",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config_path = resolve_config_path(args.config)
    config.read(config_path)

    test_file = resolve_path(config["SIMILARITY"]["test_file"])
    train_file = resolve_path(config["SIMILARITY"]["train_file"])
    train_emb = resolve_path(config["SIMILARITY"]["train_emb"])
    test_emb = resolve_path(config["SIMILARITY"]["test_emb"])
    output_sim_path = resolve_path(config["SIMILARITY"]["output_index"])
    dataset = config["SETTINGS"].get("dataset", "semeval")

    main(str(test_file), str(train_file), str(train_emb), str(test_emb), str(output_sim_path), dataset=dataset)