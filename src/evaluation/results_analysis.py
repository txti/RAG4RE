import os
import sys
import argparse
from pathlib import Path
from sklearn.metrics import  precision_recall_fscore_support
import configparser
import numpy as np
import pandas as pd
from utils import read_json, write_json
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]


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


def error_analysis(ground_truths, preds, labels):
    """False predicitons analysis

    Args:
        ground_truths (list): ground truth labels
        preds (list): predicted labels
        labels (list): target labels

    Returns:
        int, int: false positives and false negatives
    """
    tp, fp, fn = 0, 0, 0
    
    if type(preds[0]) == dict:
        preds = [pred.values()for pred in preds]
        preds = list(preds[0])
    # print("preds", type(preds))
    ground_truths = [ground.split(" ")[-1].split(":")[-1].strip() for ground in ground_truths]
    preds = [pred.split(":")[-1].strip() for pred in preds]
    for i, truth in enumerate(ground_truths):
        # print("truth", preds[i])

        if truth == preds[i]:
            tp += 1
        elif preds[i] in labels:
            fp += 1
        else:
            fn += 1

    return fp, fn

def get_results(preds, grounds, targets):
    """Compute precision, recall and f1 scores"""

    if type(preds[0]) == dict:
        preds = [pred.values() for pred in preds]
        preds = list(preds[0])
    # print("preds", preds)
    preds = [pred.split(":")[-1].strip() for pred in preds]
    
    grounds = [ground.split(" ")[-1].split(":")[-1].strip() for ground in grounds]
    prec, recall, f1, s = precision_recall_fscore_support(grounds, preds, labels=targets, average='micro')
       
    return prec, recall, f1

def compute_scores(predictions, ground_truths, labels):
    """Compute precision, recall and f1 scores"""
    labels = list(set(labels))
    prec, recall, f1 = get_results(predictions, ground_truths, labels)
    
    return prec, recall, f1, predictions


def run_from_config(config_file_path="config.ini"):
    """Run evaluation from a config file."""
    config = configparser.ConfigParser()
    config_path = resolve_config_path(config_file_path)
    config.read(config_path)

    prompt_type = config["SETTINGS"]["prompt_type"]
    print(prompt_type)
    if prompt_type == "rag":
        prediction_path = resolve_path(config["OUTPUT"]["rag_test_responses_path"])
        result_path = resolve_path(config["RESULTS"]["rag_test_prompt_path"])
        error_path = resolve_path(config["RESULTS"]["rag_test_error_analysis_path"])
    else:
        prediction_path = resolve_path(config["OUTPUT"]["simple_prompt_responses_path"])
        result_path = resolve_path(config["RESULTS"]["simple_prompt_results_path"])
        error_path = resolve_path(config["RESULTS"]["simple_prompt_error_analysis_path"])

    ground_truths_path = resolve_path(config["PATH"]["test_ground_truth_path"])
    labels_path = resolve_path(config["PATH"]["relations_path"])

    labels = read_json(str(labels_path))
    predictions = read_json(str(prediction_path))
    ground_truths = read_json(str(ground_truths_path)).values()
    if config["SETTINGS"]['dataset'] == "semeval":
        print(labels)
        labels = labels["relation"]['names']
        ground_truths = [labels[id] for id in ground_truths]
    else:
        predictions = [tt.replace(" ","_").lower() for tt in predictions]
        labels = labels.keys()

    prec, recall, f1, preds = compute_scores(predictions, ground_truths, labels)

    result_metrics = {
        "Precision": [prec],
        "Recall": [recall],
        "F1": [f1],
    }

    result_df = pd.DataFrame(result_metrics)
    result_df.to_json(str(result_path))

    fp, fn = error_analysis(ground_truths, preds, labels)
    analysis = {
        "False Positives": [fp],
        "False Negatives": [fn],
    }
    error_df = pd.DataFrame(analysis)
    error_df.to_json(str(error_path))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for generated responses.")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="Path to config file. Relative paths resolve from the src directory.",
    )
    args = parser.parse_args()

    run_from_config(args.config)
