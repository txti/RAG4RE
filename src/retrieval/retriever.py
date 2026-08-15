import os
import sys
from pathlib import Path

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

try:
    from .refinement import postprocessing
except ImportError:
    from refinement import postprocessing
from data_augmentation.prompt_generation.prompt_generation import generate_prompts
from generation_module.generation import LLM
import configparser
from utils import read_json, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value):
    """Resolve relative paths from the project root and keep absolute paths intact."""
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

def benchmark_data_augmentation_call(config_file_path):
    """
    This function is used to benchmark the retrieval module.
    Args:
    config_file_path: str: Path to the config file.
    """
    config = configparser.ConfigParser()
    config_path = resolve_config_path(config_file_path)
    config.read(config_path)
    
    test_data_path = resolve_path(config["PATH"]["test_data_path"])
    similar_sentences_path = resolve_path(config["SIMILARITY"]["output_index"])
    relations_path = resolve_path(config["PATH"]["relations_path"])
    dataset = config["SETTINGS"]["dataset"]
    prompt_type = config["SETTINGS"]["prompt_type"]
    model_name = config["SETTINGS"]["model_name"]
    similar_sentences = read_json(similar_sentences_path)
    relations = read_json(relations_path)

    if dataset != "semeval":
        relations = relations.keys()
    else:
        relations = relations
    test_data = read_json(test_data_path)

    if prompt_type == "rag":
        # print("RAG")
        output_prompts_path = resolve_path(config["OUTPUT"]["rag_test_prompts_path"])
        output_responses_path = resolve_path(config["OUTPUT"]["rag_test_responses_path"])
        prompts = generate_prompts(test_data, relations, similar_sentences,  dataset, prompt_type)
    else:
        output_prompts_path = resolve_path(config["OUTPUT"]["simple_prompt_path"])
        output_responses_path = resolve_path(config["OUTPUT"]["simple_prompt_responses_path"])
        prompts = generate_prompts(test_data, relations, similar_sentences,  dataset, prompt_type)

    llm_instance = LLM(model_name)
    
    responses = []

    for prompt in prompts:
        prompt = prompt["prompt"]

        if not "t5" in model_name:
            prompt = """[INST]{prompt}[/INST] Answer:"""

        response = llm_instance.get_prediction(prompt)
        responses.append(response)

    responses = postprocessing(dataset, test_data, responses, relations, model_name)
    
    write_json(str(output_prompts_path), prompts)
    write_json(str(output_responses_path), responses)
    
