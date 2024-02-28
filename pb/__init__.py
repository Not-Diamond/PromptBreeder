import os
import json
import copy
import time
import logging
from rich import print
from typing import List, Union
from pathlib import Path

from mmengine.config import Config

from pb import gsm
from pb.types import EvolutionUnit, Population

import ipdb
from dotenv import load_dotenv

from .run_opencompass import run_opencompass, EvaluationConfig

load_dotenv()

EVALUATION_WORK_DIR = os.getenv("EVALUATION_WORK_DIR")
EVALUATION_DATA_DIR = os.getenv("EVALUATION_DATA_DIR")

logger = logging.getLogger(__name__)

gsm8k_examples = gsm.read_jsonl('pb/data/gsm.jsonl')


class ModelWrapper:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def generate(self, message: str):
        match self.model_name:
            case "gemini-pro":
                return self.model.generate_content(message).text
            case _:
                raise NotImplementedError(f"{self.model_name} not implemented yet.")


from pb.mutation_operators import mutate


def create_population(tp_set: List, mutator_set: List, problem_description: str) -> Population:
    """samples the mutation_prompts and thinking_styles and returns a 'Population' object.

    Args:
        'size' (int): the size of the population to create.
        'problem_description (D)' (str): the problem description we are optimizing for.
    """
    data = {
        'size': len(tp_set)*len(mutator_set),
        'age': 0,
        'problem_description' : problem_description,
        'elites' : [],
        'units': [EvolutionUnit(**{
            'T' : t, 
            'M' : m,
            'P' : '',
            'fitness' : 0,
            'history' : []
            }) for t in tp_set for m in mutator_set]
    }

    return Population(**data)

def init_run(population: Population, writer_model: ModelWrapper, target_model: str, dataset_name: str, loaded_cfg: Config, eval_args: EvaluationConfig, wandb_run):
    """ The first run of the population that consumes the prompt_description and 
    creates the first prompt_tasks.
    
    Args:
        population (Population): A population created by `create_population`.
    """

    start_time = time.time()

    results = []
    for unit in population.units:
        template= f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT:"
        response = writer_model.generate(template)
        results.append(response)

    end_time = time.time()

    logger.info(f"Prompt initialization done. {end_time - start_time}s")
    assert len(results) == population.size, "size of google response to population is mismatched"

    for i, item in enumerate(results):
        population.units[i].P = item

    _evaluate_fitness(population, loaded_cfg, eval_args, target_model, dataset_name, wandb_run)
    
    return population

def run_for_n(n: int, population: Population, writer_model: ModelWrapper, target_model: str, dataset_name: str, loaded_cfg: Config, eval_args: EvaluationConfig, wandb_run):
    """ Runs the genetic algorithm for n generations.
    """     
    for i in range(n):
        print(f"================== Population {i} ================== ")
        mutate(population, writer_model, target_model, dataset_name)
        print("done mutation")
        _evaluate_fitness(population, loaded_cfg, eval_args, target_model, dataset_name, wandb_run)
        print("done evaluation")

    return population

def modify_prompt_template(cfg: Config, new_template: str):
    assert len(cfg.datasets) == 1, "Only one dataset can be run."
    assert len(cfg.models) == 1, "Only one model can be run."

    cfg["datasets"][0]["infer_cfg"]["prompt_template"]["template"]["round"][0]["prompt"] = new_template
    if "ice_template" in cfg["datasets"][0]["infer_cfg"]:
        cfg["datasets"][0]["infer_cfg"]["ice_template"]["template"]["round"][0]["prompt"] = new_template
    return cfg

def dataset_prompt_obj(new_prompt: Union[str, None], model: str, dataset_name: str, loaded_cfg: Config, eval_cfg: EvaluationConfig, wandb_run):
    dataset_path = Path(EVALUATION_DATA_DIR) / "processed" / model / f"{dataset_name}.json"
    tmp_root = Path(EVALUATION_DATA_DIR) / "tmp" / model
    with open(dataset_path, "r") as fp:
        dataset = json.load(fp)

    sample_id = list(dataset.keys())[0]
    sample = dataset[sample_id]
    prompt_added = sample["prompt_added"]
    prompt = sample["components"]["prompt"]["prompt"]
    if prompt_added:
        original_prompt = ""
        if new_prompt is None:
            new_prompt = ""
    else:
        original_prompt = copy.deepcopy(prompt)
        if new_prompt is None:
            new_prompt = prompt

    if prompt_added:
        prompt_template = sample["prompt_template"]["template"]
        modified_prompt_template = prompt_template.replace("{prompt}", new_prompt)
        loaded_cfg = modify_prompt_template(loaded_cfg, modified_prompt_template)

        modified_dataset = {}
        for sample_id, sample in dataset.items():
            sample["components"]["prompt"]["prompt"] = new_prompt
            modified_dataset[sample_id] = sample

        tmp_file = tmp_root / f"{dataset_name}.json"
        with open(tmp_file, "w") as fp:
            json.dump(modified_dataset, fp)

        dir_time_str = run_opencompass(loaded_cfg, eval_cfg)
    else:
        modified_dataset = {}
        for sample_id, sample in dataset.items():
            sample["components"]["prompt"]["prompt"] = new_prompt
            modified_dataset[sample_id] = sample

        tmp_file = tmp_root / f"{dataset_name}.json"
        with open(tmp_file, "w") as fp:
            json.dump(modified_dataset, fp)

        dir_time_str = run_opencompass(loaded_cfg, eval_cfg)

    results_path = Path(EVALUATION_WORK_DIR) / dir_time_str / "training_data" / model / f"{dataset_name}.json"
    reward_model_root = Path(EVALUATION_WORK_DIR) / dir_time_str / "reward_model_data" / model
    reward_model_path = reward_model_root / f"{dataset_name}.json"
    if not os.path.exists(reward_model_root):
        os.makedirs(reward_model_root)

    with open(results_path, "r") as fp:
        results = json.load(fp)

    n_samples = 0
    score = 0
    reward_model_data = {}
    for sample_id, sample in results.items():
        new_prompt = sample["sample_details"]["components"]["prompt"]["prompt"]
        n_samples += 1
        if "score" in sample["result"]:
            score += sample["result"]["score"]
            metric = sample["result"]["metric"]
        else:
            for _, subresult in sample["result"].items():
                score += subresult["score"]

    avg_score = score / n_samples
    reward_model_data["training_data"] = results
    reward_model_data["prompt_opt"] = {
        "original_prompt": original_prompt,
        "new_prompt": new_prompt,
        "score": avg_score
    }

    with open(reward_model_path, "w") as fp:
        json.dump(reward_model_data, fp)

    prompt_table_root = Path(EVALUATION_WORK_DIR) / "prompt_table"
    prompt_table_path = prompt_table_root / f"prompt_table_{wandb_run.config['exec_datetime_str']}.json"
    if not os.path.exists(prompt_table_root):
        os.makedirs(prompt_table_root)

    if not os.path.exists(prompt_table_path):
        prompt_table = {"columns": ["dataset_name", "origin_prompt", "new_prompt", "score", "device:path"],
                        "data": [[dataset_name, original_prompt, new_prompt, avg_score, f"{wandb_run.config['device']}:{str(reward_model_path)}"],]}
    else:
        with open(prompt_table_path, "r") as fp:
            prompt_table = json.load(fp)
            prompt_table["data"].append([dataset_name, original_prompt, new_prompt, avg_score, f"{wandb_run.config['device']}:{str(reward_model_path)}"])

    with open(prompt_table_path, "w") as fp:
        json.dump(prompt_table, fp)

    return {"score": avg_score, "metric": metric, "results_path": results_path}

def _evaluate_fitness(population: Population, loaded_cfg, eval_args, target_model, dataset_name, wandb_run) -> Population:
    logger.info(f"Starting fitness evaluation...")
    start_time = time.time()

    elite_fitness = -1
    for unit in population.units:
        new_prompt = unit.P
        result = dataset_prompt_obj(new_prompt, target_model, dataset_name, loaded_cfg, eval_args, wandb_run)
        score = result["score"]
        unit.fitness = score
        if score > elite_fitness:
            current_elite = unit.model_copy()
            elite_fitness = score

        if score > wandb_run.summary["best_score"]:
            wandb_run.summary["best_score"] = elite_fitness
            wandb_run.summary["best_experiment_dir"] = str(result['results_path'])
            wandb_run.summary["best_prompt"] = new_prompt

    # append best unit of generation to the elites list.
    population.elites.append(current_elite)

    end_time = time.time()
    logger.info(f"Done fitness evaluation. {end_time - start_time}s")
    wandb_run.log({f"{result['metric']}": elite_fitness})
    return population
