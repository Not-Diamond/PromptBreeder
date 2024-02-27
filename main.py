import os
import wandb
import logging
import argparse
from datetime import datetime

from opencompass.utils.run import get_config_from_arg

from dotenv import load_dotenv
from rich import print

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from utils import bcolors

from pb import create_population, init_run, run_for_n, ModelWrapper
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
from pb.run_opencompass import EvaluationConfig

import ipdb

load_dotenv() # load environment variables
EVALUATION_WORK_DIR = os.getenv("EVALUATION_WORK_DIR")
EVALUATION_DATA_DIR = os.getenv("EVALUATION_DATA_DIR")


def get_prompt_writer():
    safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        }

    # model = genai.GenerativeModel("models/gemini-pro", safety_settings=safety_settings, generation_config=genai.types.GenerationConfig(temperature=0.))
    model = genai.GenerativeModel("models/gemini-pro", safety_settings=safety_settings)
    return model

def get_opencompass_cfg(eval_args: EvaluationConfig):
    assert eval_args.dump_eval_details, "Error: args.dump_eval_details must be set"
    if eval_args.dry_run:
        eval_args.debug = True
    cfg = get_config_from_arg(eval_args)
    return cfg

def dataset_loop(writer_model: ModelWrapper, target_model: str, study_configs: dict):
    logger = logging.getLogger(__name__)

    eval_args = EvaluationConfig(
        config="configs/update_config.py",
        dry_run=False,
        debug=False,
        work_dir=EVALUATION_WORK_DIR,
        mode="all",
        dump_eval_details=True,
        db_url=f"{EVALUATION_DATA_DIR}/tmp/{target_model}",
        size=study_configs["sample_size"],
        seed=1
    )
    loaded_cfg = get_opencompass_cfg(eval_args)
    model = [llm for llm in loaded_cfg.models if llm.abbr == target_model]
    assert len(model) == 1, f"model name is invalid, requested {model}, available models {[llm.abbr for llm in loaded_cfg.models]}"
    loaded_cfg.models = model
    model_name = model[0].abbr

    if study_configs["dataset"] is not None:
        dataset_cfg = [dataset for dataset in loaded_cfg.datasets if dataset.abbr == study_configs["dataset"]]
        loaded_cfg.datasets = dataset_cfg

    for i, dataset in enumerate(loaded_cfg.datasets):
        print(f"{bcolors.OKGREEN}Optimizing {dataset.abbr} on {model_name} ({i+1}/{len(loaded_cfg.datasets)}){bcolors.ENDC}")
        dataset_name = dataset.abbr

        with wandb.init(project="PromptBreeder", name=f"({study_configs['exec_datetime_str']}) {dataset_name}", job_type="train", config=study_configs) as wandb_run:
            logger.info(f'Creating the population...')
            _ = wandb_run.use_artifact("LLM Eval/training-dataset:v2")
            wandb_run.summary["best_score"] = -1.
            wandb_run.summary["best_experiment_dir"] = ""
            wandb_run.summary["prompt_table"] = []

            m_prompts = mutation_prompts[:study_configs["mut_prompts"]]
            t_styles = thinking_styles[:study_configs["think_sty"]]
            population = create_population(tp_set=m_prompts, mutator_set=t_styles, problem_description="Answer the following question")

            logger.info(f'Generating the initial prompts...')
            init_run(population, writer_model, target_model, dataset_name, loaded_cfg, eval_args, wandb_run)

            logger.info(f'Starting the genetic algorithm...')
            run_for_n(study_configs["epochs"], population, writer_model, target_model, dataset_name, loaded_cfg, eval_args, wandb_run)

def get_params():
    shell_parse = argparse.ArgumentParser()
    shell_parse.add_argument('--comment', type=str, default="", help='comments about the run')
    shell_parse.add_argument('--device', type=str, required=True, help='where was the experiment run? Helps to track down opencompass output files.')

    shell_parse.add_argument('--LLM', type=str, required=True, help='LLM to optimize the prompt for')
    shell_parse.add_argument('--dataset', type=str, help='dataset to optimize for; if unset, all datasets will be used')
    shell_parse.add_argument('--sample_size', type=int, default=1, help='Number of samples to draw from each dataset for metric calculation')
    shell_parse.add_argument('--epochs', type=int, default=1, help='Number of epochs to run the training algorithm')

    shell_parse.add_argument('--mut_prompts', type=int, default=2)
    shell_parse.add_argument('--think_sty',type=int, default=4)
    shell_params = shell_parse.parse_args()
    return shell_params

if __name__ == "__main__":
    params = get_params()
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    exec_datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_configs = {
        "exec_datetime_str": exec_datetime_str,
        "comment": params.comment,
        "device": params.device,
        "LLM": params.LLM,
        "dataset": params.dataset,
        "sample_size": params.sample_size,
        "epochs": params.epochs,
        "mut_prompts": params.mut_prompts,
        "think_sty": params.think_sty
    }

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    writer_model = get_prompt_writer()
    writer_model = ModelWrapper(writer_model, "gemini-pro")
    target_model = params.LLM

    dataset_loop(writer_model, target_model, study_configs)
