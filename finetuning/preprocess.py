import os
import glob
import json
import random
from itertools import product
from pathlib import Path

import ipdb

WITH_SUBSETS = ["bbh", "mmlu", "superglue"]

problem_descriptions = {
    "hellaswag": "Finish the sentence from multiple choice endings.",
    "bbh": "This is a task believed to be beyond the capabilities of current language models.",
    "ARC_c": "This is a grade-school level, multiple-choice science question.",
    "ARC_e": "This is a grade-school level, multiple-choice science question.",
    "gsm8k": "This is a grade school math word problem.",
    "humaneval": "This is a coding problem.",
    "mbpp": "This is a Python programming problem, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on.",
    "mmlu": "This is a question related to elementary mathematics, US history, computer science, law, and more.",
    "piqa": "This is a question related to task of physical commonsense reasoning.",
    "siqa": "This is a question related to social commonsense intelligence.",
    "squadv2": "Answer the question if possible, but also determine when no answer is supported by the paragraph and abstain from answering.",
    "superglue": "This is a difficult language understanding task.",
    "winogrande": "The following is a difficult multiple-choice test.",
    "xsum": "This is an extreme summarization task.",
    "race": "This is a reading comprehension problem designed for middle school and high school students."
}


def data_preprocess(data_path: Path, save_dir: Path):
    prompt_data = {}
    for file in data_path.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            prompt_table = json.load(f)

        table_data = prompt_table["data"]
        for prompt_sample in table_data:
            # new prompt, score
            if prompt_sample[0] in prompt_data:
                prompt_data[prompt_sample[0]].append((prompt_sample[2], prompt_sample[3]))
            else:
                prompt_data[prompt_sample[0]] = [(prompt_sample[2], prompt_sample[3]), ]

    sorted_prompt_data = []
    for dataset_name, prompt_samples in prompt_data.items():
        if any([sub in dataset_name for sub in WITH_SUBSETS]):
            superset_name = dataset_name.split(".")[0]
            problem_desc = problem_descriptions[superset_name]
        else:
            problem_desc = problem_descriptions[dataset_name]

        prompt_sample_pairs = product(prompt_samples, prompt_samples)
        for pair in prompt_sample_pairs:
            if pair[0][0] == pair[1][0] or pair[0][1] == pair[1][1]:
                continue

            sorted_pairs = sorted(pair, key=lambda x: x[1])
            rejected, chosen = sorted_pairs
            sample = {
                "problem_description": problem_desc,
                "chosen": chosen[0],
                "rejected": rejected[0],
                "chosen_score": chosen[1],
                "rejected_score": rejected[1],
                }
            sorted_prompt_data.append(sample)

    random.shuffle(sorted_prompt_data)
    sorted_prompt_data = sorted_prompt_data[:200000]
    with open(save_dir / "preference_data.json", "w", encoding="utf-8") as jsonf:
        json.dump(sorted_prompt_data, jsonf)


if __name__ == "__main__":
    data_path = Path("datasets/prompt_table/gpt-3.5-turbo")
    save_dir = Path("datasets/finetuning/gpt-3.5-turbo")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_preprocess(data_path, save_dir)