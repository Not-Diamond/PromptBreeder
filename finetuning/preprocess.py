import os
import glob
import json
import random
from itertools import product
from pathlib import Path


def data_preprocess(data_path: Path, save_dir: Path):
    prompt_data = []
    for file in glob.iglob(data_path / "*.json"):
        prompt_table = json.load(file)
        table_data = prompt_table["data"]
        for prompt_sample in table_data:
            prompt_data.append((prompt_sample[2], prompt_sample[3]))

    prompt_sample_pairs = product(prompt_data)
    sorted_prompt_data = []
    for pair in prompt_sample_pairs:
        sorted_pairs = sorted(pair, key=lambda x: x[1])
        rejected, chosen = sorted_pairs
        sample = {
        "chosen": chosen[0],
        "rejected": rejected[0],
        "chosen_score": chosen[1],
        "rejected_score": rejected[1],
        }
        sorted_prompt_data.append(sample)

    random.shuffle(sorted_prompt_data)
    with open(save_dir / "preference_data.json", "w", encoding="utf-8") as jsonf:
        json.dump(sorted_prompt_data, jsonf)


if __name__ == "__main__":
    data_path = Path("datasets/prompt_table/gpt-3.5-turbo")
    save_dir = Path("datasets/finetuning/gpt-3.5-turbo")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_preprocess(data_path, save_dir)