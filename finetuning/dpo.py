import os
from pathlib import Path
from more_itertools import batched
from functools import partial
from tqdm import tqdm
import torch

from datasketch import MinHash
import tlsh

import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer, ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config

import ipdb

from dotenv import load_dotenv


def formatting_prompts_func(example: dict):

    def hash_prompt(prompt: str):
        separate_words = prompt.split(" ")
        min_hash_set = set(separate_words)
        min_hash = MinHash(num_perm=128)

        for word in min_hash_set:
            min_hash.update(word.encode("utf8"))

        return [int(i) for i in min_hash.digest()]

    def hash_to_string(hash_array: np.ndarray):
        hash_str = ""
        for val in hash_array:
            bin_val = f"{val:032b}"
            for byte in batched(bin_val, 8):
                byte_str = "".join(byte)
                char = chr(int(byte_str, 2))
                hash_str += char
        return hash_str

    output_texts = []
    for i in range(len(example['chosen'])):
        # prompt_hash = hash_prompt(example['rejected'][i])
        # prompt_hash_str = hash_to_string(prompt_hash)
        # prompt_hash_str = tlsh.hash(example['rejected'][i].encode("utf-8"))
        prompt_hash_str = example['rejected'][i]
        text = f"### Instruction: Rewrite the following prompt for the problem description.\n### Prompt: {prompt_hash_str}\n### Problem description: {example['problem_description'][i]}\n### Rewritten prompt:"
        output_texts.append(text)
    example["prompt"] = output_texts
    return example


if __name__ == "__main__":
    load_dotenv()
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_dir = checkpoint_dir / "raw_str" / "gpt-3.5-turbo" / "roberta-base" / "dpo_model"

    model = AutoModelForCausalLM.from_pretrained("checkpoints/raw_str/gpt-3.5-turbo/roberta-base/sft_model/checkpoint-95110", is_decoder=True)
    model_ref = AutoModelForCausalLM.from_pretrained("checkpoints/raw_str/gpt-3.5-turbo/roberta-base/sft_model/checkpoint-95110", is_decoder=True)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/raw_str/gpt-3.5-turbo/roberta-base/sft_model/checkpoint-95110")

    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json")
    dataset = dataset_dict["train"]
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.remove_columns(["chosen_score", "rejected_score", "problem_description"])

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        max_steps=100000,
        logging_steps=100,
        warmup_steps=150,
        save_strategy="steps",
        evaluation_strategy="steps",
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=1000,
        weight_decay=0.01,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        max_prompt_length=tokenizer.model_max_length,
        generate_during_eval=False
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)