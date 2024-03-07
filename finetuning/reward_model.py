import os
from pathlib import Path
from functools import partial

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from trl import RewardTrainer


def preprocess_function(examples, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "margin": [],
    }
    for chosen, rejected, chosen_score, rejected_score in zip(examples["chosen"], examples["rejected"], examples["chosen_score"], examples["rejected_score"]):
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["margin"].append(chosen_score - rejected_score)

    return new_examples

"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512 \
"""


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_dir = checkpoint_dir / "hashed" / "gpt-3.5-turbo" / "reward_model"

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-multilingual-cased", num_labels=1
    )
    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json", cache_dir="/data/notdiamond/.cache")
    raw_dataset = dataset_dict["train"]
    raw_dataset = raw_dataset.train_test_split(test_size=0.2)

    raw_datasets = raw_dataset.map(
        partial(preprocess_function, max_length=tokenizer.model_max_length),
        batched=True,
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1.41e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        optim="adamw_torch",
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False
    )


    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=tokenizer.model_max_length
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)