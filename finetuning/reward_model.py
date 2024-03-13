import os
from pathlib import Path
from functools import partial

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from trl import RewardTrainer

from dotenv import load_dotenv


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


if __name__ == "__main__":
    load_dotenv()
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_dir = checkpoint_dir / "hashed" / "tlsh" / "gpt-3.5-turbo" / "roberta-base" / "reward_model"

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=1)
    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json")
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
        num_train_epochs=10,
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