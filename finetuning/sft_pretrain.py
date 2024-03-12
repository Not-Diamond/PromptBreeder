import os
from pathlib import Path
from more_itertools import batched
from functools import partial

from datasketch import MinHash
import tlsh

import numpy as np

from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from dotenv import load_dotenv

import ipdb


def formatting_prompts_func(example: dict, tokenizer: AutoTokenizer):

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

        text = f"### Instruction: Rewrite the following prompt for the problem description.\n### Prompt: {prompt_hash_str}\n### Problem description: {example['problem_description'][i]}\n### Rewritten prompt: {example['chosen'][i]}"
        if len(tokenizer.encode(text)) <= tokenizer.model_max_length:
            output_texts.append(text)

    # print(f"{len(output_texts)}/{len(example['chosen'])} kept")
    return output_texts


def sft_train(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, save_dir: Path):
    response_template = "### Rewritten prompt:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1.41e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        gradient_checkpointing=True,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=partial(formatting_prompts_func, tokenizer=tokenizer),
        data_collator=collator,
        args=training_args,
        max_seq_length=tokenizer.model_max_length
    )

    trainer.train()
    trainer.save_model(save_dir)


if __name__ == "__main__":
    load_dotenv()
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model = AutoModelForCausalLM.from_pretrained("FacebookAI/roberta-base", is_decoder=True)
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json")
    dataset = dataset_dict["train"]
    dataset = dataset.train_test_split(test_size=0.2)

    save_dir = checkpoint_dir / "raw_str" / "gpt-3.5-turbo" / "roberta-base" / "sft_model"

    sft_train(model, tokenizer, dataset, save_dir)