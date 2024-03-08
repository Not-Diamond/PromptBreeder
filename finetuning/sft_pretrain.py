import os
from pathlib import Path
from more_itertools import batched

from datasketch import MinHash

import numpy as np

from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from dotenv import load_dotenv

import ipdb


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
        prompt_hash = hash_prompt(example['rejected'][i])
        prompt_hash_str = hash_to_string(prompt_hash)
        text = f"""### Instruction: Rewrite the following prompt for the problem description.
        ### Prompt: {prompt_hash_str}
        ### Problem description: {example['problem_description'][i]}
        ### Rewritten prompt: {example['chosen'][i]}
        """
        output_texts.append(text)
    return output_texts


def sft_train(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, save_dir: Path):
    response_template = "### Rewritten prompt:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1.41e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
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
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
        max_seq_length=tokenizer.model_max_length
    )

    trainer.train()
    trainer.save_model(save_dir)


def sft_test(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, save_dir: Path):
    pass


if __name__ == "__main__":
    load_dotenv()
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json", cache_dir="/data/notdiamond/.cache")
    dataset = dataset_dict["train"]
    dataset = dataset.train_test_split(test_size=0.2)

    model = AutoModelForCausalLM.from_pretrained("google/bigbird-roberta-base", is_decoder=True, cache_dir="/data/notdiamond/.cache")
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base", cache_dir="/data/notdiamond/.cache")

    sft_train(model, tokenizer, dataset, checkpoint_dir / "hashed" / "gpt-3.5-turbo" / "sft_model")