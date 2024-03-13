import os
from pathlib import Path
from more_itertools import batched
from functools import partial
from tqdm import tqdm
import torch

from datasketch import MinHash
import tlsh

import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

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
        # prompt_hash = hash_prompt(example['rejected'][i])
        # prompt_hash_str = hash_to_string(prompt_hash)

        prompt_hash_str = tlsh.hash(example['rejected'][i].encode("utf-8"))
        text = f"### Instruction: Rewrite the following prompt for the problem description.\n### Prompt: {prompt_hash_str}\n### Problem description: {example['problem_description'][i]}\n### Rewritten prompt:"
        output_texts.append(text)
    example["formatted"] = output_texts
    return example


def tokenize(sample, tokenizer):
    input_ids = []
    for i in range(len(sample['chosen'])):
        tokens = tokenizer.encode(sample["formatted"][i])
        if len(tokens) < tokenizer.model_max_length:
            tokens += [tokenizer.eos_token_id for _ in range((tokenizer.model_max_length - len(tokens)))]
        input_ids.append(tokens)
    sample["input_ids"] = input_ids
    return sample


def ppo_train(ppo_trainer: PPOTrainer, tokenizer: AutoTokenizer, reward_model: pipeline, generation_kwargs: dict, save_dir: Path):
    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader): 
            query_tensors = batch["input_ids"]
        
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
            texts = [res for res in batch["response"]]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_model(save_dir)


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_dir = checkpoint_dir / "hashed" / "tlsh" / "gpt-3.5-turbo" / "roberta-base" / "ppo_model"

    sft_model = AutoModelForCausalLM.from_pretrained("checkpoints/hashed/tlsh/gpt-3.5-turbo/roberta-base/sft_model/checkpoint-100000", is_decoder=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/hashed/tlsh/gpt-3.5-turbo/roberta-base/sft_model/checkpoint-100000", padding_side="left")
    tokenizer.padding_side = "left"

    reward_model = pipeline("text-classification", "checkpoints/hashed/tlsh/gpt-3.5-turbo/roberta-base/reward_model/checkpoint-60000")

    dataset_dict = load_dataset("json", data_files="datasets/finetuning/gpt-3.5-turbo/preference_data.json")
    dataset = dataset_dict["train"]
    dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.map(partial(tokenize, tokenizer=tokenizer), batched=True)

    config = PPOConfig(
        model_name="roberta-promptwriter",
        learning_rate=1.41e-5,
        ppo_epochs=10
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset["train"],
        tokenizer=tokenizer
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_length": tokenizer.model_max_length
    }

    ppo_train(ppo_trainer, tokenizer, reward_model, generation_kwargs, save_dir)