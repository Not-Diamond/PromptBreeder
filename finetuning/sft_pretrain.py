import os
from pathlib import Path

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def formatting_prompts_func(example: dict):
    output_texts = []
    for i in range(len(example['chosen'])):
        text = f"""### Instruction: Rewrite the following prompt to make it better.
        ### Prompt: {example['rejected'][i]}
        ### Rewritten prompt: {example['chosen'][i]}
        """
        output_texts.append(text)
    return output_texts


def sft_train(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, save_dir: Path):
    response_template = " ### Rewritten prompt:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(save_dir)


if __name__ == "__main__":
    checkpoint_dir = Path("checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset_path = Path("datasets/finetuning/gpt-3.5-turbo/preference_data.json")
    dataset = load_dataset(dataset_path)

    model = AutoModelForCausalLM.from_pretrained("google/bigbird-roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")

    sft_train(model, tokenizer, checkpoint_dir / "hashed" / "gpt-3.5-turbo")