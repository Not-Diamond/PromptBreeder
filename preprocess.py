import os
import os.path as osp
from dotenv import load_dotenv

import glob
import json
import wandb


def add_prompt_to_template(sample):
    # Prepend random prompt string to template
    prompt = "Answer the following question."
    sample["sample_details"]["prompt_template"]["template"] = "{prompt}\n" + sample['sample_details']['prompt_template']["template"]
    sample["sample_details"]["components"]["prompt"] = {"prompt": prompt}
    sample["sample_details"]["prompt_added"] = True
    return sample


def download_training_dataset(dataset_path):
    # Download training data from wandb
    run = wandb.init(project="LLM Eval", job_type="download")
    artifact = run.use_artifact("training-dataset:latest")
    datadir = artifact.download(dataset_path)
    return datadir


def create_dataset(model, root):
    # Read training data from wandb
    # If sample doesn't have prompt component, prepend a random prompt string using tokenizer (use add_prompt_to_template)
    # Save each dataset as a json file
    data_path = osp.join(root, model, "*.json")
    save_root = osp.join(root, "processed", model)

    for fn in glob.iglob(data_path):
        with open(fn, "r") as json_data:
            data = json.load(json_data)

            dataset = {}
            for sample_id, sample in data.items():
                source = sample["sample_details"]["source"]
                if "prompt" not in sample["sample_details"]["components"]:
                    sample = add_prompt_to_template(sample)
                else:
                    sample["sample_details"]["prompt_added"] = False

                dataset[sample_id] = sample["sample_details"]

            f_path = osp.join(save_root, f"{source}.json")
            if not osp.exists(save_root):
                os.makedirs(save_root)
            with open(f_path, "w") as fp:
                json.dump(dataset, fp)


if __name__ == "__main__":
    load_dotenv()
    datadir = download_training_dataset("datasets")
    # create_dataset("claude-2.1", datadir)
    # create_dataset("gpt-4", datadir)