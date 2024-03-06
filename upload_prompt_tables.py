import wandb
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()
    artifact = wandb.Artifact(name="prompt-table", type="dataset")
    artifact.add_dir("datasets/prompt_table")
    with wandb.init(project="PromptBreeder") as run:
        run.log_artifact(artifact)