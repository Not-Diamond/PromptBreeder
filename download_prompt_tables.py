import wandb
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    with wandb.init(project="PromptBreeder") as run:
        artifact = run.use_artifact("prompt-table:latest")
        _ = artifact.download(root="datasets/prompt_table")