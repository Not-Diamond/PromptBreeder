from mmengine.config import read_base


with read_base():
    from ...datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*hellaswag_datasets, ]
models = [*gpt_3_5_turbo, ]
