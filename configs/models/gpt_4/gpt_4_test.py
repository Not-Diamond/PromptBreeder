from mmengine.config import read_base


with read_base():
    from ...datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .gpt_4 import models as gpt_4

datasets = [*hellaswag_datasets, ]
models = [*gpt_4, ]
