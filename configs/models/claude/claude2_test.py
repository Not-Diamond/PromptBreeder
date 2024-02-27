from mmengine.config import read_base


with read_base():
    from ...datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .claude2 import models as claude2

datasets = [*hellaswag_datasets, ]
models = [*claude2, ]
