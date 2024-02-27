from mmengine.config import read_base


with read_base():
    from .hellaswag_gen import hellaswag_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*hellaswag_datasets, ]
models = [*gpt_3_5_turbo, ]
