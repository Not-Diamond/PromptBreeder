from mmengine.config import read_base


with read_base():
    from .ARC_e_gen import ARC_e_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*ARC_e_datasets, ]
models = [*gpt_3_5_turbo, ]
