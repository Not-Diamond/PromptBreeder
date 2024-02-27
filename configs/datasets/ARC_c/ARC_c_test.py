from mmengine.config import read_base


with read_base():
    from .ARC_c_gen import ARC_c_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*ARC_c_datasets, ]
models = [*gpt_3_5_turbo, ]
