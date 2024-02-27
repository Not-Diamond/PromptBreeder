from mmengine.config import read_base


with read_base():
    from .squadv2_gen import squadv2_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*squadv2_datasets, ]
models = [*gpt_3_5_turbo, ]
