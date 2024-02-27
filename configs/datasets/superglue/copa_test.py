from mmengine.config import read_base


with read_base():
    from .copa_gen import copa_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*copa_datasets, ]
models = [*gpt_3_5_turbo, ]
