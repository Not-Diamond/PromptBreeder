from mmengine.config import read_base


with read_base():
    from .gsm8k_gen import gsm8k_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*gsm8k_datasets, ]
models = [*gpt_3_5_turbo, ]
