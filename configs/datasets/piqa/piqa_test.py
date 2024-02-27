from mmengine.config import read_base


with read_base():
    from .piqa_gen import piqa_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*piqa_datasets, ]
models = [*gpt_3_5_turbo, ]
