from mmengine.config import read_base


with read_base():
    from .siqa_gen import siqa_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*siqa_datasets, ]
models = [*gpt_3_5_turbo, ]
