from mmengine.config import read_base


with read_base():
    from .ax_b_gen import ax_b_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*ax_b_datasets, ]
models = [*gpt_3_5_turbo, ]
