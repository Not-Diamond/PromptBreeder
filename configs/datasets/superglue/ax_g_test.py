from mmengine.config import read_base


with read_base():
    from .ax_g_gen import ax_g_datasets
    from ...models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo

datasets = [*ax_g_datasets, ]
models = [*gpt_3_5_turbo, ]
