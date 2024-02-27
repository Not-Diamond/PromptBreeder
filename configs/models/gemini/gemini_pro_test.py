from mmengine.config import read_base


with read_base():
    from ...datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .gemini_pro import models as gemini_pro

datasets = [*hellaswag_datasets, ]
models = [*gemini_pro, ]
