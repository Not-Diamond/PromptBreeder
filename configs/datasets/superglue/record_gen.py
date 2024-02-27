from mmengine.config import read_base


with read_base():
    from .record_config import record_datasets
