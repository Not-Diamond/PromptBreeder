from mmengine.config import read_base


with read_base():
    from .datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    from .datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from .datasets.bbh.bbh_gen import bbh_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.hellaswag.hellaswag_gen import hellaswag_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.mbpp.mbpp_gen import mbpp_datasets
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.piqa.piqa_gen import piqa_datasets
    from .datasets.race.race_gen import race_datasets
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.squadv2.squadv2_gen import squadv2_datasets
    from .datasets.superglue.ax_b_gen import ax_b_datasets
    from .datasets.superglue.ax_g_gen import ax_g_datasets
    from .datasets.superglue.boolq_gen import boolq_datasets
    from .datasets.superglue.cb_gen import cb_datasets
    from .datasets.superglue.copa_gen import copa_datasets
    from .datasets.superglue.multirc_gen import multirc_datasets
    from .datasets.superglue.record_gen import record_datasets
    from .datasets.superglue.rte_gen import rte_datasets
    from .datasets.superglue.wic_gen import wic_datasets
    from .datasets.superglue.wsc_gen import wsc_datasets
    from .datasets.winogrande.winogrande_gen import winogrande_datasets
    from .datasets.xsum.xsum_gen import xsum_datasets

    from .models.gpt_3_5_turbo.gpt_3_5_turbo import models as gpt_3_5_turbo
    from .models.gpt_4.gpt_4 import models as gpt_4
    from .models.claude.claude2 import models as claude2
    from .models.claude.claude2_1 import models as claude2_1
    from .models.gemini.gemini_pro import models as gemini_pro


datasets = [
    *hellaswag_datasets,
    *ARC_c_datasets,
    *ARC_e_datasets,
    *bbh_datasets,
    *gsm8k_datasets,
    *humaneval_datasets,
    *mbpp_datasets,
    *mmlu_datasets,
    *piqa_datasets,
    *race_datasets,
    *siqa_datasets,
    *squadv2_datasets,
    *ax_b_datasets,
    *ax_g_datasets,
    *boolq_datasets,
    *cb_datasets,
    *copa_datasets,
    *multirc_datasets,
    *record_datasets,
    *rte_datasets,
    *wic_datasets,
    *wsc_datasets,
    *winogrande_datasets,
    *xsum_datasets
]

models = [*gpt_3_5_turbo, *gpt_4, *claude2, *gemini_pro, *claude2_1]