from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDAXDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_TEMPLATE = "{context}\n{query}\nAnswer:"


rte_reader_cfg = dict(
    input_columns=["context", "query"],
    output_column="label",
)

rte_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=PROMPT_TEMPLATE
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

rte_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

rte_datasets = [
    dict(
        abbr="superglue.rte",
        type=NDAXDataset,  # rte share the same format with ax
        subset="rte",
        reader_cfg=rte_reader_cfg,
        infer_cfg=rte_infer_cfg,
        eval_cfg=rte_eval_cfg,
    )
]
