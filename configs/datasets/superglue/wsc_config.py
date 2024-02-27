from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDWSCDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


PROMPT_TEMPLATE = "Passage: {context}\n{query}\nAnswer:"


wsc_reader_cfg = dict(
    input_columns=["context", "query"],
    output_column="label",
)


wsc_infer_cfg = dict(
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


wsc_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)


wsc_datasets = [
    dict(
        abbr="superglue.wsc",
        type=NDWSCDataset,
        reader_cfg=wsc_reader_cfg,
        infer_cfg=wsc_infer_cfg,
        eval_cfg=wsc_eval_cfg,
    )
]
