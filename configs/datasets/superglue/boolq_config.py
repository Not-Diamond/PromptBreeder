from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDBoolQDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


PROMPT_TEMPLATE = "{context}\n{query}\nAnswer:"


boolq_reader_cfg = dict(
    input_columns=["query", "context"],
    output_column="label",
)


boolq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=PROMPT_TEMPLATE),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


boolq_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)


boolq_datasets = [
    dict(
        abbr="superglue.boolq",
        type=NDBoolQDataset,
        reader_cfg=boolq_reader_cfg,
        infer_cfg=boolq_infer_cfg,
        eval_cfg=boolq_eval_cfg,
    )
]
