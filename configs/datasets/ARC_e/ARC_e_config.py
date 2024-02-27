from opencompass.datasets import NDARCDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_TEMPLATE = "{query}"


ARC_e_reader_cfg = dict(
    input_columns=["query"],
    output_column="label")


ARC_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=PROMPT_TEMPLATE
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


ARC_e_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


ARC_e_datasets = [
    dict(
        abbr="ARC_e",
        type=NDARCDataset,
        subset="ARC_e",
        reader_cfg=ARC_e_reader_cfg,
        infer_cfg=ARC_e_infer_cfg,
        eval_cfg=ARC_e_eval_cfg,
    )
]
