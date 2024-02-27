from opencompass.datasets import NDARCDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_TEMPLATE = "{query}"


ARC_c_reader_cfg = dict(
    input_columns=["query"],
    output_column="label")


ARC_c_infer_cfg = dict(
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


ARC_c_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


ARC_c_datasets = [
    dict(
        abbr="ARC_c",
        type=NDARCDataset,
        subset="ARC_c",
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
