from opencompass.datasets import NDHellaswagDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.utils.text_postprocessors import first_option_postprocess

PROMPT_TEMPLATE = "{context}\n{query}"


hellaswag_reader_cfg = dict(
    input_columns=["context", "query"],
    output_column="label",
)


hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=(PROMPT_TEMPLATE),
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever, ),
    inferencer=dict(type=GenInferencer),
)


hellaswag_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=NDHellaswagDataset,
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
