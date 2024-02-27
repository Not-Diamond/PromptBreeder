from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDEDAccEvaluator
from opencompass.datasets import NDsiqaDataset


PROMPT_TEMPLATE = '{context}\n{query}'


siqa_reader_cfg = dict(
    input_columns=["context", "query"],
    output_column="evaluator_target")


siqa_infer_cfg = dict(
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


siqa_eval_cfg = dict(
    evaluator=dict(type=NDEDAccEvaluator),
    pred_role="BOT",
)


siqa_datasets = [
    dict(
        abbr="siqa",
        type=NDsiqaDataset,
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
