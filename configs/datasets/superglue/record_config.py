from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDEMEvaluator
from opencompass.datasets import NDReCoRDDataset


PROMPT_TEMPLATE = "{context}\nQuestion: {query}"


record_reader_cfg = dict(
    input_columns=["query", "context"],
    output_column="label",
)


record_infer_cfg = dict(
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


record_eval_cfg = dict(
    evaluator=dict(type=NDEMEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type="ReCoRD"),
)


record_datasets = [
    dict(
        type=NDReCoRDDataset,
        abbr="superglue.record",
        reader_cfg=record_reader_cfg,
        infer_cfg=record_infer_cfg,
        eval_cfg=record_eval_cfg,
    )
]
