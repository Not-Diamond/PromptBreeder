from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDRougeEvaluator
from opencompass.datasets import NDXsumDataset


PROMPT_TEMPLATE = "Document：{context}\n{query}"


xsum_reader_cfg = dict(input_columns=["query", "context"], output_column="label")


xsum_infer_cfg = dict(
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

xsum_eval_cfg = dict(
    evaluator=dict(type=NDRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type="Xsum"),
)

xsum_datasets = [
    dict(
        type=NDXsumDataset,
        abbr="xsum",
        reader_cfg=xsum_reader_cfg,
        infer_cfg=xsum_infer_cfg,
        eval_cfg=xsum_eval_cfg,
    )
]
