from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDAXDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_TEMPLATE = "{context}\n{query}\nAnswer:"


ax_b_reader_cfg = dict(
    input_columns=["query", "context"],
    output_column="label",
)


ax_b_infer_cfg = dict(
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


ax_b_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)


ax_b_datasets = [
    dict(
        abbr="superglue.ax_b",
        type=NDAXDataset,
        subset="ax_b",
        reader_cfg=ax_b_reader_cfg,
        infer_cfg=ax_b_infer_cfg,
        eval_cfg=ax_b_eval_cfg,
    )
]
