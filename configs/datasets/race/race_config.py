from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDRaceDataset
from opencompass.utils.text_postprocessors import first_option_postprocess


PROMPT_TEMPLATE = "{prompt}\n{context}\n\nQ: {query}"


race_reader_cfg = dict(
    input_columns=['prompt', 'context', 'query'],
    output_column='label')


race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=PROMPT_TEMPLATE
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))


race_eval_cfg = dict(
    evaluator=dict(type=NDAccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT')


race_datasets = [
    dict(
        abbr='race',
        type=NDRaceDataset,
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
]
