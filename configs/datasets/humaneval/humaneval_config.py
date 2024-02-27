from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import humaneval_postprocess_v2, NDHumanevalDataset, NDHumanEvaluator


PROMPT_TEMPLATE = "{query}"


humaneval_reader_cfg = dict(
    input_columns=['query'], output_column='task_id')


humaneval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=PROMPT_TEMPLATE),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))


humaneval_eval_cfg = dict(
    evaluator=dict(type=NDHumanEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    dataset="humaneval",
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)


humaneval_datasets = [
    dict(
        abbr='humaneval',
        type=NDHumanevalDataset,
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg)
]
