from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NDSQuADV2Dataset, NDSQuADV2Evaluator


PROMPT_TEMPLATE = "{context}\n{prompt} {question}\nAnswer:"


squadv2_reader_cfg = dict(
    input_columns=['context', 'query', 'prompt'],
    output_column='label')


squadv2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=PROMPT_TEMPLATE),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50))


squadv2_eval_cfg = dict(
    evaluator=dict(type=NDSQuADV2Evaluator), pred_role='BOT')


squadv2_datasets = [
    dict(
        type=NDSQuADV2Dataset,
        abbr='squadv2',
        reader_cfg=squadv2_reader_cfg,
        infer_cfg=squadv2_infer_cfg,
        eval_cfg=squadv2_eval_cfg)
]
