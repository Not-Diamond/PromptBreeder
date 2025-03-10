from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import NDAccEvaluator
from opencompass.datasets import NDMMLUDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess


# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

mmlu_all_sets = [
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "electrical_engineering",
    "astronomy",
    "anatomy",
    "abstract_algebra",
    "machine_learning",
    "clinical_knowledge",
    "global_facts",
    "management",
    "nutrition",
    "marketing",
    "professional_accounting",
    "high_school_geography",
    "international_law",
    "moral_scenarios",
    "computer_security",
    "high_school_microeconomics",
    "professional_law",
    "medical_genetics",
    "professional_psychology",
    "jurisprudence",
    "world_religions",
    "philosophy",
    "virology",
    "high_school_chemistry",
    "public_relations",
    "high_school_macroeconomics",
    "human_sexuality",
    "elementary_mathematics",
    "high_school_physics",
    "high_school_computer_science",
    "high_school_european_history",
    "business_ethics",
    "moral_disputes",
    "high_school_statistics",
    "miscellaneous",
    "formal_logic",
    "high_school_government_and_politics",
    "prehistory",
    "security_studies",
    "high_school_biology",
    "logical_fallacies",
    "high_school_world_history",
    "professional_medicine",
    "high_school_mathematics",
    "college_medicine",
    "high_school_us_history",
    "sociology",
    "econometrics",
    "high_school_psychology",
    "human_aging",
    "us_foreign_policy",
    "conceptual_physics",
]

PROMPT_TEMPLATE = "{prompt}\n{query}\nAnswer: "


mmlu_reader_cfg = dict(
    input_columns=["query", "prompt"],
    output_column="label"
)


mmlu_datasets = []
for _name in mmlu_all_sets:
    mmlu_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role="HUMAN",
                    prompt=PROMPT_TEMPLATE,
                ),
                dict(role="BOT", prompt="{label}\n")
            ]),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin="</E>",
                round=[
                    dict(
                        role="HUMAN",
                        prompt=PROMPT_TEMPLATE
                    ),
                ],
            ),
            ice_token="</E>",
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=NDAccEvaluator),
        pred_postprocessor=dict(type=first_capital_postprocess))

    mmlu_datasets.append(
        dict(
            abbr=f"mmlu.{_name}",
            type=NDMMLUDataset,
            subset=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))

del _name
