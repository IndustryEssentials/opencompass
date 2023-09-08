from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import RougeEvaluator
from opencompass.datasets import XsumDataset

Xsum_reader_cfg = dict(input_columns=["dialogue"], output_column="summary")

Xsum_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt= "Document：{dialogue}\nBased on the previous text, provide a brief single summary:"
                ),
                dict(role="BOT", prompt="{label}"),
            ]),
        ice_token="</E>",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        ice_token="</E>",
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt= "Document：{dialogue}\nBased on the previous text, provide a brief single summary:"
            ),
        ]),
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4])
)

Xsum_eval_cfg = dict(
    evaluator=dict(type=RougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type="Xsum"),
)

Xsum_datasets = [
    dict(
        type=XsumDataset,
        abbr="Xsum",
        path="./data/Xsum/dev.jsonl",
        reader_cfg=Xsum_reader_cfg,
        infer_cfg=Xsum_infer_cfg,
        eval_cfg=Xsum_eval_cfg,
    )
]
