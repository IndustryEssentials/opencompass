from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BoolQDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

BoolQ_reader_cfg = dict(
    input_columns=["question", "passage"],
    output_column="label",
)

BoolQ_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(role="HUMAN", prompt="{passage}\nQuestion: {question}\nA. Yes\nB. No\nAnswer:"),
                dict(role="BOT", prompt="{label}"),
            ]),
        ice_token="</E>",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role="HUMAN", prompt="{passage}\nQuestion: {question}\nA. Yes\nB. No\nAnswer:"),
        ]),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
)

BoolQ_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

BoolQ_datasets = [
    dict(
        abbr="BoolQ",
        type=BoolQDataset_V2,
        path="./data/SuperGLUE/BoolQ/val.jsonl",
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
