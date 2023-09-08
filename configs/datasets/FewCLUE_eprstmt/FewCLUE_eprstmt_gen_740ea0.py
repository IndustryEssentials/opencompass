from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import eprstmtDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

eprstmt_reader_cfg = dict(
    input_columns=["sentence"], output_column="label", test_split="train")

eprstmt_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    '内容： "{sentence}"。请对上述内容进行情绪分类。\nA. 积极\nB. 消极\n请从”A“，”B“中进行选择。\n答：'
                ),
                dict(role="BOT", prompt="{label}\n"),
            ]
        ),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        ice_token="</E>",
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=
                '内容： "{sentence}"。请对上述内容进行情绪分类。\nA. 积极\nB. 消极\n请从”A“，”B“中进行选择。\n答：'
            ),
        ]),
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
)

eprstmt_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

eprstmt_datasets = [
    dict(
        abbr="eprstmt-dev",
        type=eprstmtDataset_V2,
        path="./data/FewCLUE/eprstmt/dev_few_all.json",
        reader_cfg=eprstmt_reader_cfg,
        infer_cfg=eprstmt_infer_cfg,
        eval_cfg=eprstmt_eval_cfg,
    ),
    dict(
        abbr="eprstmt-test",
        type=eprstmtDataset_V2,
        path="./data/FewCLUE/eprstmt/test_public.json",
        reader_cfg=eprstmt_reader_cfg,
        infer_cfg=eprstmt_infer_cfg,
        eval_cfg=eprstmt_eval_cfg,
    ),
]
