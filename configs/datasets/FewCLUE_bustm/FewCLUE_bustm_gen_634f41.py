from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import AFQMCDataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

bustm_reader_cfg = dict(
    input_columns=["sentence1", "sentence2"],
    output_column="label",
    test_split="train")

bustm_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=
                    "语句一：“{sentence1}”\n语句二：“{sentence2}”\n请判断语句一和语句二说的是否是一个意思？\nA. 无关\nB. 相关\n请从“A”，“B”中进行选择。\n答：",
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
                "语句一：“{sentence1}”\n语句二：“{sentence2}”\n请判断语句一和语句二说的是否是一个意思？\nA. 无关\nB. 相关\n请从“A”，“B”中进行选择。\n答：",
            ),
        ]),
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
)

bustm_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

bustm_datasets = [
    dict(
        abbr="bustm-dev",
        type=AFQMCDataset_V2,  # bustm share the same format with AFQMC
        path="./data/FewCLUE/bustm/dev_few_all.json",
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg,
    ),
    dict(
        abbr="bustm-test",
        type=AFQMCDataset_V2,  # bustm share the same format with AFQMC
        path="./data/FewCLUE/bustm/test_public.json",
        reader_cfg=bustm_reader_cfg,
        infer_cfg=bustm_infer_cfg,
        eval_cfg=bustm_eval_cfg,
    ),
]
