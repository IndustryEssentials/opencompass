from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import piqaDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess

piqa_reader_cfg = dict(
    input_columns=["goal", "sol1", "sol2"],
    output_column="answer",
    test_split="validation")

piqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt="{goal}\nA. {sol1}\nB. {sol2}\nAnswer:"
                ),
                dict(role="BOT", prompt="{answer}"),
            ]),
        ice_token="</E>",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        ice_token="</E>",
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{goal}\nA. {sol1}\nB. {sol2}\nAnswer:")
            ], ),
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4])
)

piqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

piqa_datasets = [
    dict(
        abbr="piqa",
        type=piqaDataset_V2,
        path="piqa",
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
