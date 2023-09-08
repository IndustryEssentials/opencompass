from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MultiRCDataset_V2
from opencompass.utils.text_postprocessors import first_option_postprocess

MultiRC_reader_cfg = dict(
    input_columns=["question", "text", "answer"],
    output_column="label",
)

MultiRC_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt = "{text}\nQuestion: {question}\nAnswer: {answer}\nIs it true?\nA. Yes\nB. No\nAnswer:"
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
                prompt=
                "{text}\nQuestion: {question}\nAnswer: {answer}\nIs it true?\nA. Yes\nB. No\nAnswer:"
            ),
        ]),
    ),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4])
)

MultiRC_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

MultiRC_datasets = [
    dict(
        abbr="MultiRC",
        type=MultiRCDataset_V2,
        path="./data/SuperGLUE/MultiRC/val.jsonl",
        reader_cfg=MultiRC_reader_cfg,
        infer_cfg=MultiRC_infer_cfg,
        eval_cfg=MultiRC_eval_cfg,
    )
]
