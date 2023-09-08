from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import HFDataset, HumanEvaluator, humaneval_postprocess

humaneval_reader_cfg = dict(
    input_columns=['prompt'], output_column='task_id', train_split='test')

# TODO: allow empty output-column
humaneval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt='Complete the following python code:\n{prompt}',
                ),
                dict(role="BOT", prompt="{task_id}"),
            ]),
        ice_token="</E>",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        ice_token="</E>",
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Complete the following python code:\n{prompt}'),
        ])),
    retriever=dict(type=FixKRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512, fix_id_list=[0, 1, 2, 3, 4]))

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvaluator),
    pred_role='BOT',
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess),
)

humaneval_datasets = [
    dict(
        type=HFDataset,
        path='openai_humaneval',
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg)
]
