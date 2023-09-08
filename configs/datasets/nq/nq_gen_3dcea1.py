from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import NaturalQuestionDataset, NQEvaluator

nq_reader_cfg = dict(
    input_columns=['question'], output_column='answer', train_split='test')

nq_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(role='HUMAN', prompt='Question: {question}?\nAnswer: '),
                dict(role="BOT", prompt="{answer}"),
            ]),
        ice_token="</E>",
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Question: {question}?\nAnswer: '),
            ]
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=FixKRetriever),
#   inferencer=dict(type=GenInferencer)
    inferencer=dict(type=GenInferencer, fix_id_list=[0, 1, 2, 3, 4]),
)


nq_eval_cfg = dict(evaluator=dict(type=NQEvaluator), pred_role="BOT")

nq_datasets = [
    dict(
        type=NaturalQuestionDataset,
        abbr='nq',
        path='./data/nq/',
        reader_cfg=nq_reader_cfg,
        infer_cfg=nq_infer_cfg,
        eval_cfg=nq_eval_cfg)
]
