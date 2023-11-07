import json
import argparse
import fnmatch
from typing import Dict, List
import itertools
from pathlib import Path

from mmengine.config import Config, ConfigDict

from opencompass.openicl.icl_inferencer import (CLPInferencer, GenInferencer,
                                                PPLInferencer)
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.utils import (Menu, build_dataset_from_cfg,
                               build_model_from_cfg, dataset_abbr_from_cfg,
                               model_abbr_from_cfg)

FEW_SHOTS_TEMPLATES = {
    "ceval": "{_few_shots_str}\n{_input_text}",
    "chid": "{_input_texts}"
}

def get_few_shots_template(dataset_abbr) -> str:
    dataset_abbr = dataset_abbr.split("-")[0]
    return FEW_SHOTS_TEMPLATES.get(dataset_abbr) or "{_few_shots_str}\n{_input_text}"


def parse_args():
    parser = argparse.ArgumentParser(description='View generated prompts based on datasets (and models)')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('-p',
                        '--pattern',
                        type=str,
                        help='To match the dataset abbr.')
    parser.add_argument('-c', '--count', type=int, default=1000000, help='Number of prompts to print')
    parser.add_argument("-o", "--output_dir", default=".", type=str, help="output dir")
    args = parser.parse_args()
    return args


def parse_dataset_cfg(dataset_cfg: ConfigDict) -> Dict[str, ConfigDict]:
    dataset2cfg = {}
    for dataset in dataset_cfg:
        dataset2cfg[dataset_abbr_from_cfg(dataset)] = dataset
    return dataset2cfg


def extract_prompt_item(item):
    if isinstance(item, str) and item:
        return item
    elif isinstance(item, dict):
        if set(['section', 'pos']) == set(item.keys()):
            return None
        return item.get('prompt', None)
    return None


def dump_prompt_list(prompt_list) -> List[str]:
    """
    Adapted from opencompass/models/base.py

    prompt_list is of type PromptList
    """
    if isinstance(prompt_list, str):
        return [prompt_list, ""]
    prompts = [p for p in (extract_prompt_item(item) for item in prompt_list) if p is not None]
    return prompts


def pair_elements(lst, join=True):
    """
    Make pairs out of input list, for example:
    input: ["Qustion", "Answer", "Question2", "Answer2"]
    output: ["Qustion Answer", "Question2 Answer2"]
    """
    it = iter(lst)
    return [f"{q}{a}" if join else [q, a] for q, a in itertools.zip_longest(it, it)]


def dump_ice_list(prompt_list, join=True) -> List[str]:
    prompts = [p for p in (extract_prompt_item(item) for item in prompt_list) if p is not None]
    return pair_elements(prompts, join)


def write_prompts(dataset_cfg, dataset_abbr, output_file, count=1, file_mode="w"):
    infer_cfg = dataset_cfg.get('infer_cfg')

    fix_id_list = infer_cfg.inferencer.get('fix_id_list', [])
    dataset = build_dataset_from_cfg(dataset_cfg)

    ice_template = None
    if hasattr(infer_cfg, 'ice_template'):
        ice_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['ice_template'])

    prompt_template = None
    if hasattr(infer_cfg, 'prompt_template'):
        prompt_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['prompt_template'])

    infer_cfg['retriever']['dataset'] = dataset
    retriever = ICL_RETRIEVERS.build(infer_cfg['retriever'])

    if fix_id_list:
        ice_idx_list = retriever.retrieve(fix_id_list)
    else:
        ice_idx_list = retriever.retrieve()

    assert infer_cfg.inferencer.type in [PPLInferencer, GenInferencer], \
        'Only PPLInferencer and GenInferencer are supported'

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, file_mode) as f:
        for idx in range(min(count, len(ice_idx_list))):
            if infer_cfg.inferencer.type == PPLInferencer:
                labels = retriever.get_labels(ice_template=ice_template, prompt_template=prompt_template)
                # [0, 1, 2, 3, 4, 5, 6]
                # ["Negative", "Positive"]
                ice_idx = ice_idx_list[idx]
                ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                prompts_with_label = []
                for label in labels:
                    prompt = retriever.generate_label_prompt(
                        idx,
#                       ice[idx],
                        "",
                        label,
                        ice_template=ice_template,
                        prompt_template=prompt_template,
                        remain_sep=None
                    )
                    prompts_with_label.append(prompt)

                data = dataset.test[idx]
                gt = data[retriever.dataset_reader.output_column]
                if not isinstance(gt, str):
                    gt = str(gt)
                item = {
                    "inputs": {
                        "_prefix": "",
                        "_few_shots": dump_ice_list(ice),
                        "_ice": dump_ice_list(ice, join=False),
                        "_input_text": "\n".join(dump_prompt_list(prompts_with_label[0])), # just a placeholder
                        "_input_texts": [list(dump_prompt_list(prompt)) for prompt in prompts_with_label],
                        "_few_shots_prompt_format": "{_input_texts}",
                        "_choices": {label: "" for idx, label in enumerate(labels)},
                        **data,
                    },
                    "gt": {"gt_text": gt}
                }
                f.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            elif infer_cfg.inferencer.type == GenInferencer:
                ice_idx = ice_idx_list[idx]
                ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                # get zeroshot prompt by explicitly pass ice=""
                prompt = retriever.generate_prompt_for_generate_task(
                    idx,
                    "",
                    gen_field_replace_token=infer_cfg.inferencer.get('gen_field_replace_token', ''),
                    ice_template=ice_template,
                    prompt_template=prompt_template
                )
                data = dataset.test[idx]
                _few_shots_prompt_format = get_few_shots_template(dataset_abbr)
                templates = {}
                if prompt_template:
                    templates["prompt_template"] = prompt_template.template.round[0].prompt
                if ice_template:
                    templates["ice_template"] = ice_template.template.round[0].prompt
                item = {
                    "inputs": {
                        "_prefix": "",
                        "_few_shots": dump_ice_list(ice),
                        "_ice": dump_ice_list(ice, join=False),
                        "_input_text": "\n".join(dump_prompt_list(prompt)),
                        "_few_shots_prompt_format": _few_shots_prompt_format,
                        **data,
                        **templates,
                    },
                    "gt": {"gt_text": data[retriever.dataset_reader.output_column]}
                }
                f.write(f"{json.dumps(item, ensure_ascii=False)}\n")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    output_path = Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if 'datasets' in cfg:
        dataset2cfg = parse_dataset_cfg(cfg.datasets)
    else:
        dataset2cfg = {}
        for key in cfg.keys():
            if key.endswith('_datasets'):
                dataset2cfg.update(parse_dataset_cfg(cfg[key]))
    if args.pattern is not None:
        matches = fnmatch.filter(dataset2cfg, args.pattern)
        if len(matches) == 0:
            raise ValueError(
                'No dataset match the pattern. Please select from: \n' +
                '\n'.join(dataset2cfg.keys()))
        dataset2cfg = {k: dataset2cfg[k] for k in matches}

    for dataset_abbr, dataset_cfg in dataset2cfg.items():
        print('=' * 64, '[BEGIN]', '=' * 64)
        print(f'[DATASET]: {dataset_abbr}')
        print('---')
        try:
            evaluator_type = dataset_cfg['eval_cfg']['evaluator']['type']
            if not isinstance(evaluator_type, str):
                evaluator_type = evaluator_type.__name__
        except Exception:
            evaluator_type = None
        language = dataset_cfg.pop("language") or None

        output_path = Path(args.output_dir)
        if evaluator_type:
            output_path = output_path / evaluator_type
        if language:
            output_path = output_path / language

        output_file = output_path / f"{dataset_abbr}.jsonl"
        write_prompts(dataset_cfg, dataset_abbr, output_file, args.count)
        print('=' * 65, '[END]', '=' * 65, '\n')


if __name__ == '__main__':
    main()
