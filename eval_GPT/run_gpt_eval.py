import os
import pickle as pkl
import openai
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import backoff
from functools import lru_cache
import random
from pathlib import Path
from dotenv import load_dotenv


def prompt_direct_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_direct_inferring_masked(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'the sentiment polarity towards {target} is [mask]'
    return new_context, prompt


def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the sentence "{context}", '
    prompt = new_context + f'which specific aspect of {target} is possibly mentioned?'
    return new_context, prompt


def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' ' + aspect_expr  # + ' The mentioned aspect is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + ' ' + opinion_expr  # + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr):
    prompt = polarity_expr + ' Based on these contexts, summarize the sentiment polarity, and return only one of these words: positive, neutral, or negative.'
    return prompt


def preprocess_data(dataname, config):
    def read_file(dataname, config):
        repo_root = Path(__file__).resolve().parents[1]
        test_file = repo_root / 'data' / dataname / f'{dataname.capitalize()}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
        test_data = pkl.load(open(test_file, 'rb'))
        return test_data

    def transformer2indices(cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append([text, target, label, implicit])
        return res

    data = read_file(dataname, config)
    return transformer2indices(data)


def prepare_data(args, config):
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / 'data' / 'preprocessed' / f'{args.data_name}_base_google-flan-t5-base.pkl'

    if path.exists():
        data = pkl.load(open(path, 'rb'))
    else:
        data = preprocess_data(args.data_name, config)
        pkl.dump(data, open(path, 'wb'))
    return data


def report_score(golds, preds, mode='test'):
    res = {}
    print("Golds (total):", golds['total'])
    print("Preds (total):", preds['total'])
    res['Acc_SA'] = accuracy_score(golds['total'], preds['total'])
    print("Acc_SA (accuracy score):", res['Acc_SA'])

    print("Golds (explicits):", golds['explicits'])
    print("Preds (explicits):", preds['explicits'])
    res['F1_SA'] = f1_score(golds['total'], preds['total'], labels=[0, 1, 2], average='macro')
    print("F1_SA (macro F1 score):", res['F1_SA'])

    res['F1_ESA'] = f1_score(golds['explicits'], preds['explicits'], labels=[0, 1, 2], average='macro')
    print("F1_ESA (macro F1 score for explicits):", res['F1_ESA'])

    res['F1_ISA'] = f1_score(golds['implicits'], preds['implicits'], labels=[0, 1, 2], average='macro')
    print("F1_ISA (macro F1 score for implicits):", res['F1_ISA'])
    res['default'] = res['F1_SA']
    res['mode'] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)
    return res


@backoff.on_exception(backoff.expo, Exception)
def request_result(conversation, prompt_text, model_name: str):
    conversation.append(
        {'role': 'user', "content": prompt_text}
    )
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=conversation,
    )
    conversation.append(
        {"role": "assistant",
         "content": response.choices[0].message.content}
    )
    result = response.choices[0].message.content.replace('\n', ' ').strip()
    return conversation, result


def eval_run(args):
    dataname = args.data_name
    config_dict = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config_dict)

    repo_root = Path(__file__).resolve().parents[1]
    eval_dir = repo_root / 'eval_GPT'
    eval_dir.mkdir(parents=True, exist_ok=True)
    counter_path = eval_dir / f'counter_{dataname}.txt'
    output_path = eval_dir / f'output_{dataname}.txt'

    label_list = ['positive', 'negative', 'neutral']
    label_dict = {w: i for i, w in enumerate(label_list)}

    data = prepare_data(args, config)

    system_role = dict({'role': 'system', "content": "Now you are an expert of sentiment and opinion analysis."})

    preds, golds = defaultdict(list), defaultdict(list)
    keys = ['total', 'explicits', 'implicits']

    i = 0
    for line in tqdm(data[:]):
        i += 1

        if not counter_path.exists():
            counter_path.write_text('0', encoding='utf-8')
        # Note: on Windows, PowerShell's `Set-Content -Encoding UTF8` often writes a UTF-8 BOM.
        # Use utf-8-sig and sanitize to avoid ValueError like "\ufeff0".
        counter_raw = counter_path.read_text(encoding='utf-8-sig')
        counter_str = (counter_raw or '').strip().lstrip('\ufeff')
        counter_int = int(counter_str) if counter_str.isdigit() else 0

        # If the output file was cleared (size 0) but the counter wasn't reset,
        # we'd skip items and appear to "make progress" without writing output.
        # In that case, reset the counter to 0 so outputs are consistent.
        output_is_empty = (not output_path.exists()) or output_path.stat().st_size == 0
        if output_is_empty and counter_int > 0:
            counter_int = 0
            counter_path.write_text('0', encoding='utf-8')

        if i <= counter_int:
            continue

        sent, target, label, implicit = line[0], line[1], line[2], line[3]

        conversation = [system_role]
        context_step1, step_1_prompt = prompt_for_aspect_inferring(sent, target)
        conversation, aspect_expr = request_result(conversation, step_1_prompt, args.model_name)

        context_step2, step_2_prompt = prompt_for_opinion_inferring(context_step1, target, aspect_expr)
        conversation, opinion_expr = request_result(conversation, step_2_prompt, args.model_name)

        context_step3, step_3_prompt = prompt_for_polarity_inferring(context_step2, target, opinion_expr)
        conversation, polarity_expr = request_result(conversation, step_3_prompt, args.model_name)

        step_lb_prompt = prompt_for_polarity_label(context_step3, polarity_expr)
        conversation, output_lb = request_result(conversation, step_lb_prompt, args.model_name)

        output_lb = output_lb.lower().strip()
        output = 2
        for k, lb in enumerate(label_list):
            if lb in output_lb: output = k; break

        reasoning_text = sent + "\t" + target + "\t" + label_list[label] + "\t" + str(implicit) + "\n" + \
                         step_1_prompt + "\n" + aspect_expr + "\n" + \
                         step_2_prompt + "\n" + opinion_expr + "\n" + \
                         step_3_prompt + "\n" + polarity_expr + "\n" + \
                         step_lb_prompt + "\n" + output_lb + "\n" + \
                         'gold: ' + label_list[label] + "\tpredicted: " + label_list[output] + "\n\n\n"

        with open(output_path, 'a', encoding='utf8') as f:
            f.write(reasoning_text)

        counter_path.write_text(str(i), encoding='utf-8')

    # post-calculate results.
    if not output_path.exists():
        raise FileNotFoundError(
            f"Missing output file: {output_path}. "
            "No predictions were written yet. If this is the first run, let it process at least one item."
        )
    with open(output_path, 'r', encoding='utf-8-sig') as f:
        content = f.readlines()

    # Robust parsing: each sample starts with a 4-column TSV header line:
    #   sent\t target\t gold_label\t implicit_flag
    # and later contains a line:
    #   gold: <label>\tpredicted: <label>
    samples = []
    current = None
    for raw_line in content:
        line = (raw_line or '').strip()
        if not line:
            continue

        # Header line
        parts = line.split('\t')
        if len(parts) == 4 and parts[2] in label_list and parts[3] in {'0', '1'}:
            current = {
                'implicit': int(parts[3]),
            }
            samples.append(current)
            continue

        # Result line
        if line.startswith('gold:') and ('\tpredicted:' in line):
            if current is None:
                continue
            res = line.split('\t')
            try:
                gd = res[0].strip().split()[1].strip()
                pd = res[1].strip().split()[1].strip()
            except Exception:
                continue
            current['gold'] = label_dict.get(gd)
            current['pred'] = label_dict.get(pd)

    is_implicits = []
    gold_lbs = []
    outputs = []
    for s in samples:
        if 'gold' not in s or 'pred' not in s:
            continue
        is_implicits.append(int(s['implicit']))
        gold_lbs.append(s['gold'])
        outputs.append(s['pred'])

    for i, key in enumerate(keys):
        if i == 0:
            preds[key] += outputs
            golds[key] += gold_lbs
        else:
            if i == 1:
                ids = np.argwhere(np.array(is_implicits) == 0).flatten()
            else:
                ids = np.argwhere(np.array(is_implicits) == 1).flatten()
            preds[key] += [outputs[w] for w in ids]
            golds[key] += [gold_lbs[w] for w in ids]

    # Build index maps from overall index -> subset index
    explicit_pos = {}
    implicit_pos = {}
    e_i = 0
    i_i = 0
    for idx, flag in enumerate(is_implicits):
        if int(flag) == 0:
            explicit_pos[idx] = e_i
            e_i += 1
        else:
            implicit_pos[idx] = i_i
            i_i += 1

    result = report_score(golds, preds, mode='test')
    print(f'Zero-shot performance on {dataname} data by {args.model_name} + THOR:')
    print(result)

    # Test with a random sample
    sample_index = random.randint(0, len(data) - 1)  # Randomly select a sample index
    print("Testing with a random sample...")
    print("Sample content:", data[sample_index])
    print("Gold (total):", golds['total'][sample_index])
    print("Prediction (total):", preds['total'][sample_index])

    # Explicits/implicits are subsets, so we must remap indices.
    exp_idx = explicit_pos.get(sample_index)
    if exp_idx is not None:
        print("Gold (explicits):", golds['explicits'][exp_idx])
        print("Prediction (explicits):", preds['explicits'][exp_idx])
    else:
        print("Gold (explicits): Not applicable")
        print("Prediction (explicits): Not applicable")

    imp_idx = implicit_pos.get(sample_index)
    if imp_idx is not None:
        print("Gold (implicits):", golds['implicits'][imp_idx])
        print("Prediction (implicits):", preds['implicits'][imp_idx])
    else:
        print("Gold (implicits): Not applicable")
        print("Prediction (implicits): Not applicable")


if __name__ == '__main__':
    # Load configuration from YAML file
    repo_root = Path(__file__).resolve().parents[1]
    # Load .env if present (expects OPENAI_API_KEY=...)
    load_dotenv(repo_root / '.env')
    with open(repo_root / 'config' / 'main_config.yaml', 'r', encoding='utf-8') as config_file:
        config_data = yaml.safe_load(config_file)

    # Debugging statement to confirm the correct file is loaded
    safe_config = dict(config_data or {})
    if isinstance(safe_config.get('gpt_eval'), dict) and 'openai_key' in safe_config['gpt_eval']:
        safe_config['gpt_eval'] = dict(safe_config['gpt_eval'])
        safe_config['gpt_eval']['openai_key'] = '***REDACTED***'
    print("Debug: Loaded config data:", safe_config)

    gpt_cfg = config_data.get('gpt_eval', {}) if isinstance(config_data, dict) else {}
    args = argparse.Namespace(
        openai_key=gpt_cfg.get('openai_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_APIKEY'),
        data_name=gpt_cfg.get('data_name'),
        config=gpt_cfg.get('config'),
        model_name=gpt_cfg.get('model_name'),
    )
    if not args.openai_key:
        raise ValueError(
            "Missing OpenAI API key. Set OPENAI_API_KEY env var or set gpt_eval.openai_key in config/main_config.yaml"
        )
    if not args.data_name or not args.config or not args.model_name:
        raise ValueError("Missing gpt_eval.{data_name, config, model_name} in config/main_config.yaml")
    config_dict = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = SimpleNamespace(**config_dict)

    # Add model_name to the config namespace
    config.model_name = args.model_name

    openai.api_key = args.openai_key

    print(f"Debug: Using model: {config.model_name}")

    eval_run(args)
