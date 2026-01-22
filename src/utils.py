import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


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
    new_context = context + ' The mentioned aspect is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, what is the implicit opinion towards the mentioned aspect of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + f' The opinion towards the mentioned aspect of {target} is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the sentiment polarity towards {target}?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr):
    prompt = context + f' The sentiment polarity is {polarity_expr}.' + ' Based on these contexts, summarize and return the sentiment polarity only, such as positive, neutral, or negative.'
    return prompt


def rvisa_prompt_th_re(context: str, target: str) -> str:
    context = ' '.join((context or '').split())
    target = (target or '').strip()
    return (
        f'Given the sentence "{context}", what is the sentiment polarity towards {target}, why? '
        "Let’s think step by step. "
        f"The mentioned aspect towards {target} is about ... "
        f"The underlying opinion towards {target} is about ... "
        f"Therefore, the sentiment polarity towards {target} is ..."
    )


def rvisa_prompt_th_ra(context: str, target: str, gold_label: str) -> str:
    context = ' '.join((context or '').split())
    target = (target or '').strip()
    gold_label = (gold_label or '').strip()
    return (
        f'Given the sentence "{context}", the sentiment polarity towards {target} is {gold_label}, why? '
        "Let’s think step by step. "
        f"The mentioned aspect towards {target} is about ... "
        f"The underlying opinion towards {target} is about ... "
        f"Therefore, the sentiment polarity towards {target} is ..."
    )


def rvisa_prompt_reasoning(context: str, target: str) -> str:
    context = ' '.join((context or '').split())
    target = (target or '').strip()
    return f'Given the sentence "{context}", what is the sentiment polarity towards {target}, why? Explain your reasoning.'


def rvisa_prompt_zero_cot(context: str, target: str) -> str:
    context = ' '.join((context or '').split())
    target = (target or '').strip()
    return f'Given the sentence "{context}", what is the sentiment polarity towards {target}? Let’s think step by step.'


def rvisa_prompt_verification(rationale: str) -> str:
    rationale = (rationale or '').strip()
    return (
        "Given the rationale below, please verify whether it is reasonable. Return True or False.\n\n"
        + rationale
    )


def parse_polarity_fcfs(text: str, label_list=None):
    """Parse first-mentioned polarity label (FCFS). Returns index or 0 if unknown."""
    if label_list is None:
        label_list = ['positive', 'negative', 'neutral']
    if not text:
        return 0
    lower = text.lower()
    hits = []
    for idx, lb in enumerate(label_list):
        pos = lower.find(lb)
        if pos >= 0:
            hits.append((pos, idx))
    if not hits:
        return 0
    hits.sort(key=lambda x: x[0])
    return hits[0][1]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res
