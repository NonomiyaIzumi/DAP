import os
import math
import torch
import numpy as np
import pickle as pkl
from src.utils import (
    prompt_direct_inferring,
    prompt_direct_inferring_masked,
    prompt_for_aspect_inferring,
    rvisa_prompt_th_re,
    rvisa_prompt_th_ra,
    rvisa_prompt_reasoning,
    rvisa_prompt_zero_cot,
    rvisa_prompt_verification,
)
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, config):
        self.config = config
        config.preprocessor = Preprocessor(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_data(self):
        cfg = self.config
        # RVISA uses a dedicated preprocessed file produced by scripts/generate_rvisa_stage1.py
        if cfg.reasoning == 'rvisa':
            rvisa_path = getattr(cfg, 'rvisa_data_path', '')
            if not rvisa_path:
                raise ValueError(
                    "Missing config.rvisa_data_path for reasoning='rvisa'. "
                    "Generate it via scripts/generate_rvisa_stage1.py"
                )
            payload = pkl.load(open(rvisa_path, 'rb'))
            train_data = payload['train']
            valid_data = payload['valid']
            test_data = payload['test']
        else:
            path = os.path.join(
                self.config.preprocessed_dir,
                '{}_{}_{}_{}.pkl'.format(cfg.data_name, cfg.reasoning, cfg.model_size, cfg.model_path).replace('/', '-'),
            )
            if os.path.exists(path):
                self.data = pkl.load(open(path, 'rb'))
            else:
                self.data = self.config.preprocessor.forward()
                pkl.dump(self.data, open(path, 'wb'))

            train_data, valid_data, test_data = self.data[:3]

        load_data = lambda dataset: DataLoader(
            MyDataset(dataset),
            num_workers=0,
            worker_init_fn=self.worker_init,
            shuffle=self.config.shuffle,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
        )
        train_loader, valid_loader, test_loader = map(load_data, [train_data, valid_data, test_data])
        train_loader.data_length, valid_loader.data_length, test_loader.data_length = math.ceil(
            len(train_data) / self.config.batch_size), \
            math.ceil(len(valid_data) / self.config.batch_size), \
            math.ceil(len(test_data) / self.config.batch_size)


        res = [train_loader, valid_loader, test_loader]

        return res, self.config

    def collate_fn(self, data):
        if not data:
            raise ValueError("Empty batch passed to collate_fn")

        if self.config.reasoning == 'rvisa':
            # Each item is a dict: {text, target, label, implicit, rationale, verification}
            texts = [d['text'] for d in data]
            targets = [d['target'] for d in data]
            input_labels = [int(d['label']) for d in data]
            implicits = [int(d.get('implicit', 0)) for d in data]
            rationales = [d.get('rationale', '') for d in data]
            ver_bools = [bool(d.get('verification', False)) for d in data]

            # --- Prediction task (label) ---
            pred_prompts = []
            for t, tgt in zip(texts, targets):
                t = ' '.join((t or '').split()[: self.config.max_length - 25])
                _, p = prompt_direct_inferring_masked(t, tgt)
                pred_prompts.append(p)

            pred_in = self.tokenizer.batch_encode_plus(
                pred_prompts,
                padding=True,
                return_tensors='pt',
                max_length=self.config.max_length,
                truncation=True,
            ).data
            pred_labels = [self.config.label_list[int(w)] for w in input_labels]
            pred_out = self.tokenizer.batch_encode_plus(
                pred_labels,
                max_length=3,
                padding=True,
                return_tensors='pt',
                truncation=True,
            ).data

            # --- Explanation task (rationale) ---
            exp_mode = getattr(self.config, 'rvisa_prompt_style', 'th-re')
            exp_prompts = []
            for t, tgt, y in zip(texts, targets, pred_labels):
                t = ' '.join((t or '').split()[: self.config.max_length - 25])
                if exp_mode == 'th-re':
                    exp_prompts.append(rvisa_prompt_th_re(t, tgt))
                elif exp_mode == 'th-ra':
                    exp_prompts.append(rvisa_prompt_th_ra(t, tgt, y))
                elif exp_mode == 'reasoning':
                    exp_prompts.append(rvisa_prompt_reasoning(t, tgt))
                elif exp_mode == 'zero-cot':
                    exp_prompts.append(rvisa_prompt_zero_cot(t, tgt))
                else:
                    raise ValueError(f"Unknown config.rvisa_prompt_style: {exp_mode}")

            exp_in = self.tokenizer.batch_encode_plus(
                exp_prompts,
                padding=True,
                return_tensors='pt',
                max_length=self.config.max_length,
                truncation=True,
            ).data
            exp_out = self.tokenizer.batch_encode_plus(
                rationales,
                padding=True,
                return_tensors='pt',
                max_length=self.config.max_length,
                truncation=True,
            ).data

            # --- Verification task (True/False) ---
            use_ver = bool(getattr(self.config, 'rvisa_use_verification', True))
            if use_ver:
                ver_inputs = [rvisa_prompt_verification(r) for r in rationales]
                ver_in = self.tokenizer.batch_encode_plus(
                    ver_inputs,
                    padding=True,
                    return_tensors='pt',
                    max_length=self.config.max_length,
                    truncation=True,
                ).data
                ver_targets = ['True' if b else 'False' for b in ver_bools]
                ver_out = self.tokenizer.batch_encode_plus(
                    ver_targets,
                    max_length=3,
                    padding=True,
                    return_tensors='pt',
                    truncation=True,
                ).data
            else:
                ver_in = {'input_ids': torch.zeros((len(texts), 1), dtype=torch.long), 'attention_mask': torch.ones((len(texts), 1), dtype=torch.long)}
                ver_out = {'input_ids': torch.zeros((len(texts), 1), dtype=torch.long), 'attention_mask': torch.ones((len(texts), 1), dtype=torch.long)}

            res = {
                'pred_input_ids': pred_in['input_ids'],
                'pred_input_masks': pred_in['attention_mask'],
                'pred_output_ids': pred_out['input_ids'],
                'pred_output_masks': pred_out['attention_mask'],
                'exp_input_ids': exp_in['input_ids'],
                'exp_input_masks': exp_in['attention_mask'],
                'exp_output_ids': exp_out['input_ids'],
                'exp_output_masks': exp_out['attention_mask'],
                'ver_input_ids': ver_in['input_ids'],
                'ver_input_masks': ver_in['attention_mask'],
                'ver_output_ids': ver_out['input_ids'],
                'ver_output_masks': ver_out['attention_mask'],
                'input_labels': torch.tensor(input_labels, dtype=torch.long),
                'implicits': torch.tensor(implicits, dtype=torch.long),
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        # Each item is a 4-tuple: (text, target, label, implicit)
        input_tokens, input_targets, input_labels, implicits = zip(*data)
        input_labels = [int(x) for x in input_labels]
        implicits = [int(x) for x in implicits]

        if self.config.reasoning == 'prompt':
            new_tokens = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                if self.config.zero_shot:
                    _, prompt = prompt_direct_inferring(line, input_targets[i])
                else:
                    _, prompt = prompt_direct_inferring_masked(line, input_targets[i])
                new_tokens.append(prompt)

            batch_input = self.tokenizer.batch_encode_plus(new_tokens, padding=True, return_tensors='pt',
                                                           max_length=self.config.max_length)
            batch_input = batch_input.data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(labels, max_length=3, padding=True,
                                                            return_tensors="pt").data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels),
                'implicits': torch.tensor(implicits),
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        elif self.config.reasoning == 'thor':

            new_tokens = []
            contexts_A = []
            for i, line in enumerate(input_tokens):
                line = ' '.join(line.split()[:self.config.max_length - 25])
                context_step1, prompt = prompt_for_aspect_inferring(line, input_targets[i])
                new_tokens.append(prompt)
                contexts_A.append(context_step1)

            batch_input = self.tokenizer.batch_encode_plus(
                new_tokens,
                padding=True,
                return_tensors='pt',
                max_length=self.config.max_length,
                truncation=True,
            ).data

            labels = [self.config.label_list[int(w)] for w in input_labels]
            batch_output = self.tokenizer.batch_encode_plus(
                labels,
                max_length=3,
                padding=True,
                return_tensors="pt",
                truncation=True,
            ).data

            batch_context_A = self.tokenizer.batch_encode_plus(
                contexts_A,
                padding=True,
                return_tensors='pt',
                max_length=self.config.max_length,
                truncation=True,
            ).data
            batch_targets = self.tokenizer.batch_encode_plus(
                list(input_targets),
                padding=True,
                return_tensors='pt',
                max_length=min(16, self.config.max_length),
                truncation=True,
            ).data

            res = {
                'input_ids': batch_input['input_ids'],
                'input_masks': batch_input['attention_mask'],
                'output_ids': batch_output['input_ids'],
                'output_masks': batch_output['attention_mask'],
                'input_labels': torch.tensor(input_labels, dtype=torch.long),
                'implicits': torch.tensor(implicits, dtype=torch.long),
                'context_A_ids': batch_context_A['input_ids'],
                'target_ids': batch_targets['input_ids'],
            }
            res = {k: v.to(self.config.device) for k, v in res.items()}
            return res

        else:
            raise ValueError('Choose correct reasoning mode: prompt, thor, or rvisa.')


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def read_file(self):
        dataname = self.config.dataname
        train_file = os.path.join(self.config.data_dir, dataname,
                                  '{}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        test_file = os.path.join(self.config.data_dir, dataname,
                                 '{}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'.format(dataname.capitalize()))
        train_data = pkl.load(open(train_file, 'rb'))
        test_data = pkl.load(open(test_file, 'rb'))

        n = len(train_data.get('raw_texts', []))
        if n == 0:
            raise ValueError(f"Empty train dataset loaded from: {train_file}")

        ids = np.arange(n)
        np.random.shuffle(ids)
        lens = 150

        valid_ids = ids[-lens:] if n > lens else ids
        train_ids = ids[:-lens] if n > lens else ids

        def _subset(d: dict, sel: np.ndarray) -> dict:
            out = {}
            for k, v in d.items():
                if isinstance(v, list):
                    out[k] = [v[int(i)] for i in sel]
                else:
                    out[k] = v
            return out

        valid_data = _subset(train_data, valid_ids)
        train_data = _subset(train_data, train_ids)
        return train_data, valid_data, test_data

    def transformer2indices(self, cur_data):
        res = []
        for i in range(len(cur_data['raw_texts'])):
            text = cur_data['raw_texts'][i]
            target = cur_data['raw_aspect_terms'][i]
            implicit = 0
            if 'implicits' in cur_data:
                implicit = cur_data['implicits'][i]
            label = cur_data['labels'][i]
            implicit = int(implicit)
            res.append((text, target, label, implicit))
        return res

    def forward(self):
        modes = 'train valid test'.split()
        dataset = self.read_file()
        res = []
        for i, mode in enumerate(modes):
            data = self.transformer2indices(dataset[i])
            res.append(data)
        return res
