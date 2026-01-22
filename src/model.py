import os
import torch
import torch.nn as nn
import requests
from transformers import AutoTokenizer, T5ForConditionalGeneration


class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        self.use_hf_inference = bool(getattr(config, 'use_hf_inference', False))
        self.hf_model_id = getattr(config, 'hf_model_id', None) or config.model_path
        self.hf_api_key = (
            getattr(config, 'hf_api_key', None)
            or os.getenv('HF_API_KEY')
            or os.getenv('HUGGINGFACEHUB_API_TOKEN')
            or os.getenv('HUGGINGFACE_API_TOKEN')
        )
        self.hf_timeout = int(getattr(config, 'hf_timeout', 60))
        self.hf_max_new_tokens = int(getattr(config, 'hf_max_new_tokens', 64))

        if self.use_hf_inference:
            self.engine = None
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            if not self.hf_api_key:
                raise ValueError(
                    'Missing Hugging Face API key. Set HF_API_KEY or HUGGINGFACEHUB_API_TOKEN.'
                )
        else:
            self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    def forward(self, **kwargs):
        if self.use_hf_inference:
            raise RuntimeError('HF Inference API mode does not support training (forward/loss).')
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def _hf_generate_texts(self, input_ids, input_masks):
        if self.tokenizer is None:
            raise RuntimeError('Tokenizer missing for HF Inference API mode.')

        prompts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in input_ids
        ]
        url = f'https://api-inference.huggingface.co/models/{self.hf_model_id}'
        headers = {'Authorization': f'Bearer {self.hf_api_key}'}
        results = []
        for prompt in prompts:
            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': self.hf_max_new_tokens,
                },
                'options': {'wait_for_model': True},
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=self.hf_timeout)
            if resp.status_code != 200:
                raise RuntimeError(f'HF Inference API error {resp.status_code}: {resp.text}')
            data = resp.json()
            if isinstance(data, dict) and data.get('error'):
                raise RuntimeError(f'HF Inference API error: {data.get("error")}')
            if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
                results.append(data[0]['generated_text'])
            else:
                results.append(str(data))
        return results

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        if self.use_hf_inference:
            return [o.strip() for o in self._hf_generate_texts(input_ids, input_masks)]
        else:
            output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                          max_length=self.config.max_length)
            dec = [self.tokenizer.decode(ids) for ids in output]
            output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
            return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        if self.use_hf_inference:
            dec = self._hf_generate_texts(input_ids, input_masks)
        else:
            output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200)
            dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.replace('<pad>', '').replace('</s>', '').strip(), 0) for w in dec]
        return output
