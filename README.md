## 🧛🏼‍♀️⚡⚒️ THOR: Three-hop Reasoning for Implicit Sentiment
<a href="https://github.com/scofield7419/THOR-ISA">
  <img src="https://img.shields.io/badge/THOR-1.0-blue" alt="pytorch 1.8.1">
</a>
<a href="https://github.com/scofield7419/THOR-ISA" rel="nofollow">
  <img src="https://img.shields.io/badge/CoT-1.0-red" alt="pytorch 1.8.1">
</a>
<a href="https://huggingface.co/docs/transformers/model_doc/flan-t5" rel="nofollow">
  <img src="https://img.shields.io/badge/Flan-T5-purple" alt="Build Status">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.24.0-green" alt="Build Status">
</a>
<a href="https://pytorch.org" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.10.0-orange" alt="pytorch 1.8.1">
</a>


**The pytroch implementation of the ACL23 paper [Reasoning Implicit Sentiment with Chain-of-Thought Prompting](https://arxiv.org/abs/2305.11255)**

This workspace also includes an implementation path to reproduce the RVISA paper in the repo PDF (`2407.02340v1.pdf`):
- Stage 1: DO LLM generates rationales (TH-RE / TH-RA) + answer-based verification signal
- Stage 2: ED backbone (Flan-T5) multi-task fine-tuning with $(\alpha,\gamma)$ and verification task

----------
 ### 🎉 Visit the project page: [THOR-ISA](http://haofei.vip/THOR/)

----------


## Quick Links
- [Overview](#overview)
- [Code Usage](#code)
  - [Requirement](#requirement)
  - [Dataset](#data)
  - [LLMs](#llm)
  - [Run with Flan-T5](#runt5)
  - [Run with GPT-3.5](#GPT)
  - [Reproduce RVISA](#rvisa)
  - [Suggestions](#suggest)
- [MISC](#misc)


----------

## Overview<a name="overview" />

> While sentiment analysis systems try to determine the sentiment polarities of given targets based on the key opinion expressions in input texts, 
in implicit sentiment analysis (ISA) the opinion cues come in an implicit and obscure manner.

<p align="center">
  <img src="./figures/task.png" width="450"/>
</p>


> Thus detecting implicit sentiment requires the common-sense and multi-hop reasoning ability to infer the latent intent of opinion.
Inspired by the recent chain-of-thought (CoT) idea, in this work we introduce a *Three-hop Reasoning* (**THOR**) CoT framework to mimic the human-like reasoning process for ISA.
We design a three-step prompting principle for THOR to step-by-step induce the implicit aspect, opinion, and finally the sentiment polarity.

<p align="center">
  <img src="./figures/framework.png" width="1000"/>
</p>


----------


----------

## Code Usage<a name="code" />


----------
### Requirement<a name="requirement" />

``` bash 
conda create -n thor python=3.8
```

``` bash
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

```bash
pip install -r requirements.txt
```

----------

### Dataset<a name="data" />

SemEval14 Laptop ([laptops](data%2Flaptops)) and Restaurant ([restaurants](data%2Frestaurants)), with fine-grained target-level annotations.



----------
### LLMs<a name="llm" />

A. Use the Flan-T5 as the backbone LLM reasoner:
  - [google/flan-t5-base](https://huggingface.co/google/flan-t5-base),  
  - [google/flan-t5-large](https://huggingface.co/google/flan-t5-large), 
  - [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl),  
  - [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl),  

B. Evaluate with OpenAI [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)

C. Optional: Use Hugging Face Inference API for Flan-T5 (no local weights). Set in [config/config.yaml](config/config.yaml):
- `use_hf_inference: true`
- `hf_model_id: google/flan-t5-xxl` (or any hosted Flan-T5)
- Set env `HF_API_KEY` or `HUGGINGFACEHUB_API_TOKEN`

Note: HF Inference API mode supports zero-shot/eval only (no training).

----------
### Training and Evaluating with Flan-T5<a name="runt5" />

Use the [main.py](main.py) script. This repo now expects runtime settings from `config/main_config.yaml`.


Configure `config/main_config.yaml` (see `config/main_config.example.yaml`), then run:

```bash
python main.py
```

Core settings live in `config/config.yaml`.


----------

### Evaluating with GPT-3.5<a name="GPT" />

Go to the [eval_GPT](eval_GPT) fold, and run the [run_gpt_eval.py](eval_GPT%2Frun_gpt_eval.py) script:

```bash
python run_gpt_eval.py -k <openai_key> -d [restaurants|laptops]
```

Indicating your openai key. 
The reasoning traces and outputs of GPT for all instances are saved in `output_<data_name>.txt` file.

----------

### Reproduce RVISA<a name="rvisa" />

RVISA is the two-stage framework described in the paper PDF (`2407.02340v1.pdf`).

1) Install deps and set your OpenAI key (recommended via environment variable):

```bash
setx OPENAI_API_KEY "<your_key>"
```

2) Stage 1: generate rationales + verification signal (writes a PKL under `data/preprocessed/`):

```bash
python scripts/generate_rvisa_stage1.py --data-name restaurants --prompt-style th-re --teacher-model gpt-3.5-turbo
```

3) Stage 2: set `config/main_config.yaml`:
- `main.reasoning: rvisa`
- `main.zero_shot: false`
- `main.rvisa_data_path: data/preprocessed/<the_file_from_stage1>.pkl`
- (optional) `main.rvisa_alpha`, `main.rvisa_gamma`, `main.rvisa_use_verification`

Then run:

```bash
python main.py
```

Ablations from the paper map to:
- w/o VE: set `main.rvisa_use_verification: false`
- w/o VE and TH: generate stage-1 data with `--prompt-style reasoning` (or `zero-cot`) and set `main.rvisa_use_verification: false`

----------

### Suggestions<a name="suggest" />

- Suggest start with big enough LLM (e.g., `flan-t5-xl`), to better see the extraordinary reasoning ability.
- To tune the system with supervision, preferred with bigger batch size, and with large GPU ram; suggest with A100. 
- THOR is quite slower than the prompting mode.



----------

----------

## MISC<a name="misc" />

----------

### Citation

If you use this work, please kindly cite:

```
@inproceedings{FeiAcl23THOR,
  title={Reasoning Implicit Sentiment with Chain-of-Thought Prompting},
  author={Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua},
  booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
  pages = "1171--1182",
  year={2023}
}
```


----------


### Acknowledgement

This code is referred from following projects:
[CoT](https://arxiv.org/abs/2201.11903); 
[Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5);
[OpenAI-GPT3](https://platform.openai.com/docs/models/gpt-3-5);
[Transformer](https://github.com/huggingface/transformers),


----------


### License

The code is released under Apache License 2.0 for Noncommercial use only. 



