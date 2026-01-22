import argparse
import json
import os
import pickle as pkl
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import openai
import backoff
from tqdm import tqdm
from dotenv import load_dotenv
import httpx


LABEL_LIST = ["positive", "negative", "neutral"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_polarity_fcfs(text: str) -> Optional[str]:
    """Parse label using First-Come-First-Served (first mention)."""
    if not text:
        return None
    lower = text.lower()
    hits: List[Tuple[int, str]] = []
    for label in LABEL_LIST:
        idx = lower.find(label)
        if idx >= 0:
            hits.append((idx, label))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[0][1]


def build_prompt(
    *,
    sentence: str,
    target: str,
    gold_label: Optional[str],
    prompt_style: str,
) -> str:
    """Return a single-shot prompt that asks the DO LLM to output a rationale including final polarity.

    prompt_style:
      - th-re: three-hop reasoning (no gold label)
      - th-ra: three-hop rationalization (with gold label)
      - reasoning: generic 'why' rationale (no three-hop cues)
      - zero-cot: generic + 'let's think step by step'

    We intentionally keep the template close to the paper text in `paper_extracted.txt`.
    """

    sentence = normalize_ws(sentence)
    target = normalize_ws(target)

    if prompt_style == "th-re":
        return (
            f'Given the sentence "{sentence}", what is the sentiment polarity towards {target}, why? '
            "Let’s think step by step. "
            f"The mentioned aspect towards {target} is about ... "
            f"The underlying opinion towards {target} is about ... "
            f"Therefore, the sentiment polarity towards {target} is ..."
        )

    if prompt_style == "th-ra":
        if gold_label is None:
            raise ValueError("gold_label is required for prompt_style='th-ra'")
        return (
            f'Given the sentence "{sentence}", the sentiment polarity towards {target} is {gold_label}, why? '
            "Let’s think step by step. "
            f"The mentioned aspect towards {target} is about ... "
            f"The underlying opinion towards {target} is about ... "
            f"Therefore, the sentiment polarity towards {target} is ..."
        )

    if prompt_style == "reasoning":
        return (
            f'Given the sentence "{sentence}", what is the sentiment polarity towards {target}, why? '
            "Explain your reasoning."
        )

    if prompt_style == "zero-cot":
        return (
            f'Given the sentence "{sentence}", what is the sentiment polarity towards {target}? '
            "Let’s think step by step."
        )

    raise ValueError(f"Unknown prompt_style: {prompt_style}")


def build_verification_input(rationale: str) -> str:
    rationale = (rationale or "").strip()
    return (
        "Given the rationale below, please verify whether it is reasonable. "
        "Return True or False.\n\n"
        f"{rationale}"
    )


def build_openai_client(*, api_key: str, timeout_total_s: float, timeout_connect_s: float) -> openai.OpenAI:
    timeout = httpx.Timeout(timeout_total_s, connect=timeout_connect_s)
    # We handle retries ourselves via backoff (below) for clearer behavior.
    return openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)


@dataclass
class ExampleOut:
    text: str
    target: str
    label: int
    implicit: int
    rationale: str
    teacher_pred_label: Optional[str]
    verification: bool


def read_base_split(data_name: str, split: str) -> Dict[str, Any]:
    base_dir = _repo_root() / "data" / data_name
    if split == "train":
        path = base_dir / f"{data_name.capitalize()}_Train_v2_Implicit_Labeled_preprocess_finetune.pkl"
    elif split == "test":
        path = base_dir / f"{data_name.capitalize()}_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl"
    else:
        raise ValueError("split must be train or test")

    with open(path, "rb") as f:
        return pkl.load(f)


def to_items(cur_data: Dict[str, Any]) -> List[Tuple[str, str, int, int]]:
    items = []
    n = len(cur_data.get("raw_texts", []))
    for i in range(n):
        text = cur_data["raw_texts"][i]
        target = cur_data["raw_aspect_terms"][i]
        label = int(cur_data["labels"][i])
        implicit = int(cur_data.get("implicits", [0] * n)[i])
        items.append((text, target, label, implicit))
    return items


def deterministic_split(items: List[Any], valid_size: int, seed: int) -> Tuple[List[Any], List[Any]]:
    import random

    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)

    if len(items) <= valid_size:
        return items, items

    valid_idxs = idxs[-valid_size:]
    train_idxs = idxs[:-valid_size]
    train = [items[i] for i in train_idxs]
    valid = [items[i] for i in valid_idxs]
    return train, valid


def main() -> None:
    repo_root = _repo_root()
    load_dotenv(repo_root / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-name", required=True, choices=["restaurants", "laptops", "social"])
    parser.add_argument("--out", required=False, default="")
    parser.add_argument("--prompt-style", required=False, default="th-re", choices=["th-re", "th-ra", "reasoning", "zero-cot"])
    parser.add_argument("--teacher-model", required=False, default="o4-mini-2025-04-16")
    parser.add_argument("--openai-key", required=False, default="")
    parser.add_argument("--config", required=False, default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=int, default=150)
    parser.add_argument("--max-items", type=int, default=0, help="Debug: cap number of training items (0 = all)")
    parser.add_argument("--resume-jsonl", default="", help="Optional JSONL cache file to resume from")

    parser.add_argument("--timeout-total", type=float, default=90.0, help="Total request timeout in seconds")
    parser.add_argument("--timeout-connect", type=float, default=15.0, help="Connect timeout in seconds")
    parser.add_argument("--retry-max-tries", type=int, default=8, help="Max retry attempts for transient OpenAI errors")

    args = parser.parse_args()

    # Resolve key: CLI > env/.env > main_config (if present)
    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        main_cfg_path = _repo_root() / "config" / "main_config.yaml"
        if main_cfg_path.exists():
            try:
                main_cfg = load_yaml(str(main_cfg_path))
                api_key = (main_cfg.get("gpt_eval", {}) or {}).get("openai_key", "") or api_key
            except Exception:
                pass

    if not api_key:
        raise RuntimeError(
            "Missing OpenAI key. Set OPENAI_API_KEY env var or pass --openai-key."
        )

    client = build_openai_client(
        api_key=api_key,
        timeout_total_s=float(args.timeout_total),
        timeout_connect_s=float(args.timeout_connect),
    )

    retryable_errors = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
        openai.APIStatusError,
    )

    @backoff.on_exception(
        backoff.expo,
        retryable_errors,
        max_tries=lambda: int(args.retry_max_tries),
        jitter=backoff.full_jitter,
    )
    def openai_chat(model: str, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = client.chat.completions.create(model=model, messages=messages)
        return (resp.choices[0].message.content or "").strip()

    out_path = args.out.strip()
    if not out_path:
        out_dir = _repo_root() / "data" / "preprocessed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{args.data_name}_rvisa_{args.prompt_style}_{args.teacher_model}.pkl")

    cache_jsonl = args.resume_jsonl.strip()
    if not cache_jsonl:
        cache_jsonl = str(Path(out_path).with_suffix(".jsonl"))

    # Load base data
    train_raw = read_base_split(args.data_name, "train")
    test_raw = read_base_split(args.data_name, "test")
    train_items_all = to_items(train_raw)
    test_items = to_items(test_raw)

    if args.max_items and args.max_items > 0:
        train_items_all = train_items_all[: args.max_items]

    train_items, valid_items = deterministic_split(train_items_all, args.valid_size, args.seed)

    # Resume cache
    cached: Dict[Tuple[str, str, int, int], ExampleOut] = {}
    cache_file = Path(cache_jsonl)
    if cache_file.exists():
        skipped_lines = 0
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Likely a partially-written line due to interruption.
                    skipped_lines += 1
                    continue
                key = (obj["text"], obj["target"], int(obj["label"]), int(obj["implicit"]))
                cached[key] = ExampleOut(
                    text=obj["text"],
                    target=obj["target"],
                    label=int(obj["label"]),
                    implicit=int(obj["implicit"]),
                    rationale=obj.get("rationale", ""),
                    teacher_pred_label=obj.get("teacher_pred_label"),
                    verification=bool(obj.get("verification", False)),
                )

        if skipped_lines:
            print(f"Warning: skipped {skipped_lines} malformed lines in resume cache: {cache_jsonl}")

    system = "You are an expert of sentiment and opinion analysis."

    def process_one(item: Tuple[str, str, int, int]) -> ExampleOut:
        text, target, label, implicit = item
        gold = LABEL_LIST[label]

        prompt = build_prompt(
            sentence=text,
            target=target,
            gold_label=gold,
            prompt_style=args.prompt_style,
        )
        rationale = openai_chat(args.teacher_model, system, prompt)
        rationale = rationale.strip()

        pred_label = parse_polarity_fcfs(rationale)
        verification = bool(pred_label == gold)

        return ExampleOut(
            text=text,
            target=target,
            label=label,
            implicit=implicit,
            rationale=rationale,
            teacher_pred_label=pred_label,
            verification=verification,
        )

    def run_split(items: List[Tuple[str, str, int, int]], split_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in tqdm(items, desc=f"Stage-1 {split_name}"):
            key = (item[0], item[1], int(item[2]), int(item[3]))
            if key in cached:
                ex = cached[key]
            else:
                ex = process_one(item)
                with open(cache_file, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "text": ex.text,
                                "target": ex.target,
                                "label": ex.label,
                                "implicit": ex.implicit,
                                "rationale": ex.rationale,
                                "teacher_pred_label": ex.teacher_pred_label,
                                "verification": ex.verification,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            out.append(
                {
                    "text": ex.text,
                    "target": ex.target,
                    "label": ex.label,
                    "implicit": ex.implicit,
                    "rationale": ex.rationale,
                    "verification": ex.verification,
                }
            )
        return out

    train_aug = run_split(train_items, "train")
    valid_aug = run_split(valid_items, "valid")
    test_aug = run_split(test_items, "test")

    payload = {
        "meta": {
            "data_name": args.data_name,
            "prompt_style": args.prompt_style,
            "teacher_model": args.teacher_model,
            "seed": args.seed,
            "valid_size": args.valid_size,
        },
        "train": train_aug,
        "valid": valid_aug,
        "test": test_aug,
    }

    with open(out_path, "wb") as f:
        pkl.dump(payload, f)

    print(f"Wrote RVISA stage-1 dataset: {out_path}")
    print(f"Cache JSONL: {cache_jsonl}")


if __name__ == "__main__":
    main()
