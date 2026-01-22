import pickle as pkl
import random
from pathlib import Path

from transformers import BertTokenizerFast


LABELS = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}


def build_bert_fields(tokenizer: BertTokenizerFast, text: str, target: str, max_length: int = 128):
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    target_tokens = tokenizer.tokenize(target)

    aspect_mask = [0] * len(tokens)
    if target_tokens:
        # Find target token span in the token list (includes [CLS] at 0)
        for start in range(0, len(tokens) - len(target_tokens) + 1):
            if tokens[start : start + len(target_tokens)] == target_tokens:
                for j in range(start, start + len(target_tokens)):
                    aspect_mask[j] = 1
                break

    return input_ids, aspect_mask


def minor_augment(text: str) -> str:
    # Very light perturbations to make a "new" split without changing meaning too much.
    replacements = {
        " very ": " really ",
        " really ": " quite ",
        " good ": " nice ",
        " great ": " excellent ",
        " bad ": " awful ",
    }
    out = f" {text.strip()} "
    for a, b in replacements.items():
        if a in out and random.random() < 0.35:
            out = out.replace(a, b, 1)
    out = out.strip()

    if random.random() < 0.20:
        out = out.replace(".", "!") if out.endswith(".") else out
    if random.random() < 0.10:
        out = out + ""
    return out


def write_dataset(out_path: Path, rows: list[dict], tokenizer: BertTokenizerFast):
    obj = {
        "raw_texts": [],
        "raw_aspect_terms": [],
        "bert_tokens": [],
        "aspect_masks": [],
        "implicits": [],
        "labels": [],
    }

    for r in rows:
        text = r["text"].strip()
        target = r["target"].strip()
        label = int(r["label"])
        implicit = bool(r["implicit"])

        bert_tokens, aspect_mask = build_bert_fields(tokenizer, text, target)

        obj["raw_texts"].append(text)
        obj["raw_aspect_terms"].append(target)
        obj["bert_tokens"].append(bert_tokens)
        obj["aspect_masks"].append(aspect_mask)
        obj["implicits"].append(implicit)
        obj["labels"].append(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(obj, f)


def main():
    random.seed(42)

    repo_root = Path(__file__).resolve().parents[1]
    src_train = repo_root / "data" / "restaurants" / "Restaurants_Train_v2_Implicit_Labeled_preprocess_finetune.pkl"

    out_dir = repo_root / "data" / "toy_restaurants"
    out_train = out_dir / "Toy_restaurants_Train_v2_Implicit_Labeled_preprocess_finetune.pkl"
    out_test = out_dir / "Toy_restaurants_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Create a small hand-crafted test set (fast to run, easy to interpret)
    test_rows = [
        {"text": "The service was lightning fast.", "target": "service", "label": LABELS["positive"], "implicit": False},
        {"text": "We waited forever for our food.", "target": "food", "label": LABELS["negative"], "implicit": True},
        {"text": "The soup tasted like it came from a can.", "target": "soup", "label": LABELS["negative"], "implicit": True},
        {"text": "Portions were okay, nothing special.", "target": "portions", "label": LABELS["neutral"], "implicit": False},
        {"text": "The patio view made the meal.", "target": "patio", "label": LABELS["positive"], "implicit": True},
        {"text": "My steak arrived cold in the middle.", "target": "steak", "label": LABELS["negative"], "implicit": True},
        {"text": "The staff kept checking on us.", "target": "staff", "label": LABELS["positive"], "implicit": True},
        {"text": "The music was so loud we could not talk.", "target": "music", "label": LABELS["negative"], "implicit": True},
        {"text": "Prices are fair for the location.", "target": "prices", "label": LABELS["positive"], "implicit": False},
        {"text": "The dessert menu is small.", "target": "dessert", "label": LABELS["neutral"], "implicit": False},
        {"text": "My water glass stayed empty.", "target": "water", "label": LABELS["negative"], "implicit": True},
        {"text": "The waiter remembered our order without writing it down.", "target": "waiter", "label": LABELS["positive"], "implicit": True},
        {"text": "The table was sticky.", "target": "table", "label": LABELS["negative"], "implicit": True},
        {"text": "The ramen broth was rich and comforting.", "target": "broth", "label": LABELS["positive"], "implicit": False},
        {"text": "The kitchen handled my allergy request carefully.", "target": "allergy", "label": LABELS["positive"], "implicit": True},
        {"text": "The menu descriptions are confusing.", "target": "menu", "label": LABELS["negative"], "implicit": True},
        {"text": "The chairs are fine.", "target": "chairs", "label": LABELS["neutral"], "implicit": False},
        {"text": "The cashier rushed me.", "target": "cashier", "label": LABELS["negative"], "implicit": True},
        {"text": "The noodles were perfectly cooked.", "target": "noodles", "label": LABELS["positive"], "implicit": False},
        {"text": "The restroom was spotless.", "target": "restroom", "label": LABELS["positive"], "implicit": False},
        {"text": "The appetizer arrived after the main course.", "target": "appetizer", "label": LABELS["negative"], "implicit": True},
        {"text": "The atmosphere is calm.", "target": "atmosphere", "label": LABELS["positive"], "implicit": False},
        {"text": "The sauce is too salty.", "target": "sauce", "label": LABELS["negative"], "implicit": False},
        {"text": "Parking is limited.", "target": "parking", "label": LABELS["neutral"], "implicit": False},
        {"text": "Our server disappeared for twenty minutes.", "target": "server", "label": LABELS["negative"], "implicit": True},
        {"text": "The chef greeted us at the table.", "target": "chef", "label": LABELS["positive"], "implicit": True},
        {"text": "The rice is plain.", "target": "rice", "label": LABELS["neutral"], "implicit": False},
        {"text": "The pizza crust was burnt.", "target": "crust", "label": LABELS["negative"], "implicit": False},
        {"text": "They refilled the bread basket without us asking.", "target": "bread", "label": LABELS["positive"], "implicit": True},
        {"text": "The line moved quickly.", "target": "line", "label": LABELS["positive"], "implicit": True},
    ]

    # Create a small train set by sampling and lightly augmenting existing restaurants train.
    # This avoids edge cases in the training/valid split logic (it expects >150 items).
    with open(src_train, "rb") as f:
        train_src = pkl.load(f)

    n_src = len(train_src["raw_texts"])
    sample_n = min(500, n_src)
    ids = random.sample(range(n_src), sample_n)

    train_rows = []
    for idx in ids:
        train_rows.append(
            {
                "text": minor_augment(train_src["raw_texts"][idx]),
                "target": train_src["raw_aspect_terms"][idx],
                "label": int(train_src["labels"][idx]),
                "implicit": bool(train_src["implicits"][idx]),
            }
        )

    write_dataset(out_test, test_rows, tokenizer)
    write_dataset(out_train, train_rows, tokenizer)

    print(f"Wrote test:  {out_test} ({len(test_rows)} rows)")
    print(f"Wrote train: {out_train} ({len(train_rows)} rows)")


if __name__ == "__main__":
    main()
