"""Convert SemEval-style ABSA XML files to the repo's *_preprocess_finetune.pkl schema.

This repo primarily consumes dataset pickles with keys:
  - raw_texts
  - raw_aspect_terms
  - bert_tokens
  - aspect_masks
  - implicits
  - labels

The original datasets (laptops/restaurants) also exist as SemEval XML:
  data/<domain>/*_Train_v2*.xml
  data/<domain>/*_Test_Gold*.xml

This script recreates the missing XML -> PKL conversion pipeline.

Example:
  python scripts/xml_to_preprocess_finetune_pkl.py \
    --xml data/social/Social_Test_Gold_Implicit_Labeled.xml \
    --out data/social/Social_Test_Gold_Implicit_Labeled_preprocess_finetune_fromxml.pkl

Notes:
- We create one PKL row per <aspectTerm>.
- Polarity mapping: positive->0, negative->1, neutral->2.
  'conflict' is optionally skipped (default) or mapped to neutral.
- aspect_masks are computed by matching tokenized target span in tokenized text.
"""

from __future__ import annotations

import argparse
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET
import html

from transformers import BertTokenizerFast


LABELS = {"positive": 0, "negative": 1, "neutral": 2}


@dataclass(frozen=True)
class Row:
    text: str
    target: str
    label: int
    implicit: bool
    span_from: int | None = None
    span_to: int | None = None


def normalize_for_span(text: str) -> str:
    # Preserve string length as much as possible to keep XML from/to offsets meaningful.
    # Avoid collapsing whitespace.
    out = html.unescape(text)
    return out


def normalize_for_storage(text: str) -> str:
    # Match the typical stored format in existing pickles: lowercased and trimmed.
    out = normalize_for_span(text)
    return out.strip().lower()


def normalize_term(term: str) -> str:
    out = html.unescape(term)
    return out.strip().lower()


def legacy_text_store(text: str) -> str:
    # Legacy pickles appear to store lowercased text but preserve leading/trailing whitespace
    # and preserve NBSP (\u00a0).
    return html.unescape(text).lower()


def legacy_term_store(term: str) -> str:
    # Legacy pickles appear to preserve original casing and whitespace for aspect terms.
    return html.unescape(term)


def build_bert_fields(
    tokenizer: BertTokenizerFast,
    text: str,
    target: str,
    max_length: int = 128,
    span_from: int | None = None,
    span_to: int | None = None,
    legacy_mask: bool = False,
) -> tuple[list[int], list[int]]:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids: list[int] = encoded["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offsets = encoded.get("offset_mapping") or [(0, 0)] * len(tokens)

    aspect_mask = [0] * len(tokens)

    # Prefer XML-provided char span. This resolves repeated targets like "battery life".
    if (
        span_from is not None
        and span_to is not None
        and isinstance(span_from, int)
        and isinstance(span_to, int)
        and span_to > span_from
    ):
        if legacy_mask:
            # Legacy behavior: find the token whose offset start equals `from`, then mark
            # the next N tokens where N=len(tokenize(target)). This can differ from full
            # span-overlap marking when the surface form is concatenated in the text.
            start_idx = None
            for i, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if s == span_from:
                    start_idx = i
                    break

            if start_idx is not None:
                target_tokens = tokenizer.tokenize(target)
                n = max(1, len(target_tokens))
                for j in range(start_idx, min(start_idx + n, len(aspect_mask))):
                    aspect_mask[j] = 1
                return input_ids, aspect_mask

        # Default (recommended): mark every token that overlaps the character span.
        for i, (s, e) in enumerate(offsets):
            if s == e:
                continue  # special tokens
            if (s < span_to) and (e > span_from):
                aspect_mask[i] = 1
        return input_ids, aspect_mask

    # Fallback: token-span matching by target tokens (first occurrence)
    target_tokens = tokenizer.tokenize(target)
    if target_tokens:
        for start in range(0, len(tokens) - len(target_tokens) + 1):
            if tokens[start : start + len(target_tokens)] == target_tokens:
                for j in range(start, start + len(target_tokens)):
                    aspect_mask[j] = 1
                break

    return input_ids, aspect_mask


def _bool_from_xml(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    return default


def iter_rows_from_xml(
    xml_path: Path,
    *,
    skip_conflict: bool = True,
    conflict_to_neutral: bool = False,
    implicit_default: bool = False,
    legacy: bool = False,
) -> Iterable[Row]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sentence in root.findall(".//sentence"):
        text_el = sentence.find("text")
        raw_text = (text_el.text or "") if text_el is not None else ""
        text_span = normalize_for_span(raw_text)
        text_store = legacy_text_store(raw_text) if legacy else normalize_for_storage(raw_text)
        if not text_store:
            continue

        aspect_terms = sentence.findall(".//aspectTerms/aspectTerm")
        for at in aspect_terms:
            term_raw = at.attrib.get("term") or ""
            term_store = legacy_term_store(term_raw) if legacy else normalize_term(term_raw)
            polarity = (at.attrib.get("polarity") or "").strip().lower()
            if not term_store or not polarity:
                continue

            if polarity == "conflict":
                if skip_conflict and not conflict_to_neutral:
                    continue
                polarity = "neutral"

            if polarity not in LABELS:
                continue

            implicit_attr = at.attrib.get("implicit_sentiment")
            implicit = _bool_from_xml(implicit_attr, default=implicit_default)

            span_from = at.attrib.get("from")
            span_to = at.attrib.get("to")
            try:
                sf = int(span_from) if span_from is not None else None
                st = int(span_to) if span_to is not None else None
            except Exception:
                sf, st = None, None

            yield (
                Row(
                    text=text_store,
                    target=term_store,
                    label=LABELS[polarity],
                    implicit=implicit,
                    span_from=sf,
                    span_to=st,
                ),
                text_span,
            )


def write_pkl(
    rows: list[tuple[Row, str]],
    out_path: Path,
    tokenizer: BertTokenizerFast,
    max_length: int,
    *,
    legacy_mask: bool = False,
) -> None:
    obj = {
        "raw_texts": [],
        "raw_aspect_terms": [],
        "bert_tokens": [],
        "aspect_masks": [],
        "implicits": [],
        "labels": [],
    }

    for r, text_span in rows:
        bert_tokens, aspect_mask = build_bert_fields(
            tokenizer,
            text_span,
            r.target,
            max_length=max_length,
            span_from=r.span_from,
            span_to=r.span_to,
            legacy_mask=legacy_mask,
        )
        obj["raw_texts"].append(r.text)
        obj["raw_aspect_terms"].append(r.target)
        obj["bert_tokens"].append(bert_tokens)
        obj["aspect_masks"].append(aspect_mask)
        obj["implicits"].append(bool(r.implicit))
        obj["labels"].append(int(r.label))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True, help="Path to input XML file")
    parser.add_argument("--out", type=str, required=True, help="Path to output PKL file")
    parser.add_argument("--bert", type=str, default="bert-base-uncased", help="BERT tokenizer name")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")

    parser.add_argument(
        "--skip-conflict",
        action="store_true",
        help="Skip polarity='conflict' aspect terms (default).",
    )
    parser.add_argument(
        "--conflict-to-neutral",
        action="store_true",
        help="Map polarity='conflict' to neutral instead of skipping.",
    )
    parser.add_argument(
        "--implicit-default",
        action="store_true",
        help="If XML lacks implicit_sentiment attribute, treat as implicit=True (default False).",
    )

    parser.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Mimic the repo's legacy preprocess behavior (preserve NBSP/whitespace in stored fields, "
            "preserve aspect term casing/whitespace, and build aspect_masks using span-start + len(tokenize(term)))."
        ),
    )

    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_path = Path(args.out)

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML: {xml_path}")

    tokenizer = BertTokenizerFast.from_pretrained(args.bert)

    rows = list(
        iter_rows_from_xml(
            xml_path,
            skip_conflict=args.skip_conflict or not args.conflict_to_neutral,
            conflict_to_neutral=bool(args.conflict_to_neutral),
            implicit_default=bool(args.implicit_default),
            legacy=bool(args.legacy),
        )
    )

    if not rows:
        raise ValueError("No aspectTerm rows found in XML (check XML structure)")

    write_pkl(rows, out_path, tokenizer, max_length=int(args.max_length), legacy_mask=bool(args.legacy))
    print(f"Wrote: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
