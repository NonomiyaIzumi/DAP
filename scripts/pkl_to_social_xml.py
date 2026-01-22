"""Convert the generated social-media dataset from *.pkl to the SemEval-style XML format.

This repo's original datasets (laptops/restaurants) use a <sentences> XML format.
The generated social dataset is already preprocessed into a pickle with keys:
  - raw_texts
  - raw_aspect_terms
  - labels (0=positive, 1=negative, 2=neutral)
  - implicits (0=explicit, 1=implicit)

This script produces:
  data/social/Social_Train_v2_Implicit_Labeled.xml
  data/social/Social_Test_Gold_Implicit_Labeled.xml

Notes:
- We attempt to locate the aspect term inside the sentence to populate from/to.
- If the term cannot be found, we fall back to from=0,to=0.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import xml.etree.ElementTree as ET


LABEL_TO_POLARITY = {0: "positive", 1: "negative", 2: "neutral"}


def _find_span(text: str, term: str) -> tuple[int, int]:
    if not text or not term:
        return 0, 0

    idx = text.find(term)
    if idx != -1:
        return idx, idx + len(term)

    # case-insensitive fallback
    idx2 = text.lower().find(term.lower())
    if idx2 != -1:
        return idx2, idx2 + len(term)

    return 0, 0


def pkl_to_xml(pkl_path: Path, xml_path: Path) -> None:
    data = pickle.load(open(pkl_path, "rb"))

    texts = data.get("raw_texts", [])
    aspects = data.get("raw_aspect_terms", [])
    labels = data.get("labels", [])
    implicits = data.get("implicits", [0] * len(texts))

    if not (len(texts) == len(aspects) == len(labels) == len(implicits)):
        raise ValueError(
            "Mismatched lengths: "
            f"raw_texts={len(texts)}, raw_aspect_terms={len(aspects)}, "
            f"labels={len(labels)}, implicits={len(implicits)}"
        )

    root = ET.Element("sentences")

    for i, (text, term, label, implicit) in enumerate(zip(texts, aspects, labels, implicits), start=1):
        sent_el = ET.SubElement(root, "sentence", {"id": str(i)})
        text_el = ET.SubElement(sent_el, "text")
        text_el.text = str(text)

        aspect_terms_el = ET.SubElement(sent_el, "aspectTerms")

        polarity = LABEL_TO_POLARITY.get(int(label), "neutral")
        span_from, span_to = _find_span(str(text), str(term))

        ET.SubElement(
            aspect_terms_el,
            "aspectTerm",
            {
                "term": str(term),
                "polarity": polarity,
                "from": str(span_from),
                "to": str(span_to),
                "implicit_sentiment": "True" if int(implicit) == 1 else "False",
            },
        )

    # Pretty-print (Python 3.9+)
    try:
        ET.indent(ET.ElementTree(root), space="    ")
    except Exception:
        pass

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    social_dir = repo_root / "data" / "social"

    train_pkl = social_dir / "Social_Train_v2_Implicit_Labeled_preprocess_finetune.pkl"
    test_pkl = social_dir / "Social_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl"

    train_xml = social_dir / "Social_Train_v2_Implicit_Labeled.xml"
    test_xml = social_dir / "Social_Test_Gold_Implicit_Labeled.xml"

    if not train_pkl.exists():
        raise FileNotFoundError(f"Missing train pickle: {train_pkl}")
    if not test_pkl.exists():
        raise FileNotFoundError(f"Missing test pickle: {test_pkl}")

    pkl_to_xml(train_pkl, train_xml)
    pkl_to_xml(test_pkl, test_xml)

    print(f"Wrote: {train_xml}")
    print(f"Wrote: {test_xml}")


if __name__ == "__main__":
    main()
