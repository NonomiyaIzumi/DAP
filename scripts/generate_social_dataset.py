import pickle as pkl
import random
from pathlib import Path


LABELS = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}


def dummy_bert_fields(text: str, target: str, max_len: int = 48):
    # This repo's data loader only uses raw_texts/raw_aspect_terms/labels/implicits.
    # We still provide bert_tokens/aspect_masks to match the existing schema.
    words = text.strip().split()
    n = min(len(words) + 2, max_len)
    bert_tokens = [101] + [1100 + i for i in range(n - 2)] + [102]

    # Mark a token as aspect if it roughly matches the target (very approximate).
    aspect_masks = [0] * n
    target_words = target.lower().split()
    if target_words and n > 2:
        # Mark token 2 as aspect by default; purely for schema compatibility.
        aspect_masks[2 if n > 2 else 1] = 1

    return bert_tokens, aspect_masks


def write_pickle(out_path: Path, rows: list[dict]):
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

        bert_tokens, aspect_masks = dummy_bert_fields(text, target)

        obj["raw_texts"].append(text)
        obj["raw_aspect_terms"].append(target)
        obj["bert_tokens"].append(bert_tokens)
        obj["aspect_masks"].append(aspect_masks)
        obj["implicits"].append(implicit)
        obj["labels"].append(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(obj, f)


def main():
    random.seed(7)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "data" / "social"

    train_path = out_dir / "Social_Train_v2_Implicit_Labeled_preprocess_finetune.pkl"
    test_path = out_dir / "Social_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl"

    # Social-media style, Vietnamese, synthetic examples (original text; not scraped).
    # implicit=True means sentiment is implied rather than explicitly stated.
    base = [
        # --- Product/app/service style ---
        {"text": "Update xong app hết lag luôn 😭🙏", "target": "app", "label": LABELS["positive"], "implicit": True},
        {"text": "Camera chụp đêm ổn áp phết 📸✨", "target": "camera", "label": LABELS["positive"], "implicit": False},
        {"text": "Pin tụt 20% trong 10 phút... chịu 😵", "target": "pin", "label": LABELS["negative"], "implicit": True},
        {"text": "Ship tới sớm hơn dự kiến, nice 👍 #happy", "target": "giao hàng", "label": LABELS["positive"], "implicit": True},
        {"text": "CSKH trả lời kiểu copy paste, nản thật sự...", "target": "hỗ trợ", "label": LABELS["negative"], "implicit": True},
        {"text": "Bàn phím gõ cũng được, không có gì wow.", "target": "bàn phím", "label": LABELS["neutral"], "implicit": False},
        {"text": "Bản cập nhật mới xóa mất tính năng mình dùng mỗi ngày 🙃", "target": "cập nhật", "label": LABELS["negative"], "implicit": True},
        {"text": "Loa nghe rõ kể cả bật nhỏ 🎧", "target": "loa", "label": LABELS["positive"], "implicit": False},
        {"text": "Nhận hàng mà seal rách toang, hơi rén 😬", "target": "đóng gói", "label": LABELS["negative"], "implicit": True},
        {"text": "Giá vậy là hợp lý đó chứ.", "target": "giá", "label": LABELS["positive"], "implicit": False},
        {"text": "Login được nhưng UI nhìn rối quá :/", "target": "ui", "label": LABELS["neutral"], "implicit": False},
        {"text": "Mở Maps phát máy nóng như lò 🤡", "target": "nhiệt", "label": LABELS["negative"], "implicit": True},
        {"text": "Shop rep tin nhắn nhanh ghê, 10 điểm.", "target": "người bán", "label": LABELS["positive"], "implicit": True},
        {"text": "Thông báo delay kiểu... giờ mới hiện, chịu luôn 😑", "target": "thông báo", "label": LABELS["negative"], "implicit": True},
        {"text": "Màn hình đủ sáng ngoài trời, ok nha 🌞", "target": "màn hình", "label": LABELS["positive"], "implicit": False},
        {"text": "Mic hút ồn nền quá trời, call mệt 😩", "target": "micro", "label": LABELS["negative"], "implicit": True},
        {"text": "Ship 1 tuần, cũng bình thường thôi.", "target": "ship", "label": LABELS["neutral"], "implicit": False},
        {"text": "Tai nghe cứ lên tàu là mất kết nối 🤦", "target": "bluetooth", "label": LABELS["negative"], "implicit": True},
        {"text": "Đồng hồ đo chạy bộ khá chuẩn 🏃", "target": "theo dõi", "label": LABELS["positive"], "implicit": False},
        {"text": "Cụm cam lồi để lên bàn cứ lắc lắc...", "target": "thiết kế", "label": LABELS["neutral"], "implicit": True},
        {"text": "CSKH xử lý gọn lẹ, khỏi drama 👌", "target": "chăm sóc khách hàng", "label": LABELS["positive"], "implicit": True},
        {"text": "Quạt laptop hú như máy bay ✈️", "target": "quạt", "label": LABELS["negative"], "implicit": True},
        {"text": "Cài đặt nhanh gọn, khỏi đọc hướng dẫn 😄", "target": "cài đặt", "label": LABELS["positive"], "implicit": True},
        {"text": "Cục sạc hơi nóng nhưng vẫn dùng được.", "target": "sạc", "label": LABELS["neutral"], "implicit": True},
        {"text": "Hoàn tiền về nhanh, bất ngờ luôn 💸", "target": "hoàn tiền", "label": LABELS["positive"], "implicit": True},
        {"text": "Giá gói thuê bao lại tăng nữa rồi...", "target": "thuê bao", "label": LABELS["negative"], "implicit": False},
        {"text": "Hiệu năng ổn cho lướt mạng thôi.", "target": "hiệu năng", "label": LABELS["neutral"], "implicit": False},
        {"text": "Mở hộp thiếu phụ kiện, bực mình 😤", "target": "phụ kiện", "label": LABELS["negative"], "implicit": True},
        {"text": "Ốp lưng vừa khít, đẹp xịn 😍", "target": "ốp lưng", "label": LABELS["positive"], "implicit": False},
        {"text": "App ngày nào cũng popup xin 5 sao, khó chịu thật 🤨", "target": "popup", "label": LABELS["negative"], "implicit": True},
        {"text": "Tính năng mới nhìn chung ổn, chưa có gì để khen/chê.", "target": "tính năng", "label": LABELS["neutral"], "implicit": False},
        {"text": "Giao diện dark mode nhìn đã mắt 😎", "target": "dark mode", "label": LABELS["positive"], "implicit": False},
        {"text": "Mạng Wi-Fi bắt yếu, đứng sát router mới ăn 🫠", "target": "wifi", "label": LABELS["negative"], "implicit": True},
        {"text": "Mua sale nên thấy đáng tiền #deal 🛒", "target": "giá", "label": LABELS["positive"], "implicit": True},

        # --- Photo comments ---
        {"text": "Ảnh chụp góc này xịn thiệt 😍", "target": "ảnh", "label": LABELS["positive"], "implicit": False},
        {"text": "Ánh sáng đẹp quá trời ơi ✨", "target": "ánh sáng", "label": LABELS["positive"], "implicit": True},
        {"text": "Filter hơi quá tay nha 😅", "target": "filter", "label": LABELS["neutral"], "implicit": True},
        {"text": "Caption dễ thương ghê 🥹", "target": "caption", "label": LABELS["positive"], "implicit": False},
        {"text": "Ủa sao ảnh mờ vậy, lấy nét đâu rồi 🤨", "target": "chất lượng", "label": LABELS["negative"], "implicit": True},
        {"text": "Màu da nhìn ảo ma canada 😬", "target": "màu sắc", "label": LABELS["negative"], "implicit": True},
        {"text": "Bố cục cũng ổn, nhưng nền hơi rối.", "target": "bố cục", "label": LABELS["neutral"], "implicit": False},

        # --- Video comments ---
        {"text": "Video cut nhịp cuốn phết 👏", "target": "edit", "label": LABELS["positive"], "implicit": True},
        {"text": "Âm thanh rõ, không bị rè 👍", "target": "âm thanh", "label": LABELS["positive"], "implicit": False},
        {"text": "Nội dung ok nhưng hơi dài, coi tới đoạn cuối hơi đuối 😴", "target": "nội dung", "label": LABELS["neutral"], "implicit": True},
        {"text": "Video giật lag như phim kinh dị 🙃", "target": "mượt mà", "label": LABELS["negative"], "implicit": True},
        {"text": "Thumbnail nhìn clickbait ghê á 😑", "target": "thumbnail", "label": LABELS["negative"], "implicit": True},
        {"text": "Voiceover nghe dễ chịu, kể chuyện hay 🫶", "target": "giọng", "label": LABELS["positive"], "implicit": False},

        # --- Sarcasm / irony (mostly implicit) ---
        {"text": "Wow, chất lượng đỉnh quá ha 🙃", "target": "chất lượng", "label": LABELS["negative"], "implicit": True},
        {"text": "Hay dữ ta, xem xong muốn xem lại liền... (không) 😐", "target": "video", "label": LABELS["negative"], "implicit": True},
        {"text": "Đẹp quá trời, nhìn mà 'muốn' khóc luôn 😭 (mỉa)", "target": "ảnh", "label": LABELS["negative"], "implicit": True},
        {"text": "Cười xỉu, nội dung tinh tế ghê cơ 🤡", "target": "nội dung", "label": LABELS["negative"], "implicit": True},
        {"text": "Ủa tưởng clip hài, ai dè hài thật... hài ở mình 🤦", "target": "kịch bản", "label": LABELS["negative"], "implicit": True},
        {"text": "Đỉnh của chóp, xem mà chill lắm 😌", "target": "video", "label": LABELS["positive"], "implicit": True},
    ]

    # Build a bigger train set by templating (so main.py can run too).
    templates = [
        # product/service
        ("Vừa thử {target} mới, thấy {adj} {emo}", "explicit"),
        ("Chưa biết nói sao về {target} nữa...", "neutral"),
        ("Sao {target} cứ bị vậy hoài trời ơi {emo}", "implicit_neg"),
        ("Tự nhiên {target} làm mình đỡ tốn thời gian ghê {emo}", "implicit_pos"),
        ("{target} dùng tạm ổn.", "neutral"),
        ("{target} ok nhưng vẫn có điểm lăn tăn.", "neutral"),
        ("{target} xịn nhaaa {emo} #recommend", "explicit_pos"),
        ("{target} tệ thiệt sự {emo}", "explicit_neg"),
        ("Ai giúp mình với, {target} lỗi suốt {emo}", "implicit_neg"),
        ("{target} hôm nay chạy mượt hẳn {emo}", "implicit_pos"),

        # photo/video
        ("Ảnh này {adj} á {emo}", "photo_pos"),
        ("Góc chụp {adj} nhưng màu {adj2}.", "photo_neu"),
        ("Filter kiểu này nhìn {adj} quá {emo}", "photo_neg"),
        ("Clip edit {adj} nha {emo}", "video_pos"),
        ("Nội dung {adj}, nhưng hơi {adj2} 😅", "video_neu"),
        ("Âm thanh {adj2} quá, nghe nhức đầu {emo}", "video_neg"),

        # sarcasm/irony
        ("Wow {target} {adj} quá ha 🙃", "sarcasm_neg"),
        ("Đỉnh của chóp luôn, {target} {adj} ghê 😐", "sarcasm_neg"),
        ("Hay dữ ta, coi xong muốn coi lại liền... (không) {emo}", "sarcasm_neg"),
    ]

    positives = ["xịn", "mượt", "ngon", "ổn áp", "đỉnh", "đáng tiền"]
    negatives = ["tệ", "lỗi", "lag", "chập chờn", "khó chịu", "bất ổn"]
    neutrals = ["bình thường", "tạm", "ổn", "không có gì đặc biệt"]
    emotes_pos = ["😄", "😍", "✨", "👍", "👌", "🔥"]
    emotes_neg = ["😩", "😤", "🙃", "🤦", "😑", "🫠"]
    emotes_neu = ["🤷", "😶", "🙂"]

    targets = [
        "pin",
        "camera",
        "giao hàng",
        "hỗ trợ",
        "cập nhật",
        "ui",
        "hiệu năng",
        "bluetooth",
        "màn hình",
        "micro",
        "đóng gói",
        "hoàn tiền",
        "ship",
        "thuê bao",
        "sạc",
        "wifi",
        "loa",
        "app",
        "tính năng",

        # photo/video targets
        "ảnh",
        "caption",
        "filter",
        "bố cục",
        "ánh sáng",
        "màu sắc",
        "video",
        "edit",
        "âm thanh",
        "nội dung",
        "thumbnail",
        "kịch bản",
    ]

    train_rows = []
    # Larger train set for richer fine-tuning experiments.
    for _ in range(3000):
        t, kind = random.choice(templates)
        target = random.choice(targets)

        # Default fillers so every template can be formatted safely.
        adj = random.choice(neutrals)
        adj2 = random.choice(neutrals)
        emo = random.choice(emotes_neu)

        if kind == "explicit":
            if random.random() < 0.5:
                adj = random.choice(positives)
                emo = random.choice(emotes_pos)
                label = LABELS["positive"]
            else:
                adj = random.choice(negatives)
                emo = random.choice(emotes_neg)
                label = LABELS["negative"]
            implicit = False
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "explicit_pos":
            adj = random.choice(positives)
            emo = random.choice(emotes_pos)
            label = LABELS["positive"]
            implicit = False
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "explicit_neg":
            adj = random.choice(negatives)
            emo = random.choice(emotes_neg)
            label = LABELS["negative"]
            implicit = False
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "neutral":
            emo = random.choice(emotes_neu)
            label = LABELS["neutral"]
            implicit = False
            # add light code-switch
            if random.random() < 0.2:
                text = f"{target} {random.choice(neutrals)} thôi {emo} (so-so)"
            else:
                adj = random.choice(neutrals)
                text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "implicit_neg":
            adj = random.choice(negatives)
            emo = random.choice(emotes_neg)
            label = LABELS["negative"]
            implicit = True
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "implicit_pos":
            adj = random.choice(positives)
            emo = random.choice(emotes_pos)
            label = LABELS["positive"]
            implicit = True
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "photo_pos":
            adj = random.choice(positives)
            emo = random.choice(emotes_pos)
            label = LABELS["positive"]
            implicit = random.random() < 0.4
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "photo_neu":
            label = LABELS["neutral"]
            implicit = False
            adj = random.choice(neutrals)
            adj2 = random.choice(neutrals)
            emo = random.choice(emotes_neu)
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "photo_neg":
            adj = random.choice(negatives)
            emo = random.choice(emotes_neg)
            label = LABELS["negative"]
            implicit = True
            text = t.format(target=target, adj=adj, adj2=random.choice(negatives), emo=emo)

        elif kind == "video_pos":
            adj = random.choice(positives)
            emo = random.choice(emotes_pos)
            label = LABELS["positive"]
            implicit = random.random() < 0.5
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "video_neu":
            adj = random.choice(neutrals)
            adj2 = random.choice(["dài", "chậm", "nhạt", "lẹ"])
            emo = random.choice(emotes_neu)
            label = LABELS["neutral"]
            implicit = True
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        elif kind == "video_neg":
            adj2 = random.choice(["rè", "chói", "to", "bé xíu", "lệch"])
            emo = random.choice(emotes_neg)
            label = LABELS["negative"]
            implicit = True
            text = t.format(target=target, adj=random.choice(negatives), adj2=adj2, emo=emo)

        elif kind == "sarcasm_neg":
            adj = random.choice(positives)  # sarcasm uses positive word but negative meaning
            emo = random.choice(["🙃", "😐", "🤡", "😑"])
            label = LABELS["negative"]
            implicit = True
            text = t.format(target=target, adj=adj, adj2=adj2, emo=emo)

        else:
            raise ValueError(f"Unknown template kind: {kind}")

        # sprinkle some hashtags / elongated words / emojis
        if random.random() < 0.15:
            text += " #review"
        if random.random() < 0.10:
            text += random.choice([" #ảnhđẹp", " #video", " #meme", " #tiktok", " #reels", " #storytime"])
        if random.random() < 0.10:
            text = text.replace("quá", "quáaaa") if "quá" in text else text
        train_rows.append({"text": text, "target": target, "label": label, "implicit": implicit})

    # Test set: larger and diverse (note: GPT eval on this will cost more API calls).
    test_rows = base.copy()
    # Ensure >= 600 test samples.
    for _ in range(650):
        target = random.choice(targets)
        label = random.choice([LABELS["positive"], LABELS["negative"], LABELS["neutral"]])
        implicit = random.random() < 0.55
        if label == LABELS["positive"]:
            if implicit:
                text = f"Tự nhiên thấy {target} hôm nay ổn hơn hẳn {random.choice(emotes_pos)}"
            else:
                text = f"{target} ngon nha {random.choice(emotes_pos)}"
        elif label == LABELS["negative"]:
            if implicit:
                text = f"{target} làm mình muốn khóc {random.choice(emotes_neg)}"
            else:
                text = f"{target} tệ quá {random.choice(emotes_neg)}"
        else:
            if implicit:
                text = f"{target} cũng... vậy thôi {random.choice(emotes_neu)}"
            else:
                text = f"{target} bình thường {random.choice(emotes_neu)}"

        # make some explicit sarcasm in test
        if random.random() < 0.18:
            text = f"Đỉnh quá ha, {target} {random.choice(positives)} ghê 🙃"  # sarcasm
            label = LABELS["negative"]
            implicit = True

        # extra variety: photo/video specific phrasing sometimes
        if random.random() < 0.22:
            if target in {"ảnh", "caption", "filter", "bố cục", "ánh sáng", "màu sắc"}:
                text = random.choice([
                    f"Ảnh này nhìn {random.choice(positives)} ghê {random.choice(emotes_pos)}",
                    f"Filter này {random.choice(neutrals)} thôi {random.choice(emotes_neu)}",
                    f"Ánh sáng {random.choice(positives)} mà màu hơi {random.choice(neutrals)}.",
                    f"Màu da bị {random.choice(negatives)} quá {random.choice(emotes_neg)}",
                ])
            elif target in {"video", "edit", "âm thanh", "nội dung", "thumbnail", "kịch bản"}:
                text = random.choice([
                    f"Clip edit {random.choice(positives)} nha {random.choice(emotes_pos)}",
                    f"Âm thanh {random.choice(['rè', 'chói', 'to', 'bé xíu'])} quá {random.choice(emotes_neg)}",
                    f"Nội dung {random.choice(neutrals)} nhưng hơi {random.choice(['dài', 'nhạt', 'chậm'])} 😅",
                    f"Thumbnail nhìn clickbait ghê 😑",
                ])

        if random.random() < 0.25:
            text += random.choice([" #trải_nghiệm", " #feedback", " #mua_hàng", " #hỏi_đáp"])
        if random.random() < 0.15:
            text = text.replace("mình", "tui")
        if random.random() < 0.10:
            text += " (no cap)"

        test_rows.append({"text": text, "target": target, "label": label, "implicit": implicit})

    write_pickle(train_path, train_rows)
    write_pickle(test_path, test_rows)

    print(f"Wrote train: {train_path} ({len(train_rows)} rows)")
    print(f"Wrote test:  {test_path} ({len(test_rows)} rows)")


if __name__ == "__main__":
    main()
