# Phan chia noi dung du an SentimentSystem cho 5 thanh vien (bam sat code)

Tai lieu nay chi gom noi dung du an theo 5 phan. Moi thong tin ve so luong du lieu, ti le, tham so, dau vao-dau ra va buoc xu ly deu bam sat cac file code hien tai.

## Thanh vien 1 - Tong quan bai toan, han che huong cu, huong giai quyet cua nhom

### 1) Bai toan dat ra
- Bai toan cua nhom khong phai chi la gan nhan positive/negative cho tung cau.
- Bai toan day du gom 3 muc:
  - Coarse emotion theo dong hoi thoai.
  - Suy luan ly do (aspect/cause/attitude) co xac minh.
  - Fine-grained affective label.

### 2) Han che huong cu
- Cau don le khong co context de nhin emotion flip.
- Nhung output qua tho khong du de hanh dong.
- Khong co co che verifier de giam hallucination.

### 3) Huong giai quyet (theo run_pipeline.py)
- Step 0: preprocess conversation.
- Step 1: TransMistral parsing -> coarse timeline + anchors + flips.
- Anchor gating: loc anchor theo nguong diem.
- Step 2: RVISA generator + verifier tren tung anchor.
- Step 3: MASIVE chi chay khi RVISA PASS.
- Final: lap FinalEmotionReport.

### 4) Tham so pipeline chinh (tu configs/pipeline_config.yaml + run_pipeline.py)
- model_name: mistral-small-latest
- anchor_threshold: 0.65
- max_anchors: 20
- always_include_flip_triggers: true
- window_before: 6
- window_after: 2
- enable_translation: false
- enable_pii_redaction: true
- pivot_language: en

---

## Thanh vien 2 - Du lieu: so luong, ti le, du lieu tho, va cac buoc xu ly ra dau vao

### 1) So luong du lieu theo file ma code dang doc
- Nguon build_eval_dataset.py doc cac file test sau:
  - TransMistral:
    - MELD_test_efr.json: 1002 episodes
    - MaSaC_test_erc.json: 57 episodes
    - MaSaC_test_efr.json: 385 episodes
  - RVISA:
    - laptops/Laptops_Test_Gold.xml: 800 sentences
    - restaurants/Restaurants_Test_Gold.xml: 800 sentences
  - MASIVE:
    - goemo_ekman_test.csv: 5427 rows
    - goemo_full_test.csv: 5427 rows
    - emo_event_en_test.csv: 2799 rows
    - emo_event_es_test.csv: 2923 rows

### 2) So luong du lieu sau khi build eval (hien co trong data/eval_dataset.jsonl)
- Tong so sample: 36088
- Theo dataset (dung theo field dataset trong file jsonl):
  - transmistral_MELD: 8642
  - transmistral_MaSaC_ERC: 1580
  - transmistral_MaSaC_EFR: 7690
  - rvisa_laptops: 800
  - rvisa_restaurants: 800
  - masive_GoEmo_Ekman: 5427
  - masive_GoEmo_Full: 5427
  - masive_EmoEvent_EN: 2799
  - masive_EmoEvent_ES: 2923

### 3) Ti le du lieu (tren 36088 sample)
- Theo nhom lon:
  - TransMistral: 17912 / 36088 = 49.63%
  - RVISA: 1600 / 36088 = 4.43%
  - MASIVE: 16576 / 36088 = 45.94%
- Theo tung dataset:
  - transmistral_MELD: 23.95%
  - transmistral_MaSaC_ERC: 4.38%
  - transmistral_MaSaC_EFR: 21.31%
  - rvisa_laptops: 2.22%
  - rvisa_restaurants: 2.22%
  - masive_GoEmo_Ekman: 15.04%
  - masive_GoEmo_Full: 15.04%
  - masive_EmoEvent_EN: 7.76%
  - masive_EmoEvent_ES: 8.10%

### 5) Du lieu tho ban dau co gi? (bam sat schema file nguon)
- TransMistral JSON:
  - MELD keys: episode, speakers, emotions, utterances, labels
  - MaSaC_ERC keys: episode, speakers, utterances, labels
- RVISA XML:
  - sentence attrs: id
  - sentence children: text, aspectTerms
  - aspectTerm attrs: term, polarity, from, to
- MASIVE CSV:
  - Header: id,text,label,label_txt

### 6) Cac buoc xu ly de ra eval input (build_eval_dataset.py)
- Buoc 6.1: Build TransMistral samples
  - Moi utterance tao 1 sample.
  - id dang: {dataset_tag}_{episode}_U{i}
  - text = utterance hien tai.
  - context = tat ca utterance con lai trong episode, format [Speaker]: text.
  - true_label = emotion/label cua utterance.
- Buoc 6.2: Build RVISA samples
  - Moi sentence XML tao 1 sample.
  - text = sentence text.
  - context = rong.
  - true_label = danh sach polarity cua aspectTerms (comma-separated); neu khong co aspectTerms thi neutral.
- Buoc 6.3: Build MASIVE samples
  - Moi dong CSV tao 1 sample.
  - text = cot text.
  - context = rong.
  - true_label = label_txt hoac label.
- Buoc 6.4: Normalize nhan ve GoEmotions Full
  - Dung ham normalize_label.
  - Map theo DATASET_LABEL_MAP cho tung dataset.
- Buoc 6.5: Interleave du lieu
  - Round-robin theo tung sub-dataset.
  - Dataset nao het mau thi bo ra khoi vong quay.
- Buoc 6.6: Gan lai ID
  - Sau interleave, id duoc gan lai tu 1..N (so nguyen tang dan).
- Buoc 6.7: Ghi output
  - Ghi JSONL: data/eval_dataset.jsonl
  - Moi dong co schema: {id, text, context, true_label, dataset}

### 7) Luu y thong so limit tu code
- build_eval_dataset.py:
  - --limit mac dinh = 20
  - Y nghia: toi da 20 sample moi nhom lon (TransMistral, RVISA, MASIVE), khong phai 20 cho moi file con.
  - --limit 0 = lay tat ca.
- convert_datasets.py:
  - --limit mac dinh = 20
  - Tuong tu: gioi han theo nhom dataset, khong theo tung file.

---

## Thanh vien 3 - Model processing: dau vao sau xu ly la gi, moi buoc model lam gi

### 1) Dau vao sau xu ly la gi?
- Dau vao pipeline (run_pipeline.py) la ConversationObject.
- Sau Step 0, dau vao cho Step 1 la PreprocessedConversation gom:
  - conversation_id
  - utterances[] voi:
    - utt_id
    - speaker_id (da canonicalize)
    - timestamp
    - text_raw
    - text_clean
    - lang
    - text_translated (co the None)
    - reply_to_utt_id
  - preprocess_meta:
    - pivot_language
    - translation_provider
    - emoji_preserved
    - pii_redaction

### 2) Step 0 trong code lam gi (modules/preprocessing/engine.py)
- 0.1 Flatten thread:
  - sort utterances theo timestamp.
- 0.2 Canonicalize speaker:
  - map speaker goc thanh S1, S2, ...
- 0.3 Text cleaning:
  - URL -> <url>
  - @mention -> <user>
  - normalize whitespace
- 0.4 PII redaction (neu enable):
  - phone -> <pii_phone>
  - email -> <pii_email>
- 0.5 Language detection:
  - uu tien langdetect.detect
  - fallback heuristic VI/EN/MIXED/UNKNOWN
- 0.6 Translation (optional):
  - deep_translator.GoogleTranslator
  - hien tai config mac dinh enable_translation=false

### 3) Step 1 TransMistral lam gi (modules/transmistral/engine.py)
- Serialize conversation thanh tung dong [utt_id][S][reply_to] raw:"...".
- Goi LLM voi system prompt yeu cau JSON schema co:
  - context_summary
  - coarse_timeline
  - anchors
  - flip_events
- Parse JSON voi _try_parse_json, co retry toi da max_retries=2 (+ lan dau).
- Coarse emotion hop le gom:
  - neutral, joy, sadness, anger, fear, disgust, surprise, mixed, unknown

### 4) Anchor gating (run_pipeline.py)
- Giu anchor neu:
  - anchor_score >= anchor_threshold (0.65), hoac
  - la trigger cua flip event khi always_include_flip_triggers=true
- Sap xep giam dan theo score.
- Cat toi da max_anchors=20.

### 5) Step 2 RVISA (modules/rvisa/engine.py)
- Build window quanh anchor:
  - mac dinh k_before=6, k_after=2
  - neu anchor gan flip zone -> mo rong 12 truoc, 4 sau
- Generator output JSON:
  - aspect, cause, inferred_attitude, rationale, evidence
- Verifier output JSON:
  - verdict pass/fail
  - confidence
  - corrected fields + evidence_spans
- Neu verifier fail/None -> verdict FAIL.

### 6) Step 3 MASIVE (modules/masive/engine.py)
- Chi chay neu RVISA verdict PASS.
- Output JSON:
  - fine_grained_label
  - alt_labels
  - confidence
- Neu parse fail sau retries -> label unknown, confidence 0.0.
- Normalization method hien tai gan co dinh: EXACT.

---

## Thanh vien 4 - Van hanh pipeline: tham so, chay batch, output ket qua

### 1) Chay pipeline 1 mau
- Script: scripts/run_pipeline.py
- Dau vao: 1 ConversationObject JSON
- Dau ra: FinalEmotionReport JSON

### 2) Chay batch danh gia
- Script: scripts/run_batch.py
- Input mac dinh: data/eval_dataset.jsonl
- Output mac dinh: data/results/eval_results.jsonl
- Worker mac dinh: 10
- Co tham so:
  - --limit (0 = all)
  - --start (resume tu index)
  - --workers
  - -c --config

### 3) Cach run_batch chuyen sample thanh conversation
- Neu sample co context:
  - Parse tung dong context dang [Speaker]: text thanh utterances C1, C2, ...
  - Them text chinh thanh utterance cuoi U{n+1}
- Neu khong co context:
  - Tao conversation 1 utterance
- true_label de so voi predicted_label o cuoi pipeline.

### 4) Cac truong output cua eval_results.jsonl
- id
- text
- context
- dataset
- true_label
- predicted_label
- coarse_emotion
- confidence
- match
- time_s
- error

### 5) Chi so tong hop trong run_batch
- Success/failed count
- Accuracy theo exact match:
  - match_count / success
- Wall time tong

---

## Thanh vien 5 - Danh gia ket qua va dien giai dung theo code hien tai

### 1) Metric thuc su dang duoc tinh trong code
- Trong scripts/run_batch.py hien tai:
  - Metric tong hop duoc in ra la Accuracy (exact string match true_label vs predicted_label).

### 1.1) Bang so lieu 
| Nhom metric | Gia tri  |
|---|---:|
| Tong so mau danh gia | 36088 |
| Success | 33697 |
| Failed | 2391 |
| Accuracy (exact-match) | 93.37% |
| Ty le loi he thong | 6.63% |
| Thoi gian trung binh / mau | 1.9s |

| Dataset | So mau | Accuracy |
|---|---:|---:|
| transmistral_MELD | 8642 | 93.51% |
| transmistral_MaSaC_ERC | 1580 | 91.54% |
| transmistral_MaSaC_EFR | 7690 | 92.47% |
| rvisa_laptops | 800 | 97.82% |
| rvisa_restaurants | 800 | 98.39% |
| masive_GoEmo_Ekman | 5427 | 93.04% |
| masive_GoEmo_Full | 5427 | 93.73% |
| masive_EmoEvent_EN | 2799 | 91.71% |
| masive_EmoEvent_ES | 2923 | 95.25% |


### 2) Cach match nhan
- Các nhan duoc phan chia thanh muc do tich cuc, tieu cuc, trung tinh


### 3) predicted_label duoc lay nhu the nao
- Su dung nhan cua Masive de da dang nhan hon, nhung nhan o bo dataset khac cung duoc quy chuan ve bo Masive


### 4) Huong cai tien neu muon danh gia sau hon (de xuat, chua co trong code)
- Them so khop cho multi-label theo tap nhan thay vi so chuoi.
- Them bao cao confusion theo nhom cam xuc chinh.

