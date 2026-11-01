# Mini-GPT tiếng Việt (Đề tài 9)

Huấn luyện mô hình ngôn ngữ kiểu GPT cỡ nhỏ **từ đầu** trên văn bản tiếng Việt, so với **n-gram** và **LSTM**, rồi đo perplexity, chất lượng sinh, và đường cong scaling.

Trọng số (`.pt` / n-gram `.json`) **chưa có trong repo**: cài mã và cấu trúc trước, huấn luyện rồi bỏ checkpoint vào `checkpoints/` sau.

## Cấu trúc

```
configs/               # YAML: default, ngram, lstm, gpt_{tiny,small,medium}, instruct, smoke
data/sample/           # Corpus mẫu + instruct.jsonl + prompts (luôn có trong git)
data/processed/        # train/valid/test.txt — tạo bằng script
data/tokenizer/        # tokenizer.json — tạo bằng script
src/minigpt/
  models/              # gpt.py, lstm.py, ngram.py
  data/                # corpus, BPE, dataset packed + SFT
  train/               # AdamW trainer, checkpoint
  generate/            # temperature, top-k, top-p
  eval/                # perplexity
scripts/               # CLI: prepare, tokenizer, train, eval, generate, plots
tests/                 # Kiểm tra forward / packing / YAML (không cần GPU)
papers/                # proposal, milestone, final (template ACL)
slides/outline.md
checkpoints/           # best.pt sẽ đặt vào đây sau khi train
```

Kích thước GPT (vocab 16k, context 256, tied embeddings):

| Config | Lớp | d_model | heads | Tham số (xấp xỉ) |
|--------|-----|---------|-------|------------------|
| `gpt_tiny` | 4 | 256 | 4 | ~7M |
| `gpt_small` | 8 | 384 | 6 | ~20M (mô hình chính) |
| `gpt_medium` | 6 | 512 | 8 | ~27M |

## Cài đặt

Python 3.10+. GPU không bắt buộc cho đường ống mẫu.

```powershell
cd d:\DL\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python scripts/check_env.py
```

## Chạy nhanh tuần 1 (corpus mẫu)

```powershell
python scripts/prepare_corpus.py --source sample
python scripts/train_tokenizer.py
python -m pytest tests -q
python scripts/train.py --config configs/smoke.yaml --run-name gpt_smoke
python scripts/generate.py --config configs/smoke.yaml --untrained --prompt "Ngày xửa ngày xưa,"
```

Hoặc: `powershell -File scripts/run_week1.ps1`

`smoke` chỉ kiểm tra đường ống (1 epoch, mô hình rất nhỏ). Không dùng số này trong báo cáo.

## Dữ liệu Wikipedia (corpus chính)

```powershell
python scripts/prepare_corpus.py --source wikipedia --config configs/default.yaml
python scripts/train_tokenizer.py --config configs/default.yaml
```

Cần mạng và gói `datasets`. Giới hạn bài: `data.max_articles` trong `configs/default.yaml`. File local:

```powershell
python scripts/prepare_corpus.py --source local --local-path path\to\corpus.txt
```

## Huấn luyện (khi đã có tokenizer + split)

```powershell
python scripts/train.py --config configs/ngram.yaml
python scripts/train.py --config configs/lstm.yaml
python scripts/train.py --config configs/gpt_tiny.yaml
python scripts/train.py --config configs/gpt_small.yaml
python scripts/train.py --config configs/gpt_medium.yaml
```

Checkpoint mặc định:

- `checkpoints/ngram/best.json`
- `checkpoints/lstm/best.pt`
- `checkpoints/gpt_tiny/best.pt`
- `checkpoints/gpt_small/best.pt`
- `checkpoints/gpt_medium/best.pt`

Instruction-tune (sau khi có `gpt_small`):

```powershell
python scripts/train.py --config configs/instruct.yaml --instruct --init-ckpt checkpoints/gpt_small/best.pt
```

## Đánh giá và sinh văn bản

```powershell
python scripts/evaluate.py --config configs/ngram.yaml --split test
python scripts/evaluate.py --config configs/lstm.yaml --split test
python scripts/evaluate.py --config configs/gpt_small.yaml --ckpt checkpoints/gpt_small/best.pt --split test

python scripts/generate.py --config configs/gpt_small.yaml --ckpt checkpoints/gpt_small/best.pt --prompt "Ngày xửa ngày xưa," --temperature 0.8 --top-p 0.9 --top-k 50

python scripts/compare_sampling.py --ckpt checkpoints/gpt_small/best.pt
python scripts/plot_scaling.py
```

## Phân công gợi ý

| Người | Việc |
|-------|------|
| (1) | Corpus, làm sạch, BPE, thống kê split |
| (2) | Mini-GPT, trainer, checkpoint, generate |
| (3) | n-gram, LSTM, PPL, scaling, sampling, bảng ACL |

## Lịch 6 tuần và file nộp ACL

Toàn bộ báo cáo dùng template ACL (`papers/`, cần copy `acl.sty` — xem `papers/README.md`).

| Tuần | Việc | Nộp |
|------|------|-----|
| 1 | Đọc GPT / Transformer / scaling; chạy `check_env` + sample | `papers/proposal.tex` (≤ 2 trang) |
| 2 | Train n-gram + LSTM | — |
| 3 | Mini-GPT small chạy được | `papers/milestone.tex` (≤ 4 trang) |
| 4 | Siêu tham số, ablation | — |
| 5 | Scaling, sampling, lỗi, hình | bản nháp `final.tex` |
| 6 | Báo cáo, slide, demo | `final.tex`, mã nguồn, `slides/` |

## Kiến thức gắn với mã

- Language modeling + perplexity: `src/minigpt/eval/perplexity.py`
- Transformer decoder: `src/minigpt/models/gpt.py`
- Tokenization: `src/minigpt/data/tokenizer.py`
- Sampling: `src/minigpt/generate/sampler.py`

Không điền số PPL giả vào báo cáo. Ô trống trong TeX chỉ được thay khi file `outputs/*.json` tồn tại.
