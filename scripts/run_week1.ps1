# Week-1 local pipeline (sample corpus, no pretrained weights).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -m pip install -e .
python scripts/check_env.py
python scripts/prepare_corpus.py --source sample --config configs/default.yaml
python scripts/train_tokenizer.py --config configs/default.yaml
python -m pytest tests -q
python scripts/train.py --config configs/smoke.yaml --run-name gpt_smoke
python scripts/generate.py --config configs/smoke.yaml --untrained --prompt "Ngày xửa ngày xưa,"
Write-Host "Week-1 smoke path finished. Real weights: train ngram/lstm/gpt_small later."
