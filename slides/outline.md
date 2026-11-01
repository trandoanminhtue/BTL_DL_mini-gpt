# Slide thuyết trình tuần 6 (10–12 slides)

1. **Bài toán** — Language modeling tiếng Việt; đầu vào prompt, đầu ra token tiếp theo / văn bản sinh.
2. **Câu hỏi nghiên cứu** — PPL so với n-gram & LSTM; scaling size–loss; sampling.
3. **Dữ liệu** — Wikipedia-VI (và sample folk/lục bát); split theo tài liệu; BPE 16k giữ dấu.
4. **Baseline** — Trigram nội suy; LSTM 2 lớp.
5. **Mini-GPT** — Decoder-only, 10–30M; sơ đồ khối (embed → N × (Attn+MLP) → LM head).
6. **Huấn luyện** — AdamW, cosine, context 256; checkpoint `best.pt` (nêu rõ trọng số gắn sau).
7. **Kết quả PPL** — Bảng số thật từ `outputs/`.
8. **Scaling** — Hình params vs val loss.
9. **Sinh văn bản** — Cùng prompt, greedy vs T / top-k / top-p.
10. **Lỗi** — Mất dấu, lặp câu, lệch thể thơ.
11. **Demo** — `python scripts/generate.py --ckpt ... --prompt "..."` 
12. **Kết luận & hạn chế** — Context ngắn, corpus sinh viên, không SOTA.
