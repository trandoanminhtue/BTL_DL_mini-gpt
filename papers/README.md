# Báo cáo ACL (3 mốc bắt buộc)

Tải [ACL style files](https://github.com/acl-org/acl-style-files) và copy `acl.sty`, `acl_natbib.bst` vào thư mục này trước khi biên dịch.

```bash
# Từ thư mục papers/ sau khi đã có acl.sty
pdflatex proposal.tex && bibtex proposal && pdflatex proposal.tex && pdflatex proposal.tex
pdflatex milestone.tex && bibtex milestone && pdflatex milestone.tex && pdflatex milestone.tex
pdflatex final.tex && bibtex final && pdflatex final.tex && pdflatex final.tex
```

| Mốc | File | Giới hạn |
|-----|------|----------|
| Cuối tuần 1 | `proposal.tex` | tối đa 2 trang |
| Cuối tuần 3 | `milestone.tex` | tối đa 2–4 trang |
| Tuần 6 | `final.tex` | theo yêu cầu lớp |

Điền tên nhóm, số liệu perplexity và hình từ `outputs/figures/` khi đã huấn luyện xong. Không bịa số.
