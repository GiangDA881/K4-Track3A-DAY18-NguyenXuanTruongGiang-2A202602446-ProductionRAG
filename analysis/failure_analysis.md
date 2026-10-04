# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Xuân Trường Giang (2A202602446)  
**Khóa:** K4 - Track 3A  

---

> ⏳ **Trạng thái:** bảng RAGAS bên dưới cần điền sau khi chạy `python main.py` với `OPENAI_API_KEY`
> (số liệu lấy từ `reports/naive_baseline_report.json` và `reports/ragas_report.json`).
> Bottom-5 bên dưới được chọn từ **retrieval diagnostic** (kiểm tra top-3 context của từng câu
> có chứa tài liệu nguồn và các con số của ground truth hay không) — sau khi có RAGAS, đối chiếu
> với `failures` trong `reports/ragas_report.json` (đã sort theo điểm trung bình tăng dần).

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | ⏳ | ⏳ | ⏳ |
| Answer Relevancy | ⏳ | ⏳ | ⏳ |
| Context Precision | ⏳ | ⏳ | ⏳ |
| Context Recall | ⏳ | ⏳ | ⏳ |

**Production pipeline:** hierarchical chunking (parent 2048 / child 256) → M5 combined enrichment
(context prepend + HyQA, 1 call/chunk) → BM25 (underthesea) + bge-m3 dense → RRF (k=60) →
bge-reranker-v2-m3 trên top-20 → map child → parent, lấy top-3 parent → gpt-4o-mini (temperature 0).

**Retrieval diagnostic (20 câu):** 19/20 câu có tài liệu nguồn chính trong top-3 context. Các câu
multi-hop (cần ≥ 2 tài liệu) và câu có 2 phiên bản chính sách là nhóm rủi ro cao nhất.
*Lưu ý:* diagnostic này chạy trong môi trường không tải được model HuggingFace, nên dense dùng
`HashingEncoder` và rerank dùng `LexicalOverlapScorer` (fallback trong code); BM25 + underthesea là thật.
Thứ hạng "Got" bên dưới cần kiểm tra lại khi chạy với bge-m3 + bge-reranker-v2-m3.

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban (Director) phê duyệt (khoảng 5–50 triệu); cần xác nhận cấu hình kỹ thuật từ phòng CNTT; đính kèm ≥ 3 báo giá vì > 10 triệu.
- **Got:** Top-3 context = `hoan_chi_dao_tao.md`, `tam_ung.md`, `dao_tao_noi_bo.md` — **thiếu `mua_sam.md`** → answer dự kiến "Không tìm thấy" hoặc trả lời sai người phê duyệt.
- **Worst metric:** context_recall
- **Error Tree:** Output sai → Context đúng? **Không** (thiếu tài liệu mua sắm) → Query OK? **Không** — câu hỏi gộp 2 ý ("ai phê duyệt" + "cần gì từ CNTT"), từ "laptop", "nhân viên mới" kéo về tài liệu đào tạo/tạm ứng.
- **Root cause:** Retrieval — query dài, nhiều intent; từ khoá "mua sắm" không xuất hiện trong câu hỏi (vocabulary gap "mua laptop" ↔ "Quy trình mua sắm").
- **Suggested fix:** Query decomposition / multi-query (tách 2 sub-question rồi RRF), HyQA cho chunk `mua_sam.md` ("Mua laptop cần ai duyệt?"), reranker cross-encoder thật (bge-reranker-v2-m3) thay vì lexical.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép (v2024); lương Senior (P3–P4) 20–35 triệu/tháng.
- **Got:** Top-3 = `nghi_phep_khong_luong.md`, `nghi_phep_nam_v2023.md`, `nghi_phep_nam_v2024.md` — **thiếu `bang_luong_2024.md`**; có cả v2023 (cũ) lẫn v2024.
- **Worst metric:** context_recall (phần lương), có rủi ro faithfulness nếu LLM dùng công thức v2023 (5 năm/1 ngày → 13 ngày).
- **Error Tree:** Output sai một phần → Context đúng? **Một phần** (đủ phép năm, thiếu bảng lương) → Query OK? **Không** — multi-hop 2 tài liệu, top-3 bị 3 tài liệu "nghỉ phép" chiếm hết.
- **Root cause:** Retrieval + top-k quá nhỏ cho câu multi-hop; không lọc phiên bản cũ.
- **Suggested fix:** Multi-query theo từng intent; diversity khi chọn context (MMR / tối đa 1 parent mỗi chủ đề); metadata filter loại tài liệu có "ĐÃ THAY THẾ"/phiên bản cũ.

### #3
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày (hạn 15 ngày); phí 2%/tháng × 15.000.000 = 300.000 VNĐ/tháng, pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** Context đúng (`tam_ung.md` rank 1) nhưng chính sách không nói cách tính pro-rata → LLM phải tự suy luận.
- **Worst metric:** faithfulness (con số 50.000 không có trong context — là phép tính) / answer_relevancy nếu LLM trả "Không tìm thấy".
- **Error Tree:** Output sai → Context đúng? **Có** → Query OK? **Có** → lỗi ở bước **Generation** (tính toán).
- **Root cause:** Câu hỏi cần reasoning số học; prompt ban đầu "CHỈ dựa trên context" khiến LLM hoặc từ chối hoặc tính sai.
- **Suggested fix:** Prompt yêu cầu nêu các bước tính dựa trên số trong context (đã thêm vào `SYSTEM_PROMPT`); với production dùng tool calculator / chain-of-thought có kiểm tra.

### #4
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** v2.0 hiện hành: mỗi 120 ngày (v1.0 cũ: 90 ngày, đã bị thay thế).
- **Got:** Top-3 = `mat_khau_v1.md` (**cũ, rank 1**), `mat_khau_v2.md`, `so_tay_an_toan.pdf` → LLM dễ trả "90 ngày".
- **Worst metric:** context_precision (chunk cũ xếp trên chunk hiện hành) → kéo theo faithfulness/answer đúng-sai.
- **Error Tree:** Output sai → Context đúng? **Có nhưng nhiễu** (cả 2 phiên bản, bản cũ xếp trên) → Query OK? Có → lỗi ở bước **Ranking**.
- **Root cause:** Hai phiên bản gần như trùng từ vựng → BM25/dense không phân biệt; không có metadata về hiệu lực.
- **Suggested fix:** Metadata filter `status != superseded` hoặc boost theo `effective_date`; contextual prepend ghi rõ "phiên bản cũ, đã thay thế" (title có dòng version — đã thêm); prompt "dùng phiên bản mới nhất" (đã thêm).

### #5
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Thử việc không được nghỉ phép năm; cần nghỉ thì xin nghỉ không lương, trưởng phòng duyệt.
- **Got:** Top-3 = `thu_viec.md` (đúng), `nghi_phep_dac_biet.md`, `nghi_phep_nam_v2023.md` — 2/3 context không liên quan, và chứa "được nghỉ 12 ngày phép năm".
- **Worst metric:** context_precision
- **Error Tree:** Output có thể sai → Context đúng? **Có nhưng 2/3 nhiễu** → Query OK? Có → lỗi ở bước **Ranking** (+ rủi ro Generation trộn thông tin).
- **Root cause:** Câu hỏi phủ định; từ khoá "nghỉ phép năm" match mạnh với tài liệu phép năm dù câu trả lời nằm ở tài liệu thử việc.
- **Suggested fix:** Cross-encoder reranker + ngưỡng rerank score để cắt context nhiễu (không luôn lấy đủ 3); prompt nhấn mạnh trả lời Có/Không trước.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"

**Error Tree walkthrough:**
1. Output đúng? → **Không** — không nêu Director / xác nhận CNTT / 3 báo giá.
2. Context đúng? → **Không** — `mua_sam.md` không nằm trong top-3.
3. Query rewrite OK? → **Không** — pipeline chưa có query rewrite; câu hỏi gộp 2 intent và không chứa từ "mua sắm".
4. Fix ở bước: **Retrieval (query transformation)** — multi-query/decomposition + HyQA, sau đó mới đến reranking.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition (LLM tách sub-question) + RRF trên kết quả các sub-question cho câu multi-hop (#1, #2).
- Metadata filter theo phiên bản/hiệu lực (#4) và ngưỡng rerank score để bỏ context nhiễu (#5).
- So sánh RAGAS trước/sau từng thay đổi (ablation) thay vì thay đổi cùng lúc.
