# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Xuân Trường Giang (2A202602446)  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trên toàn corpus (26 tài liệu), basic tạo **51** chunk (avg 410 ký tự) còn semantic tạo **208** chunk (avg 99, min 6). Threshold 0.85 quá cao cho văn bản chính sách ngắn: header và từng câu bị tách riêng → chunk quá vụn, mất ngữ cảnh. Semantic hợp với văn bản dài, nhiều chủ đề hơn là FAQ/policy ngắn. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` + `_expand_to_parents()` | 99 child (avg 210, max 256) / 26 parent. Retrieve trên child (chính xác) rồi trả về parent (đủ ngữ cảnh) — mỗi tài liệu ~1.000 ký tự nên parent ≈ cả tài liệu, LLM thấy cả dòng phiên bản/ngày hiệu lực. Phải prefix `parent_id` bằng tên file để không trùng id giữa các tài liệu. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunk theo section markdown, giữ header + metadata `section`. Header liền nhau không có nội dung (`# Doc` → `## Mục`) được gộp làm breadcrumb thay vì tạo chunk rỗng. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | underthesea nối từ ghép bằng `_` ("nghỉ_phép") → phải `replace("_", " ")` và lowercase, nếu không query "nghỉ phép" không match. RRF chỉ dùng rank (1/(60+rank+1)) nên không cần chuẩn hoá score BM25 (0–20) với cosine (0–1); BM25 bắt tốt số liệu/từ khoá chính xác ("MFA", "PVI", "120 ngày"), dense bắt paraphrase. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank toàn bộ top-20 hybrid rồi mới gom child → parent, lấy top-3 parent. Model được cache theo tên và load trước trong `build_pipeline()` để latency query không tính thời gian load. Latency đo trong `reports/latency_report.json` (bước `rerank_ms`). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Faithfulness/answer relevancy đo bước Generation; context precision/recall đo Retrieval/Ranking. `failure_analysis()` map metric thấp nhất → bước hỏng trong Error Tree (Generation / Ranking / Retrieval) → gợi ý fix. Dự đoán metric yếu nhất là context_recall cho câu multi-hop (xem `analysis/failure_analysis.md`). |
| Contextual embeddings | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode: 1 call/chunk trả JSON gồm summary + 3 câu hỏi HyQA + câu context + metadata (`response_format=json_object`, chạy song song 8 luồng). Title tài liệu kèm dòng "Phiên bản … / ĐÃ THAY THẾ" được đưa vào context line → chunk của chính sách cũ được đánh dấu rõ khi embed. Câu hỏi HyQA được nối vào text để index (bridge vocabulary gap "mua laptop" ↔ "Quy trình mua sắm"). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  1. Không tải được model từ HuggingFace trong môi trường sandbox:
     `ProxyError` khi load `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, `all-MiniLM-L6-v2`
     (proxy trả `CONNECT tunnel failed, response 403` cho `huggingface.co`).
  2. Không gọi được OpenAI (`api.openai.com` bị chặn, chưa có `OPENAI_API_KEY`) → RAGAS không chạy.
  3. Hai PDF bị bỏ qua: `⚠️ Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).`
- **Nguyên nhân gốc rễ & Cách debug:**
  1. Kiểm tra trực tiếp `curl https://huggingface.co/api/models/BAAI/bge-m3` → 403 từ proxy, xác nhận lỗi mạng chứ không phải lỗi code.
     Giải pháp: thêm fallback offline (`src/offline_encoder.py`: hashing vectorizer; `LexicalOverlapScorer` cho reranker) có in cảnh báo rõ ràng, để test và pipeline vẫn chạy end-to-end; khi có mạng thì tự dùng model thật.
  2. `evaluate_ragas()` kiểm tra `OPENAI_API_KEY` trước và trả về 0 ngay — tránh RAGAS retry nhiều lần (mặc định `max_retries=10`) làm test bị treo. Khi có key thì đặt `RunConfig(max_retries=3)` và truyền rõ `ChatOpenAI(gpt-4o-mini)` + `text-embedding-3-small` (mặc định của ragas 0.1 là `text-embedding-ada-002`).
  3. `qdrant_client.recreate_collection()` đã deprecated → đổi sang `collection_exists()` + `delete_collection()` + `create_collection()`; search dùng `query_points()`.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cách RAGAS 0.1 tính điểm và xử lý NaN (judge parse lỗi) → đọc source `ragas/evaluation.py`; aggregate dùng `mean(skipna=True)`, per-question coi NaN là 0 để failure analysis vẫn sort được.
  - Xử lý tài liệu nhiều phiên bản → cần metadata hiệu lực + prompt "dùng phiên bản mới nhất".

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách nội bộ (HR / IT / Tài chính)

#### 1. Hiện trạng
- **Pipeline hiện tại:** paragraph chunking + dense-only search (giống `naive_baseline.py`), top-3 đưa thẳng vào LLM.
- **Vấn đề / Bottlenecks đang gặp:** trả lời theo chính sách cũ khi có nhiều phiên bản; câu hỏi multi-hop (phép năm + lương) thiếu context; câu phủ định ("thử việc có được…?") bị context nhiễu; chưa có bộ đo chất lượng.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (child 256 để retrieve, parent để trả về) — tài liệu policy ngắn, cần cả dòng phiên bản/hiệu lực; structure-aware cho tài liệu dài có nhiều mục.
2. **Search retrieval:** Hybrid BM25 (underthesea) + bge-m3, fuse bằng RRF — BM25 bắt mã/số liệu chính xác, dense bắt paraphrase; thêm metadata filter `status=active`.
3. **Reranking:** Có — bge-reranker-v2-m3 trên top-20, kèm ngưỡng score để bỏ context nhiễu; nếu latency quá cao thì dùng flashrank.
4. **Evaluation:** RAGAS 4 metrics trên test set 20 câu (mở rộng lên ~50 câu gồm multi-hop, phủ định, version conflict) + chạy lại mỗi khi đổi chunking/prompt (ablation).
5. **Enrichment:** Contextual prepend + HyQA bằng combined mode (1 call/chunk), metadata tự động (category, phiên bản) để phục vụ filter.

#### 3. Timeline triển khai
- **Tuần 1:** Chạy baseline vs production với RAGAS, chốt bộ test set mở rộng; thêm metadata phiên bản/hiệu lực cho toàn bộ tài liệu.
- **Tuần 2:** Query decomposition cho câu multi-hop, ngưỡng rerank score; đo lại RAGAS + latency.
- **Tuần 3:** OCR cho PDF scan (BCTC, Nghị định 13/2023), đưa vào index; đóng gói API + monitoring latency từng bước.
