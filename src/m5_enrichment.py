from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL, ENRICH_WORKERS


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── LLM helper ──────────────────────────────────────────

_CLIENT = None


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
    """Gọi OpenAI chat completion (client dùng chung, temperature=0 cho kết quả ổn định)."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI()
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0,
        **kwargs,
    )
    return resp.choices[0].message.content.strip()


def _sentences(text: str) -> list[str]:
    """Tách câu, bỏ markdown header/quote markers để fallback extractive đọc tự nhiên hơn."""
    cleaned = re.sub(r"^[#>\-*\s]+", "", text, flags=re.MULTILINE).replace("**", "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if len(s.strip()) > 10]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            return _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                         "Giữ nguyên các con số, mức tiền, thời hạn.", text, max_tokens=150)
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    return _extractive_summary(text)


def _extractive_summary(text: str) -> str:
    """Fallback không cần API: 2 câu đầu."""
    sentences = _sentences(text)
    if not sentences:
        return text
    summary = " ".join(sentences[:2])
    return summary if summary.endswith((".", "!", "?")) else summary + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời, "
                "theo cách một nhân viên thực sự sẽ hỏi. Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
                text, max_tokens=200,
            )
            questions = [q.strip().lstrip("0123456789.-) ").strip() for q in content.split("\n")]
            questions = [q for q in questions if q]
            if questions:
                return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    return _extractive_questions(text, n_questions)


def _extractive_questions(text: str, n_questions: int = 3) -> list[str]:
    """Fallback không cần API: biến câu khẳng định thành câu hỏi."""
    return [f"{s.rstrip('.!?')}?" for s in _sentences(text)[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY:
        try:
            context = _chat(
                "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì "
                "(nêu tên tài liệu, phiên bản nếu có). Chỉ trả về 1 câu.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80,
            )
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Simple fallback: prefix bằng tên tài liệu
    prefix = f"Trích từ {document_title}." if document_title else ""
    return f"{prefix}\n\n{text}" if prefix else text


# ─── Technique 4: Auto Metadata Extraction ──────────────

_CATEGORY_KEYWORDS = {
    "it": ["mật khẩu", "vpn", "malware", "cntt", "mfa", "bảo mật", "dữ liệu", "helpdesk"],
    "finance": ["lương", "tạm ứng", "chi phí", "thanh toán", "mua sắm", "báo giá", "vnđ", "tài chính"],
    "hr": ["nghỉ", "phép", "thử việc", "bảo hiểm", "đào tạo", "mentor", "buddy", "thưởng", "đánh giá"],
}


def _heuristic_metadata(text: str) -> dict:
    """Fallback rule-based: category theo keyword, ngôn ngữ theo dấu tiếng Việt."""
    lower = text.lower()
    hits = {cat: sum(lower.count(k) for k in kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    category = max(hits, key=hits.get) if any(hits.values()) else "policy"
    language = "vi" if re.search(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", lower) else "en"
    header = re.search(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    return {"topic": header.group(1).strip() if header else "general", "entities": [],
            "category": category, "language": language}


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(
                'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], '
                '"category": "policy|hr|it|finance", "language": "vi|en"}',
                text, max_tokens=150, json_mode=True,
            )
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    return _heuristic_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────

_COMBINED_PROMPT = """Bạn là hệ thống làm giàu dữ liệu cho RAG. Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu, giữ nguyên con số/mức tiền/thời hạn",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu (tên tài liệu, phiên bản/ngày hiệu lực nếu có, mục) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Câu hỏi phải là những câu nhân viên thực sự sẽ hỏi và đoạn văn trả lời được. Chỉ trả về JSON."""


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            content = _chat(_COMBINED_PROMPT, f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                            max_tokens=500, json_mode=True)
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Fallback không cần API — vẫn prepend tên tài liệu để BM25/dense match được theo nguồn
    return {
        "summary": _extractive_summary(text),
        "questions": _extractive_questions(text),
        "context": f"Trích từ {source}." if source else "",
        "metadata": _heuristic_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    def _enrich_one(chunk: dict) -> EnrichedChunk:
        text = chunk["text"]
        chunk_meta = chunk.get("metadata", {})
        # Tên tài liệu cho contextual prepend: title (header H1) + file nguồn nếu có
        source = chunk_meta.get("source", "")
        title = chunk_meta.get("title", "")
        doc_name = f"{title} ({source})" if title and source else (title or source)

        if use_combined:
            result = _enrich_single_call(text, doc_name)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, doc_name) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        if not isinstance(auto_meta, dict):
            auto_meta = {}
        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary if isinstance(summary, str) else str(summary),
            hypothesis_questions=[str(q) for q in questions] if isinstance(questions, list) else [],
            # Metadata gốc (source, parent_id...) không bị LLM ghi đè
            auto_metadata={**chunk_meta, **{k: v for k, v in auto_meta.items() if k not in chunk_meta}},
            method="+".join(methods),
        )

    # Gọi API song song (I/O bound) — giữ nguyên thứ tự chunks
    workers = ENRICH_WORKERS if OPENAI_API_KEY and len(chunks) > 1 else 1
    enriched = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, item in enumerate(pool.map(_enrich_one, chunks)):
            enriched.append(item)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
