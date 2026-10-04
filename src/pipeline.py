from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json, os, re, sys, time
from collections import defaultdict
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY, LLM_MODEL

# Latency breakdown (bonus): thời gian từng bước build + per-query
LATENCY: dict = {"build": {}, "query": defaultdict(list)}

SYSTEM_PROMPT = """Bạn là trợ lý chính sách nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context được cung cấp.
Quy tắc:
- Trả lời trực tiếp, ngắn gọn bằng tiếng Việt, nêu đúng con số/mức tiền/thời hạn/người phê duyệt có trong context.
- Nếu context có nhiều phiên bản chính sách, dùng phiên bản MỚI NHẤT (hiện hành) và có thể nhắc phiên bản cũ đã bị thay thế.
- Với câu hỏi cần tính toán, nêu các bước tính dựa trên số liệu trong context.
- Không thêm thông tin ngoài context. Nếu context không có thông tin → trả lời "Không tìm thấy." """


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_store = {}  # parent_id → parent text: retrieve child (precision) → return parent (context)
    for doc in docs:
        title = _document_title(doc["text"])
        meta = {**doc["metadata"], "title": title} if title else doc["metadata"]
        parents, children = chunk_hierarchical(doc["text"], metadata=meta)
        for parent in parents:
            parent_store[parent.metadata["parent_id"]] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    LATENCY["build"]["chunking_s"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} child chunks / {len(parent_store)} parents from {len(docs)} documents "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        # Index = context prepend + chunk + HyQA questions (bridge vocabulary gap query ↔ chunk)
        all_chunks = [
            {"text": "\n".join([e.enriched_text, *e.hypothesis_questions]), "metadata": e.auto_metadata}
            for e in enriched
        ]
        LATENCY["build"]["enrichment_s"] = round(time.time() - t0, 2)
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_store = parent_store
    LATENCY["build"]["indexing_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()  # load trước để latency query không tính thời gian load model
    LATENCY["build"]["reranker_load_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def _document_title(text: str) -> str:
    """Header H1 + dòng version (nếu có) — dùng làm tên tài liệu cho enrichment."""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if not match:
        return ""
    title = match.group(1).strip()
    version = re.search(r"^>\s*(Phiên bản:.+)$", text, flags=re.MULTILINE)
    return f"{title} — {version.group(1).strip()}" if version else title


def _expand_to_parents(reranked, parent_store: dict, top_k: int) -> list[str]:
    """Child → parent: thay mỗi child bằng parent của nó, bỏ trùng, giữ thứ tự rerank."""
    contexts, seen = [], set()
    for r in reranked:
        pid = r.metadata.get("parent_id")
        key = pid or r.text
        if key in seen:
            continue
        seen.add(key)
        contexts.append(parent_store.get(pid, r.text))
        if len(contexts) >= top_k:
            break
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    t0 = time.perf_counter()
    results = search.search(query)
    t1 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank toàn bộ candidates rồi mới gom theo parent (nhiều child có thể cùng 1 parent)
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    t2 = time.perf_counter()
    parent_store = getattr(search, "parent_store", {})
    if reranked:
        contexts = _expand_to_parents(reranked, parent_store, RERANK_TOP_K)
    else:
        contexts = [r.text for r in results[:RERANK_TOP_K]]

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n---\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    t3 = time.perf_counter()

    LATENCY["query"]["hybrid_search_ms"].append((t1 - t0) * 1000)
    LATENCY["query"]["rerank_ms"].append((t2 - t1) * 1000)
    LATENCY["query"]["generation_ms"].append((t3 - t2) * 1000)
    LATENCY["query"]["total_ms"].append((t3 - t0) * 1000)
    return answer, contexts


def save_latency_report(ragas_seconds: float, path: str = "reports/latency_report.json") -> dict:
    """In + lưu bảng latency từng bước (build pipeline, per-query trung bình, RAGAS)."""
    per_query = {
        step: {"avg_ms": round(sum(v) / len(v), 1), "p95_ms": round(sorted(v)[int(0.95 * (len(v) - 1))], 1),
               "max_ms": round(max(v), 1)}
        for step, v in LATENCY["query"].items() if v
    }
    report = {"build_s": LATENCY["build"], "per_query": per_query, "ragas_eval_s": round(ragas_seconds, 2),
              "num_queries": len(LATENCY["query"]["total_ms"])}

    print("\n" + "=" * 60)
    print("LATENCY BREAKDOWN")
    print("=" * 60)
    for step, sec in LATENCY["build"].items():
        print(f"  [build] {step:<20} {sec:>8.2f}s")
    print(f"  {'[query] step':<28} {'avg':>8} {'p95':>8} {'max':>8}  (ms)")
    for step, st in per_query.items():
        print(f"  {step:<28} {st['avg_ms']:>8.1f} {st['p95_ms']:>8.1f} {st['max_ms']:>8.1f}")
    print(f"  [eval] ragas                {ragas_seconds:>8.2f}s")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Latency report saved to {path}")
    return report


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    ragas_seconds = time.time() - t0
    print(f"  ✓ RAGAS done ({ragas_seconds:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    save_latency_report(ragas_seconds)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
