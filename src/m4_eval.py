from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, LLM_MODEL, JUDGE_EMBEDDING_MODEL

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {m: 0.0 for m in METRICS}
    zeros["per_question"] = []

    # RAGAS dùng LLM-as-judge → cần OPENAI_API_KEY. Không có key thì trả về 0 ngay,
    # tránh RAGAS retry nhiều lần rồi mới fail.
    if not OPENAI_API_KEY:
        print("  ⚠️  RAGAS evaluation skipped: chưa set OPENAI_API_KEY trong .env")
        return zeros

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model=LLM_MODEL, temperature=0),
            embeddings=OpenAIEmbeddings(model=JUDGE_EMBEDDING_MODEL),
            run_config=RunConfig(timeout=120, max_retries=3, max_workers=8),
        )
        df = result.to_pandas()

        def _score(row, metric: str) -> float:
            # RAGAS trả NaN khi judge parse lỗi → tính như 0 cho per-question
            value = row.get(metric, 0.0)
            return 0.0 if value is None or math.isnan(float(value)) else float(value)

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
                **{m: _score(row, m) for m in METRICS},
            )
            for _, row in df.iterrows()
        ]
        # Aggregate bỏ qua NaN (giống cách RAGAS tự tính)
        aggregate = {m: round(float(df[m].mean(skipna=True)), 4) if m in df else 0.0 for m in METRICS}
        aggregate = {m: 0.0 if math.isnan(v) else v for m, v in aggregate.items()}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Diagnostic Tree: metric thấp nhất → bước hỏng trong pipeline → cách sửa
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer chứa thông tin không có trong context",
                         "Tighten prompt (chỉ dùng context), temperature=0, yêu cầu trích dẫn"),
        "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin cần cho ground truth",
                           "Improve chunking (parent-child), thêm BM25/hybrid, tăng top_k, enrichment"),
        "context_precision": ("Too many irrelevant chunks — chunk liên quan bị xếp hạng thấp",
                              "Add reranking (cross-encoder), metadata filter (version/phòng ban)"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lan man hoặc từ chối sai",
                             "Improve prompt template: trả lời trực tiếp, đúng trọng tâm câu hỏi"),
    }
    # Error Tree theo thứ tự kiểm tra: Output → Context → Query
    error_tree_step = {
        "faithfulness": "Output sai → Context đúng → lỗi ở bước Generation",
        "answer_relevancy": "Output lệch câu hỏi → Context đúng → lỗi ở bước Generation/Prompt",
        "context_precision": "Output sai → Context có nhiễu → lỗi ở bước Ranking",
        "context_recall": "Output sai → Context thiếu → lỗi ở bước Retrieval/Chunking",
    }

    analyzed = []
    for r in eval_results:
        scores = {m: float(getattr(r, m)) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "scores": {m: round(s, 4) for m, s in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "error_tree": error_tree_step[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "failures": failures,
        # Lưu chi tiết từng câu để phân tích failure (answer + contexts thực tế)
        "per_question": [asdict(r) if isinstance(r, EvalResult) else r for r in per_question],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
