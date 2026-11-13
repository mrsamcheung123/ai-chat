"""Run a curated question set locally and export results for manual answer grading."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.answer import Answerer  # noqa: E402
from rag.config import Settings  # noqa: E402
from rag.index import Corpus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results.json"))
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())
    settings = Settings.from_env()
    corpus, answerer = Corpus(settings), Answerer(settings)
    index, issue = corpus.snapshot()
    if issue or index is None:
        raise SystemExit(issue or "Index your PDFs first.")
    results = []
    for item in questions:
        started = perf_counter()
        hits = corpus.retrieve(item["question"], index)
        answer = answerer.answer(item["question"], hits)
        expected = {(s["document"], s["page"]) for s in item.get("expected_sources", [])}
        retrieved = {(h.chunk.document, h.chunk.page) for h in hits}
        used = {(s["document"], s["page"]) for s in answer.sources}
        results.append({
            **item,
            "actual_answer": answer.text,
            "abstained": answer.abstained,
            "retrieval_hit": bool(expected & retrieved) if expected else None,
            "all_expected_pages_retrieved": expected <= retrieved if expected else None,
            "expected_page_in_context": bool(expected & used) if expected else None,
            "abstention_matches_expectation": answer.abstained != item["answerable"],
            "sources": answer.sources,
            "seconds": round(perf_counter() - started, 3),
            "manual_answer_correct": None,
            "manual_citation_support": None,
        })
        print(f"{len(results)}/{len(questions)}: {results[-1]['seconds']} s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}. Grade answer correctness and citation support manually.")


if __name__ == "__main__":
    main()
