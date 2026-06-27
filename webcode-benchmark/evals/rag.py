"""RAG eval — full-web retrieval + synthesis on long-context code docs."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.agents import SimpleRAGAgent
from shared.graders import Citation, GroundedRAGGrader
from shared.searchers import Searcher
from shared.searchers.brave import BraveSearcher
from shared.searchers.exa import ExaSearcher
from shared.searchers.parallel import ParallelSearcher
from shared.searchers.perplexity import PerplexitySearcher
from shared.searchers.tavily import TavilySearcher

from src.metrics import compute_grounded_rag_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "eval_results_oneshot"


def load_queries(limit: int | None = None) -> list[dict]:
    filepath = DATA_DIR / "rag" / "code_rag.jsonl"
    if not filepath.exists():
        return []
    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
    return queries[:limit] if limit else queries


def _build_exa_searcher() -> ExaSearcher:
    return ExaSearcher(search_type="fast", include_text=False, include_highlights=True, max_age_hours=0)


def _build_brave_searcher() -> BraveSearcher:
    return BraveSearcher(search_type="llm_context")


SEARCHER_BUILDERS: dict[str, callable] = {
    "exa": _build_exa_searcher,
    "brave": _build_brave_searcher,
    "perplexity": PerplexitySearcher,
    "parallel": ParallelSearcher,
    "tavily": TavilySearcher,
}


def build_searcher(name: str) -> Searcher | None:
    builder = SEARCHER_BUILDERS.get(name)
    if builder is None:
        console.print(f"[yellow]Unknown searcher: {name}[/yellow]")
        return None
    try:
        return builder()
    except (ValueError, ImportError) as e:
        console.print(f"[yellow]{name}: {e}[/yellow]")
        return None


async def run(
    searcher_names: list[str],
    limit: int | None = None,
    output: str | None = None,
    num_results: int = 5,
    concurrency: int = 5,
    grader_model: str = "gemini-2.5-pro",
    rag_model: str = "gemini-2.5-flash",
    save: bool = False,
):
    queries = load_queries(limit)
    if not queries:
        console.print("[red]No queries found. Ensure data/rag/code_rag.jsonl exists.[/red]")
        return

    searchers = [s for name in searcher_names if (s := build_searcher(name))]
    if not searchers:
        console.print("[red]No searchers available.[/red]")
        return

    grader = GroundedRAGGrader(model=grader_model)
    rag_agent = SimpleRAGAgent(model=rag_model)
    semaphore = asyncio.Semaphore(concurrency)
    all_results: dict[str, list[dict]] = {}

    console.print("\n[bold]RAG Eval (Long-Context Code QA)[/bold]")
    console.print(f"  Queries: {len(queries)}")
    console.print(f"  Searchers: {[s.name for s in searchers]}\n")

    with Progress(
        TextColumn("[cyan]{task.fields[name]:>12}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for searcher in searchers:
            task_id = progress.add_task("", name=searcher.name, total=len(queries))

            async def process(q: dict, _searcher=searcher) -> dict:
                async with semaphore:
                    query_text = q["query"]
                    expected = q["expected_answer"]
                    entry_id = q.get("id", "")
                    start = time.time()

                    try:
                        results = await _searcher.search(query_text, num_results=num_results)
                    except Exception as e:
                        body = getattr(getattr(e, "response", None), "text", None)
                        if body:
                            logger.warning(f"Search failed for query: {e} | body={body}")
                        else:
                            logger.warning(f"Search failed for query: {e}")
                        results = []

                    latency = time.time() - start
                    rag_result = await rag_agent.synthesize(query_text, results)

                    citations = [
                        Citation(url=c.url, title=c.title, text=c.text)
                        for c in rag_result.citations
                    ]

                    logger.info(f"[{_searcher.name}] query={query_text!r} num_citations={len(citations)}")

                    grade = await grader.grade(
                        question=query_text,
                        expected_answer=expected,
                        predicted_answer=rag_result.answer,
                        citations=citations,
                    )
                    progress.advance(task_id)

                    result = {
                        "id": entry_id,
                        "query": query_text,
                        "latency": round(latency, 2),
                        **grade.scores,
                    }

                    if save:
                        out_dir = RESULTS_DIR / _searcher.name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_dir / f"{entry_id}.json", "w") as f:
                            json.dump(result, f, indent=2)

                        raw_dir = RESULTS_DIR / _searcher.name / "raw"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        raw = {
                            "id": entry_id,
                            "query": query_text,
                            "search_results": [
                                {"url": r.url, "title": r.title, "text": r.text, "highlights": r.highlights, "metadata": r.metadata}
                                for r in results
                            ],
                            "rag_answer": rag_result.answer,
                        }
                        with open(raw_dir / f"{entry_id}.json", "w") as f:
                            json.dump(raw, f, indent=2)

                    return result

            all_results[searcher.name] = await asyncio.gather(*[process(q) for q in queries])

    _print_summary(all_results)

    if output:
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to {output}[/green]")

    if save:
        console.print(f"\n[green]Per-query results saved to {RESULTS_DIR}/[/green]")


def _print_summary(all_results: dict[str, list[dict]]):
    table = Table(title="RAG Eval Results")
    table.add_column("Searcher", style="cyan")
    for col in ["Groundedness", "Correctness", "Citation Prec.", "Avg Tokens", "Queries"]:
        table.add_column(col, justify="right")

    for name, grades in all_results.items():
        metrics = compute_grounded_rag_metrics(grades)
        table.add_row(
            name,
            f"{metrics.groundedness:.1%}",
            f"{metrics.correctness:.1%}",
            f"{metrics.citation_precision:.1%}",
            f"{metrics.avg_citation_tokens:.0f}",
            str(metrics.num_queries),
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="RAG eval (long-context code QA)")
    parser.add_argument("--searchers", nargs="+", default=["exa"], help="Searchers to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--num-results", type=int, default=5, help="Results per search query")
    parser.add_argument("--output", "-o", help="Output file for results JSON")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--grader-model", default="gemini-2.5-pro")
    parser.add_argument("--rag-model", default="gemini-2.5-flash")
    parser.add_argument("--save", action="store_true", help="Save per-query results")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    asyncio.run(run(
        searcher_names=args.searchers,
        limit=args.limit,
        output=args.output,
        num_results=args.num_results,
        concurrency=args.concurrency,
        grader_model=args.grader_model,
        rag_model=args.rag_model,
        save=args.save,
    ))


if __name__ == "__main__":
    main()
