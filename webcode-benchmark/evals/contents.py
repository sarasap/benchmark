"""Contents eval — extraction fidelity against golden markdown."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from shared.graders import ContentsGrader
from shared.searchers import Searcher
from shared.searchers.claude import ClaudeWebFetchSearcher
from shared.searchers.exa import ExaSearcher
from shared.searchers.parallel import ParallelSearcher
from shared.searchers.tavily import TavilySearcher

from src.metrics import compute_contents_metrics

console = Console()
logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"
GOLDEN_FILE = DATA_DIR / "contents" / "golden_markdown.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "eval_results"


def load_queries(limit: int | None = None) -> list[dict]:
    filepath = DATA_DIR / "contents" / "code_contents.jsonl"
    if not filepath.exists():
        return []
    queries = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
    return queries[:limit] if limit else queries


def _load_golden_markdown() -> dict[str, str]:
    if not GOLDEN_FILE.exists():
        return {}
    golden = {}
    with open(GOLDEN_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            golden[row["id"]] = row["expected_markdown"]
    return golden


def _build_exa_searcher() -> ExaSearcher:
    return ExaSearcher(max_age_hours=0)


SEARCHER_BUILDERS: dict[str, callable] = {
    "exa": _build_exa_searcher,
    "tavily": TavilySearcher,
    "parallel": ParallelSearcher,
    "claude": ClaudeWebFetchSearcher,
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
    save: bool = False,
    concurrency: int = 3,
    grader_model: str = "gemini-2.5-pro",
):
    queries = load_queries(limit)
    if not queries:
        console.print("[red]No queries found. Ensure data/contents/code_contents.jsonl exists.[/red]")
        return

    golden_markdown = _load_golden_markdown()
    if not golden_markdown:
        console.print(
            "[red]Golden markdown not found.[/red]\n"
            f"  Expected file: [bold]{GOLDEN_FILE}[/bold]\n"
            "  The golden markdown is not distributed with the dataset for licensing reasons.\n"
            "  Generate it by fetching each URL in code_contents.jsonl and writing a JSONL\n"
            '  with {id, expected_markdown} rows to the path above.'
        )
        return

    missing = [q["id"] for q in queries if q["id"] not in golden_markdown]
    if missing:
        console.print(f"[yellow]Warning: {len(missing)} queries have no golden markdown and will be skipped.[/yellow]")
        queries = [q for q in queries if q["id"] in golden_markdown]

    searchers = [s for name in searcher_names if (s := build_searcher(name))]
    if not searchers:
        console.print("[red]No searchers available.[/red]")
        return

    grader = ContentsGrader(model=grader_model)
    semaphore = asyncio.Semaphore(concurrency)
    all_results: dict[str, list[dict]] = {}

    console.print("\n[bold]Contents Extraction Eval[/bold]")
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
            grades = []

            async def process(q: dict) -> dict:
                async with semaphore:
                    url = q.get("url", "")
                    entry_id = q.get("id", "")
                    golden = golden_markdown.get(entry_id, "")
                    start = time.time()
                    try:
                        results = await searcher.extract(url)
                        extracted = results[0].text if results else ""
                    except Exception as e:
                        logger.warning(f"Extraction failed for {url}: {e}")
                        extracted = ""
                    latency = time.time() - start

                    grade = await grader.grade(url, golden, extracted)
                    progress.advance(task_id)

                    result = {
                        "id": entry_id,
                        "url": url,
                        "latency": round(latency, 2),
                        "extracted": extracted,
                        **grade.scores,
                    }

                    if save:
                        extracted_dir = RESULTS_DIR / searcher.name / "extracted"
                        metrics_dir = RESULTS_DIR / searcher.name / "metrics"
                        extracted_dir.mkdir(parents=True, exist_ok=True)
                        metrics_dir.mkdir(parents=True, exist_ok=True)

                        with open(extracted_dir / f"{entry_id}.json", "w") as f:
                            json.dump({"id": entry_id, "url": url, "extracted": extracted}, f, indent=2)

                        metrics = {k: v for k, v in result.items() if k not in ("extracted",)}
                        with open(metrics_dir / f"{entry_id}.json", "w") as f:
                            json.dump(metrics, f, indent=2)

                    return result

            grades = await asyncio.gather(*[process(q) for q in queries])
            all_results[searcher.name] = list(grades)

    _print_summary(all_results)

    if save:
        console.print(f"\n[green]Results saved to {RESULTS_DIR}/[/green]")


def load_metrics_from_disk(searcher_name: str) -> list[dict]:
    metrics_dir = RESULTS_DIR / searcher_name / "metrics"
    if not metrics_dir.exists():
        console.print(f"[yellow]No metrics found for {searcher_name} at {metrics_dir}[/yellow]")
        return []
    results = []
    for path in sorted(metrics_dir.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def run_summary_only(searcher_names: list[str]):
    all_results: dict[str, list[dict]] = {}
    for name in searcher_names:
        grades = load_metrics_from_disk(name)
        if not grades:
            continue

        extracted_dir = RESULTS_DIR / name / "extracted"
        filtered = []
        anomalies = []
        for g in grades:
            entry_id = g["id"]
            extracted_file = extracted_dir / f"{entry_id}.json"
            extracted = ""
            if extracted_file.exists():
                extracted = json.load(open(extracted_file)).get("extracted", "")
            if extracted.strip() and g["completeness"] == 0.0 and g["accuracy"] == 0.0 and g["structure"] == 0.0:
                anomalies.append(entry_id)
            else:
                filtered.append(g)
        if anomalies:
            console.print(f"[yellow]{name}: excluded {len(anomalies)} anomalies (non-empty extraction, all scores=0): {anomalies}[/yellow]")
        grades = filtered

        all_results[name] = grades

    if not all_results:
        console.print("[red]No saved metrics found for any searcher.[/red]")
        return
    _print_summary(all_results)


def _print_summary(all_results: dict[str, list[dict]]):
    table = Table(title="Contents Extraction Results")
    table.add_column("Searcher", style="cyan")
    for col in ["Completeness", "Accuracy", "Structure", "Signal", "Code Recall", "Table Recall", "ROUGE-L", "Queries"]:
        table.add_column(col, justify="right")

    for name, grades in all_results.items():
        metrics = compute_contents_metrics(grades)
        table.add_row(
            name,
            f"{metrics.completeness:.1%}",
            f"{metrics.accuracy:.1%}",
            f"{metrics.structure:.1%}",
            f"{metrics.signal:.1%}",
            f"{metrics.det_code_block_recall:.1%}",
            f"{metrics.det_table_recall:.1%}",
            f"{metrics.det_rouge_l:.1%}",
            str(metrics.num_queries),
        )

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Contents extraction eval")
    parser.add_argument("--searchers", nargs="+", default=["exa"], help="Searchers to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of queries")
    parser.add_argument("--save", action="store_true", help="Save per-id results to eval_results/")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--grader-model", default="gemini-2.5-pro")
    parser.add_argument("--summary-only", action="store_true", help="Print summary from saved metrics without running extraction")
    args = parser.parse_args()

    if args.summary_only:
        run_summary_only(args.searchers)
        return

    asyncio.run(run(
        searcher_names=args.searchers,
        limit=args.limit,
        save=args.save,
        concurrency=args.concurrency,
        grader_model=args.grader_model,
    ))


if __name__ == "__main__":
    main()
