"""CLI entry point for ResearchForge."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)

logger = structlog.get_logger()


async def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a file or directory into the RAG corpus."""
    from researchforge.config import get_settings
    from researchforge.db.repository import Repository
    from researchforge.rag.ingest import ingest_directory, ingest_file
    from researchforge.rag.store import VectorStore

    settings = get_settings()
    store = VectorStore()
    repo = Repository(settings.storage.metadata_db_path)
    await repo.initialize()

    path = Path(args.path)
    try:
        if path.is_dir():
            results = await ingest_directory(path, store, repo)
            for r in results:
                status = "skipped (duplicate)" if r.skipped else f"{r.chunk_count} chunks"
                print(f"  {r.source_path}: {status}")
            total = sum(r.chunk_count for r in results if not r.skipped)
            ingested = len([r for r in results if not r.skipped])
            print(f"\nIngested {ingested} files, {total} chunks total.")
        elif path.is_file():
            result = await ingest_file(path, store, repo)
            if result.skipped:
                print(f"Skipped (already ingested): {result.source_path}")
            else:
                print(f"Ingested: {result.source_path} ({result.chunk_count} chunks)")
        else:
            print(f"Error: Path not found: {path}", file=sys.stderr)
            sys.exit(1)
    finally:
        await repo.close()


async def cmd_search(args: argparse.Namespace) -> None:
    """Search the corpus using hybrid retrieval."""
    from researchforge.rag.retriever import retrieve
    from researchforge.rag.store import VectorStore

    store = VectorStore()
    results = await retrieve(args.query, store, top_k=args.limit)

    if not results:
        print("No results found.")
        return

    for i, chunk in enumerate(results, 1):
        source = chunk.get("source_path", "unknown")
        section = chunk.get("section_h1", "")
        text = chunk["text"][:200].replace("\n", " ")
        print(f"\n[{i}] Source: {source}")
        if section:
            print(f"    Section: {section}")
        print(f"    {text}...")


async def cmd_research(args: argparse.Namespace) -> None:
    """Run the full multi-agent research pipeline."""

    from researchforge.agents.graph import run_pipeline
    from researchforge.config import get_settings
    from researchforge.db.repository import Repository

    settings = get_settings()
    depth = getattr(args, "depth", "standard")
    pipeline_id = str(uuid.uuid4())

    repo = Repository(settings.storage.metadata_db_path)
    await repo.initialize()

    try:
        if args.verbose:
            agents = "Planner → Gatherer → Analyst → Writer"
            if depth == "standard":
                agents = "Planner → Gatherer → Analyst → Critic → Writer"
            print(
                f"[Pipeline] Starting {depth} research pipeline ({agents})",
                file=sys.stderr,
            )

        # Run the full pipeline
        final_state = await run_pipeline(
            args.question, depth=depth, pipeline_id=pipeline_id
        )

        briefing = final_state.get("briefing", "")
        status = final_state.get("status", "unknown")
        trace = final_state.get("trace", [])
        errors = final_state.get("errors", [])

        # Store briefing in DB
        await repo.insert_briefing(
            pipeline_id, args.question, status=status
        )
        await repo.update_briefing(
            pipeline_id,
            briefing_markdown=briefing,
            status=status,
            pipeline_trace=trace,
        )

        # Corpus feedback loop
        critic_verdict = None
        for entry in trace:
            if entry.get("agent") == "critic":
                critic_verdict = entry.get("verdict")
                break

        try:
            from researchforge.rag.feedback import maybe_ingest_briefing
            from researchforge.rag.store import VectorStore

            feedback_store = VectorStore()
            feedback_result = await maybe_ingest_briefing(
                pipeline_id, repo, feedback_store, critic_verdict=critic_verdict
            )
            if args.verbose:
                print(
                    f"[Feedback] score={feedback_result['quality_score']}, "
                    f"ingested={feedback_result['ingested']}",
                    file=sys.stderr,
                )
        except Exception as fb_exc:
            if args.verbose:
                print(f"[Feedback] Error: {fb_exc}", file=sys.stderr)

        # Output the briefing
        if briefing:
            print(briefing)
        else:
            print("Pipeline completed but no briefing was generated.", file=sys.stderr)

        if args.verbose:
            print(f"\n[Pipeline ID: {pipeline_id}]", file=sys.stderr)
            print(f"[Status: {status}]", file=sys.stderr)
            for entry in trace:
                agent = entry.get("agent", "?")
                model = entry.get("model", "?")
                dur = entry.get("duration_ms", 0)
                st = entry.get("status", "?")
                fb = " (fallback)" if entry.get("fallback_used") else ""
                print(
                    f"  [{agent}] {model}{fb} — {dur}ms — {st}",
                    file=sys.stderr,
                )
            if errors:
                print(f"[Errors: {len(errors)}]", file=sys.stderr)
                for e in errors:
                    print(f"  - {e}", file=sys.stderr)
    finally:
        await repo.close()


async def cmd_serve(args: argparse.Namespace) -> None:
    """Start the web server."""
    import uvicorn

    from researchforge.config import get_settings

    settings = get_settings()
    config = uvicorn.Config(
        "researchforge.web.app:create_app",
        factory=True,
        host=settings.web.host,
        port=settings.web.port,
        reload=getattr(args, "reload", False),
    )
    server = uvicorn.Server(config)
    await server.serve()


async def _cmd_eval(args: argparse.Namespace) -> None:
    """Handle eval subcommands."""
    eval_cmd = getattr(args, "eval_command", None)

    if eval_cmd == "run":
        from researchforge.eval.runner import run_full_eval

        result = await run_full_eval(skip_e2e=args.skip_e2e)

        if result.retrieval:
            r = result.retrieval
            print(
                f"Retrieval: P@5={r.get('mean_precision_at_k', 0):.4f}, "
                f"R@5={r.get('mean_recall_at_k', 0):.4f}, "
                f"MRR={r.get('mean_mrr', 0):.4f}"
            )

        if result.agent_scores:
            for role, data in result.agent_scores.items():
                if isinstance(data, dict) and "mean_score" in data:
                    print(f"Agent [{role}]: score={data['mean_score']:.4f}")

        if result.e2e:
            print(f"E2E: mean_score={result.e2e.get('mean_score', 0):.4f}")

        if result.regressions:
            print("\nRegressions detected:")
            for reg in result.regressions:
                print(
                    f"  {reg['metric']}: {reg['current']:.4f} "
                    f"(avg: {reg['rolling_avg']:.4f}, -{reg['drop_pct']}%)"
                )

    elif eval_cmd == "benchmark":
        from researchforge.eval.benchmark import benchmark_models

        models = [m.strip() for m in args.models.split(",")]
        result = await benchmark_models(args.role, models)

        print(f"\nBenchmark: {args.role}")
        print(f"{'Model':<25} {'Score':>8} {'Latency':>10} {'Tokens':>8}")
        print("-" * 55)
        for entry in result.summary_table():
            print(
                f"{entry['model']:<25} "
                f"{entry['weighted_score']:>8.4f} "
                f"{entry['latency_ms']:>8}ms "
                f"{entry['output_tokens']:>8}"
            )

    else:
        print("Usage: researchforge eval {run|benchmark}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="researchforge",
        description="ResearchForge — Local multi-agent research analyst",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into the corpus")
    p_ingest.add_argument("path", help="File or directory to ingest")

    # search
    p_search = subparsers.add_parser("search", help="Search the corpus")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=5, help="Number of results")

    # research
    p_research = subparsers.add_parser("research", help="Research a question")
    p_research.add_argument("question", help="Research question")
    p_research.add_argument(
        "--depth",
        choices=["standard", "quick"],
        default="standard",
        help="Pipeline depth: 'standard' (with critic) or 'quick' (skip critic)",
    )
    p_research.add_argument("--verbose", "-v", action="store_true", help="Show progress")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the web UI server")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # mcp
    subparsers.add_parser("mcp", help="Start the MCP server")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation suite")
    eval_sub = p_eval.add_subparsers(dest="eval_command")
    p_eval_run = eval_sub.add_parser("run", help="Run the full eval suite")
    p_eval_run.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end eval")
    p_eval_bench = eval_sub.add_parser("benchmark", help="Benchmark models for an agent role")
    p_eval_bench.add_argument("--role", required=True, help="Agent role to benchmark")
    p_eval_bench.add_argument("--models", required=True, help="Comma-separated model list")

    args = parser.parse_args()

    commands = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "research": cmd_research,
        "serve": cmd_serve,
    }

    handler = commands.get(args.command)
    if handler:
        asyncio.run(handler(args))
    elif args.command == "mcp":
        from researchforge.mcp_server.server import mcp

        mcp.run(transport="stdio")
    elif args.command == "eval":
        asyncio.run(_cmd_eval(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
