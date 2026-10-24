"""Command line entry point for local-first platform engineering labs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml_labs.cards import evaluate_cards, train_cards
from ml_labs.data import prepare_cards, prepare_movies
from ml_labs.movies import evaluate_movies, train_movies
from ml_labs.rag import (
    evaluate_retrieval,
    index_documents,
    prepare_documents,
    query_azure,
    query_local,
)
from ml_labs.registry import register_candidate

DEFAULT_WORK = Path(".ml-labs")


def _path(value: str) -> Path:
    return Path(value)


def _emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _candidate_default(task: str) -> Path:
    return DEFAULT_WORK / "runs" / task / "candidate.json"


def _add_output(parser, default: str) -> None:
    parser.add_argument("--output", type=_path, default=Path(default))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-labs",
        description="Run reproducible local ML platform labs; cloud resources are never created.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cards = commands.add_parser("cards", help="Synthetic ID-card-shaped object detection (not real IDs)")
    card_cmd = cards.add_subparsers(dest="action", required=True)
    card_prepare = card_cmd.add_parser("prepare")
    _add_output(card_prepare, "data/cards")
    card_prepare.add_argument("--seed", type=int, default=42)
    card_prepare.add_argument("--train", type=int, default=80)
    card_prepare.add_argument("--validation", type=int, default=20)
    card_prepare.add_argument("--test", type=int, default=20)
    card_train = card_cmd.add_parser("train")
    card_train.add_argument("--data", type=_path, default=Path("data/cards"))
    _add_output(card_train, ".ml-labs/runs/cards")
    card_train.add_argument("--backend", choices=["color_component_baseline", "yolov8"],
                            default="color_component_baseline")
    card_train.add_argument("--epochs", type=int, default=5)
    card_train.add_argument("--work", type=_path, default=DEFAULT_WORK / "tracking")
    card_eval = card_cmd.add_parser("evaluate")
    card_eval.add_argument("--candidate", type=_path, default=_candidate_default("cards"))
    card_eval.add_argument("--data", type=_path, default=Path("data/cards"))
    card_eval.add_argument("--min-iou", type=float, default=0.75)
    card_eval.add_argument("--report", type=_path, default=Path(".ml-labs/reports/cards-test.json"))

    movies = commands.add_parser("movies", help="Synthetic movie recommendation pipeline")
    movie_cmd = movies.add_subparsers(dest="action", required=True)
    movie_prepare = movie_cmd.add_parser("prepare")
    _add_output(movie_prepare, "data/movies")
    movie_prepare.add_argument("--seed", type=int, default=42)
    movie_import = movie_cmd.add_parser("import-movielens")
    movie_import.add_argument("--archive", type=_path, required=True)
    movie_import.add_argument("--output", type=_path, required=True)
    movie_import.add_argument("--min-events", type=int, default=10)
    movie_train = movie_cmd.add_parser("train")
    movie_train.add_argument("--data", type=_path, default=Path("data/movies"))
    _add_output(movie_train, ".ml-labs/runs/movies")
    movie_train.add_argument("--rank", type=int, default=4)
    movie_train.add_argument("--seed", type=int, default=42)
    movie_train.add_argument("--work", type=_path, default=DEFAULT_WORK / "tracking")
    movie_eval = movie_cmd.add_parser("evaluate")
    movie_eval.add_argument("--candidate", type=_path, default=_candidate_default("movies"))
    movie_eval.add_argument("--data", type=_path, default=Path("data/movies"))
    movie_eval.add_argument("--max-rmse", type=float, default=1.5)
    movie_eval.add_argument("--report", type=_path, default=Path(".ml-labs/reports/movies-test.json"))

    rag = commands.add_parser("rag", help="Synthetic document retrieval and LLMOps evaluation")
    rag_cmd = rag.add_subparsers(dest="action", required=True)
    rag_prepare = rag_cmd.add_parser("prepare")
    _add_output(rag_prepare, "data/documents")
    rag_index = rag_cmd.add_parser("index")
    rag_index.add_argument("--data", type=_path, default=Path("data/documents"))
    _add_output(rag_index, ".ml-labs/runs/rag")
    rag_index.add_argument("--top-k", type=int, default=3)
    rag_index.add_argument("--work", type=_path, default=DEFAULT_WORK / "tracking")
    rag_query = rag_cmd.add_parser("query")
    rag_query.add_argument("--provider", choices=["local", "azure"], default="local")
    rag_query.add_argument("--candidate", type=_path, default=_candidate_default("rag"))
    rag_query.add_argument("--query", required=True)
    rag_query.add_argument("--top-k", type=int, default=3)
    rag_eval = rag_cmd.add_parser("evaluate")
    rag_eval.add_argument("--candidate", type=_path, default=_candidate_default("rag"))
    rag_eval.add_argument("--report", type=_path, default=Path(".ml-labs/reports/rag-retrieval.json"))
    rag_azure_index = rag_cmd.add_parser(
        "azure-index", help="Create/update an index in an existing Azure AI Search service"
    )
    rag_azure_index.add_argument("--data", type=_path, default=Path("data/documents"))

    registry = commands.add_parser("register", help="Register a gated MLflow model and verify reload")
    registry.add_argument("--candidate", type=_path, required=True)
    registry.add_argument("--report", type=_path, required=True)
    registry.add_argument("--name", required=True)
    registry.add_argument("--alias", default="champion")

    serve = commands.add_parser("serve", help="Serve selected MLflow model candidates over HTTP")
    serve.add_argument("--cards", type=_path)
    serve.add_argument("--movies", type=_path)
    serve.add_argument("--rag", type=_path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    smoke = commands.add_parser("smoke", help="Execute all three local train/evaluate/register paths")
    smoke.add_argument("--work", type=_path, default=DEFAULT_WORK)

    azure = commands.add_parser("azure", help="Render Azure ML SDK v2 config; never provisions resources")
    azure_cmd = azure.add_subparsers(dest="action", required=True)
    for action in ("pipeline", "endpoint"):
        sub = azure_cmd.add_parser(action)
        sub.add_argument("--output", type=_path, required=True)
        sub.add_argument("--candidate", type=_path)
        sub.add_argument("--report", type=_path)
        sub.add_argument("--compute", default="azureml:cpu-cluster")
        if action == "pipeline":
            sub.add_argument("--cards-backend", choices=["color_component_baseline", "yolov8"],
                             default="color_component_baseline")
        sub.add_argument("--execute", action="store_true",
                         help="Submit a rendered job to an existing workspace; does not create Azure resources.")

    tools = commands.add_parser("tools", help="Native optional tool adapters")
    tool_cmd = tools.add_subparsers(dest="tool", required=True)
    feast = tool_cmd.add_parser("feast", help="Initialize or run Feast against an existing local feature repo")
    feast_cmd = feast.add_subparsers(dest="action", required=True)
    feast_init = feast_cmd.add_parser(
        "init", help="Build train-only point-in-time user snapshots from prepared movie data"
    )
    feast_init.add_argument("--data", type=_path, required=True,
                            help="Prepared movie dataset directory (ml-labs movies prepare)")
    _add_output(feast_init, "integrations/feast")
    feast_run = feast_cmd.add_parser("run")
    feast_run.add_argument("--repo", type=_path, default=Path("integrations/feast"))
    feast_run.add_argument(
        "--action", choices=["apply", "materialize", "online", "historical"], required=True
    )
    feast_run.add_argument("--user-id", help="Entity from the dataset (default: first user)")
    feast_run.add_argument("--end-date",
                           help="materialize only: timezone-aware ISO end time (default: all data)")
    kfp = tool_cmd.add_parser("kfp", help="Compile (not submit) the movie Kubeflow pipeline")
    kfp_cmd = kfp.add_subparsers(dest="action", required=True)
    kfp_compile = kfp_cmd.add_parser("compile")
    kfp_compile.add_argument("--output", type=_path, default=Path("integrations/kfp/movie-pipeline.yaml"))
    kfp_compile.add_argument("--source-revision", required=True,
                             help="Git commit baked into the image, e.g. $(git rev-parse HEAD)")
    kfp_compile.add_argument("--image", default=None,
                             help="Pushed image built from ./Dockerfile (default: explicit placeholder)")
    bento = tool_cmd.add_parser("bento", help="Build or run a BentoML service")
    bento_cmd = bento.add_subparsers(dest="action", required=True)
    bento_build = bento_cmd.add_parser("build")
    bento_build.add_argument("--candidate", type=_path, required=True)
    bento_build.add_argument("--output", type=_path, default=Path("integrations/bento/bentofile.yaml"))
    drift = tool_cmd.add_parser("drift", help="Run Evidently drift report or built-in PSI monitor")
    drift.add_argument("--reference", type=_path, required=True)
    drift.add_argument("--current", type=_path, required=True)
    drift.add_argument("--output", type=_path, required=True)
    drift.add_argument("--backend", choices=["evidently", "psi"], default="evidently")

    promptfoo = tool_cmd.add_parser("promptfoo", help="Render or run a native Promptfoo eval")
    promptfoo_cmd = promptfoo.add_subparsers(dest="action", required=True)
    promptfoo_init = promptfoo_cmd.add_parser("init")
    promptfoo_init.add_argument(
        "--output", type=_path, default=Path("integrations/promptfoo/promptfooconfig.yaml")
    )
    promptfoo_run = promptfoo_cmd.add_parser("run")
    promptfoo_run.add_argument(
        "--config", type=_path, default=Path("integrations/promptfoo/promptfooconfig.yaml")
    )

    for tool_name, help_text in (
        ("guardrails", "Validate a RAG response using Guardrails AI"),
        ("deepeval", "Run model-based answer relevance evaluation with DeepEval"),
        ("langfuse", "Trace local RAG retrieval to an existing Langfuse project"),
    ):
        tool_parser = tool_cmd.add_parser(tool_name, help=help_text)
        tool_parser.add_argument("--candidate", type=_path, required=True)
        tool_parser.add_argument("--query", required=True)
        if tool_name == "deepeval":
            tool_parser.add_argument("--threshold", type=float, default=0.7)

    redisvl = tool_cmd.add_parser("redisvl", help="Index and query using native RedisVL")
    redisvl_cmd = redisvl.add_subparsers(dest="action", required=True)
    redisvl_index_parser = redisvl_cmd.add_parser("index")
    redisvl_index_parser.add_argument("--data", type=_path, default=Path("data/documents"))
    redisvl_index_parser.add_argument("--state", type=_path, default=Path(".ml-labs/redisvl/state.joblib"))
    redisvl_index_parser.add_argument("--index", default="ml-labs-documents")
    redisvl_index_parser.add_argument("--redis-url")
    redisvl_query_parser = redisvl_cmd.add_parser("query")
    redisvl_query_parser.add_argument("--query", required=True)
    redisvl_query_parser.add_argument("--state", type=_path, default=Path(".ml-labs/redisvl/state.joblib"))
    redisvl_query_parser.add_argument("--top-k", type=int, default=3)
    redisvl_query_parser.add_argument("--redis-url")
    return parser


def _execute(args) -> int:
    if args.command == "cards":
        if args.action == "prepare":
            result = prepare_cards(
                args.output, seed=args.seed, train=args.train,
                validation=args.validation, test=args.test,
            )
        elif args.action == "train":
            result = train_cards(args.data, args.output, args.backend, args.work, args.epochs)
        else:
            result = evaluate_cards(args.candidate, args.data, args.report, args.min_iou)
        _emit({"result": result})
    elif args.command == "movies":
        if args.action == "import-movielens":
            from ml_labs.movielens import import_movielens

            result = import_movielens(args.archive, args.output, args.min_events)
        elif args.action == "prepare":
            result = prepare_movies(args.output, seed=args.seed)
        elif args.action == "train":
            result = train_movies(args.data, args.output, args.rank, args.seed, args.work)
        else:
            result = evaluate_movies(args.candidate, args.data, args.report, args.max_rmse)
        _emit({"result": result})
    elif args.command == "rag":
        if args.action == "prepare":
            result = prepare_documents(args.output)
        elif args.action == "index":
            result = index_documents(args.data, args.output, args.top_k, args.work)
        elif args.action == "query":
            result = (
                query_local(args.candidate, args.query)
                if args.provider == "local"
                else query_azure(args.query, args.top_k)
            )
        elif args.action == "evaluate":
            result = evaluate_retrieval(args.candidate, args.report)
        else:
            from ml_labs.rag import create_azure_search_index

            result = create_azure_search_index(args.data)
        _emit(result)
    elif args.command == "register":
        _emit(register_candidate(args.candidate, args.name, args.report, args.alias))
    elif args.command == "serve":
        if not any((args.cards, args.movies, args.rag)):
            raise ValueError("Provide at least one candidate using --cards, --movies, or --rag.")
        import uvicorn

        from ml_labs.api import create_app

        uvicorn.run(
            create_app(args.cards, args.movies, args.rag),
            host=args.host,
            port=args.port,
        )
    elif args.command == "smoke":
        result = _smoke(args.work)
        _emit(result)
    elif args.command == "azure":
        from ml_labs.azure_jobs import render_endpoint, render_pipeline, submit_existing_workspace

        if args.execute:
            if args.action != "pipeline":
                raise ValueError("Only pipeline submission is automated; endpoint deployment is render-only.")
            _emit(submit_existing_workspace(args.output))
        elif args.action == "pipeline":
            _emit({"output": render_pipeline(
                args.output, compute=args.compute, cards_backend=args.cards_backend,
            )})
        else:
            if not args.candidate:
                raise ValueError("Endpoint rendering needs --candidate.")
            if not args.report:
                raise ValueError("Endpoint rendering needs a passing --report.")
            _emit({"output": render_endpoint(args.candidate, args.output, args.report)})
    elif args.command == "tools":
        _run_tool(args)
    return 0


def _smoke(work: Path) -> dict:
    data = work / "data"
    runs = work / "runs"
    reports = work / "reports"
    tracking = work / "tracking"
    prepare_cards(data / "cards", train=24, validation=10, test=10)
    prepare_movies(data / "movies", users=24, items=18)
    prepare_documents(data / "documents")
    card_candidate = train_cards(data / "cards", runs / "cards", work=tracking)
    movie_candidate = train_movies(data / "movies", runs / "movies", work=tracking)
    rag_candidate = index_documents(data / "documents", runs / "rag", work=tracking)
    card_report_path = reports / "cards.json"
    movie_report_path = reports / "movies.json"
    rag_report_path = reports / "rag.json"
    card_report = evaluate_cards(card_candidate, data / "cards", card_report_path, min_iou=0.6)
    movie_report = evaluate_movies(movie_candidate, data / "movies", movie_report_path)
    rag_report = evaluate_retrieval(rag_candidate, rag_report_path)
    registered = []
    for task, candidate in (
        ("cards", card_candidate), ("movies", movie_candidate), ("rag", rag_candidate)
    ):
        report_path = {"cards": card_report_path, "movies": movie_report_path, "rag": rag_report_path}[task]
        registered.append(register_candidate(candidate, f"ml-labs-{task}", report_path))
    return {
        "cards": card_report, "movies": movie_report, "rag": rag_report,
        "registered": [item["registered_uri"] for item in registered],
    }


def _run_tool(args) -> None:
    if args.tool == "feast":
        from ml_labs.integrations import feast_init, feast_run

        if args.action == "init":
            _emit({"repo": feast_init(args.output, args.data)})
        else:
            if args.end_date and args.action != "materialize":
                raise ValueError("--end-date applies only to --action materialize.")
            _emit(feast_run(args.repo, args.action, user_id=args.user_id, end_date=args.end_date))
    elif args.tool == "kfp":
        from ml_labs.integrations import KFP_IMAGE_PLACEHOLDER, compile_kfp

        _emit(compile_kfp(args.output, args.source_revision, args.image or KFP_IMAGE_PLACEHOLDER))
    elif args.tool == "bento":
        from ml_labs.integrations import build_bento

        _emit({"output": build_bento(args.candidate, args.output)})
    elif args.tool == "promptfoo":
        from ml_labs.llmops import create_promptfoo_config, run_promptfoo

        if args.action == "init":
            _emit({"config": create_promptfoo_config(args.output)})
        else:
            _emit(run_promptfoo(args.config))
    elif args.tool in {"guardrails", "deepeval", "langfuse"}:
        from ml_labs.llmops import (
            check_candidate,
            evaluate_with_deepeval,
            trace_with_langfuse,
            validate_with_guardrails,
        )

        check_candidate(args.candidate)
        if args.tool == "guardrails":
            from ml_labs.rag import query_local

            response = query_local(args.candidate, args.query)
            result = validate_with_guardrails(response["answer"], response["citations"])
        elif args.tool == "deepeval":
            result = evaluate_with_deepeval(args.candidate, args.query, args.threshold)
        else:
            result = trace_with_langfuse(args.candidate, args.query)
        _emit(result)
    elif args.tool == "redisvl":
        from ml_labs.llmops import redisvl_index, redisvl_query

        if args.action == "index":
            result = redisvl_index(args.data, args.state, args.index, args.redis_url)
        else:
            result = redisvl_query(args.query, args.state, args.top_k, args.redis_url)
        _emit(result)
    else:
        from ml_labs.monitoring import create_drift_report

        _emit(create_drift_report(
            args.reference, args.current, args.output, backend=args.backend
        ))


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        sys.exit(_execute(args))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"ml-labs: error: {exc}\n")


if __name__ == "__main__":
    main()
