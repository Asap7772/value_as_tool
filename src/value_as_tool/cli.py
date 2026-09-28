"""Command-line interface for the standalone experiment pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import parse_overrides
from .harnesses import (
    BUILTIN_HARNESSES,
    builtin_entrypoint_for_condition,
    load_harness,
    resolve_harness,
)
from .pipeline import (
    judge,
    load_context,
    preflight,
    prepare,
    report,
    run_all,
    serve,
    smoke,
    solve,
    status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="value-as-tool",
        description="Run the standalone token-budgeted IMO evaluation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment.yaml"),
        help="experiment YAML (relative artifact paths are resolved from this file)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a strict config value; may be repeated",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight_parser = commands.add_parser(
        "preflight", help="check configuration and local prerequisites"
    )
    preflight_parser.add_argument(
        "--check-endpoints",
        action="store_true",
        help="also query configured /models endpoints",
    )
    preflight_parser.add_argument(
        "--require-server-binaries",
        action="store_true",
        help="treat missing SGLang/vLLM executables as errors",
    )

    prepare_parser = commands.add_parser(
        "prepare", help="download pinned assets and build the schedule"
    )
    prepare_parser.add_argument(
        "--tokenizers-only",
        action="store_true",
        help="for remote endpoints, download tokenizer/config files but not model weights",
    )
    for name in ("solve", "judge"):
        stage = commands.add_parser(name, help=f"run a resumable {name} shard")
        stage.add_argument("--shard-index", type=int, default=0)
        stage.add_argument("--shard-count", type=int, default=1)
    commands.add_parser("report", help="join schedule cells and write metric reports")
    commands.add_parser("status", help="summarize solve and judge artifact states")

    smoke_parser = commands.add_parser(
        "smoke",
        help="run one reference-bearing proof through every selected method and judge it",
    )
    smoke_parser.add_argument(
        "--benchmark", choices=("imo_proof", "proofbench"), default="imo_proof"
    )
    smoke_parser.add_argument("--problem-id")
    smoke_parser.add_argument("--seed", type=int, default=0)

    serve_parser = commands.add_parser("serve", help="exec a prepared local model server")
    serve_parser.add_argument("role", choices=("qwen", "judge"))
    serve_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    serve_parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="append one backend-specific server argument; may be repeated",
    )
    harness_parser = commands.add_parser(
        "harness", help="inspect and validate proof-harness modules"
    )
    harness_commands = harness_parser.add_subparsers(
        dest="harness_command", required=True
    )
    harness_commands.add_parser("list", help="list the built-in harness registry")
    harness_validate = harness_commands.add_parser(
        "validate",
        help="import and validate harnesses (configured harnesses by default)",
    )
    harness_validate.add_argument(
        "entrypoints",
        nargs="*",
        metavar="MODULE:CLASS",
        help="explicit harness entrypoint; may be repeated",
    )
    commands.add_parser("all", help="run prepare, solve, judge, and report in order")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _harness_record(entrypoint: str) -> dict[str, Any]:
    spec = resolve_harness(entrypoint)
    return {
        "harness_id": spec.harness_id,
        "display_name": spec.display_name,
        "access": spec.access,
        "requires_reference": spec.requires_reference,
        "condition": spec.condition.value if spec.condition is not None else None,
        "entrypoint": spec.entrypoint,
        "source_sha256": spec.source_sha256,
    }


def _inspect_harnesses(
    entrypoints: Sequence[str], *, instantiate: bool
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_ids: dict[str, str] = {}
    for entrypoint in entrypoints:
        try:
            record = _harness_record(entrypoint)
            if instantiate:
                load_harness(entrypoint, source_sha256=record["source_sha256"])
            previous = seen_ids.get(record["harness_id"])
            if previous is not None:
                raise ValueError(
                    f"duplicate harness id {record['harness_id']!r}: "
                    f"{previous!r} and {entrypoint!r}"
                )
            seen_ids[record["harness_id"]] = entrypoint
            records.append(record)
        except Exception as exc:
            errors.append(
                {
                    "entrypoint": entrypoint,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": not errors,
        "count": len(records),
        "harnesses": records,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    # Local interactive use gets the same credential behavior as the Slurm
    # launchers. Existing environment values always win and nothing is written.
    load_dotenv(config_path.parent / ".env", override=False)
    try:
        overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "harness" and args.harness_command == "list":
        result = _inspect_harnesses(BUILTIN_HARNESSES, instantiate=False)
        _print(result)
        return 0 if result["ok"] else 1

    if args.command == "harness" and args.harness_command == "validate":
        entrypoints = tuple(args.entrypoints)
        if not entrypoints:
            context = load_context(config_path, overrides=overrides)
            configured = context.config.evaluation.harnesses
            entrypoints = (
                tuple(configured)
                if configured is not None
                else tuple(
                    builtin_entrypoint_for_condition(condition)
                    for condition in context.config.evaluation.conditions
                )
            )
        result = _inspect_harnesses(entrypoints, instantiate=True)
        _print(result)
        return 0 if result["ok"] else 1

    context = load_context(config_path, overrides=overrides)

    if args.command == "preflight":
        result = preflight(
            context,
            check_endpoints=args.check_endpoints,
            require_server_binaries=args.require_server_binaries,
        )
        _print(result.to_dict())
        return 0 if result.ok else 1
    if args.command == "prepare":
        _print(prepare(context, tokenizers_only=args.tokenizers_only))
        return 0
    if args.command == "solve":
        _print(
            asyncio.run(
                solve(
                    context,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                )
            )
        )
        return 0
    if args.command == "judge":
        _print(
            asyncio.run(
                judge(
                    context,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                )
            )
        )
        return 0
    if args.command == "report":
        _print(report(context))
        return 0
    if args.command == "status":
        _print(status(context))
        return 0
    if args.command == "smoke":
        result = asyncio.run(
            smoke(
                context,
                benchmark=args.benchmark,
                problem_id=args.problem_id,
                seed=args.seed,
            )
        )
        _print(result)
        return 0 if result["ok"] else 1
    if args.command == "serve":
        serve(
            context,
            args.role,
            tensor_parallel_size=args.tensor_parallel_size,
            extra_args=args.extra_arg,
        )
        return 0  # os.execvpe does not return on success.
    if args.command == "all":
        _print(asyncio.run(run_all(context)))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["build_parser", "main"]
