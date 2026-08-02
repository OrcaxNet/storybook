#!/usr/bin/env python3
"""Generate a query-only transformation artifact for offline ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from storybook import adaptive, config, llm  # noqa: E402
from storybook.eval import benchmark as benchmark_module  # noqa: E402
from storybook.eval import runner as runner_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate query-only rewrite/multi-query/HyDE evidence"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark")
    parser.add_argument(
        "--variant",
        action="append",
        choices=["exact", "synonym", "cross_language", "cross_tool", "ambiguous"],
        help="Variant to generate; repeatable (default: ambiguous)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--model", help="Override STORYBOOK_LLM_MODEL")
    args = parser.parse_args()

    if args.model:
        config.LLM_MODEL = args.model
    variants = set(args.variant or ["ambiguous"])
    benchmark = benchmark_module.load_benchmark(args.benchmark)
    pairs = [
        pair for pair in runner_module._strategy_query_pairs(benchmark)
        if pair["variant"] in variants
    ]
    entries = []
    seen = set()
    for index, pair in enumerate(pairs, start=1):
        query = adaptive.normalize_query(pair["query"])
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        started = time.perf_counter()
        output = llm.transform_search_query(
            query,
            list(adaptive.VALID_TRANSFORMS),
            timeout_seconds=max(0.1, args.timeout),
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        entries.append({
            "query": query,
            "query_sha256": digest,
            "variant": pair["variant"],
            "elapsed_ms": elapsed_ms,
            "output": output,
        })
        status = "ok" if output else "unavailable"
        print(f"[{index}/{len(pairs)}] {pair['variant']} {status} {elapsed_ms:.1f}ms")

    artifact = {
        "schema_version": 1,
        "source": "query_only_pre_generated",
        "generator": config.LLM_MODEL,
        "prompt_version": "transform_search_query-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_inputs": [
            "raw_query", "requested_transformations", "timeout_seconds"
        ],
        "ground_truth_fields_used_for_generation": [],
        "requested_transformations": list(adaptive.VALID_TRANSFORMS),
        "variants": sorted(variants),
        "entries": entries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(entries)} query-only entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
