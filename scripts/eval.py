#!/usr/bin/env python3
"""独立评测入口：``python scripts/eval.py [all|retrieval|processing|split|ablation|strategy]``

等价于 ``storybook eval``，便于在未做 editable 安装时直接运行
（自动把 src/ 加入 sys.path）。需要 Ollama 运行 embedding。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from storybook import eval as eval_module  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Storybook 检索质量评测")
    parser.add_argument("part", nargs="?", default="all",
                        choices=[
                            "all", "retrieval", "processing", "split",
                            "ablation", "strategy",
                        ])
    parser.add_argument("--report", "-r", help="把完整 JSON 报告写入该路径")
    parser.add_argument("--benchmark", help="自定义 benchmark JSON 路径")
    parser.add_argument(
        "--transform-cache",
        help="query-only 预生成 transformation JSON；仅用于可复现质量证据",
    )
    args = parser.parse_args()

    parts = (
        "retrieval", "processing", "split", "ablation", "strategy"
    ) if args.part == "all" else (args.part,)
    print(f"📐 运行评测: {', '.join(parts)}（embedding 走真实 Ollama）\n")

    transform_provider = None
    transform_source = "live_generated"
    if args.transform_cache:
        transform_provider = eval_module.pre_generated_transform_provider(
            args.transform_cache
        )
        transform_source = "query_only_pre_generated"
    rep = eval_module.run_all(
        parts=parts,
        benchmark_path=args.benchmark,
        transform_provider=transform_provider,
        transform_source=transform_source,
    )
    rep.meta["embed_mode"] = "ollama"
    rep.meta["part"] = args.part
    print(eval_module.format_report(rep))

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📝 JSON 报告已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
