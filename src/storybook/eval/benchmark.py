"""加载并结构化 ``data/retrieval_benchmark.json`` 评测基准（人工 ground truth）。

benchmark 的 story 索引向量复刻 ``processor.process_session`` 的 create 分支：
``embedding = embed(" ".join(keywords) + " " + problem_desc)``，query 向量复刻
``search.search``：``embedding = embed(query)``。这样评测度量的是真实检索语义，
而非一个孤立的余弦玩具。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "retrieval_benchmark.json"


@dataclass
class Topic:
    """一条 ground-truth story 主题 + 其三种查询变体。"""
    id: str
    domain: str
    title: str
    problem_desc: str
    keywords: list[str]
    content: str
    queries: dict[str, str]   # {exact, synonym, cross_lang}

    def index_text(self) -> str:
        """复刻 processor create 分支的 embed 输入文本。"""
        return " ".join(self.keywords) + " " + (self.problem_desc or "")


@dataclass
class SessionSpec:
    """merge/split 评测中一条会话的人工标注（替代 LLM 输出，保证可复现）。"""
    problem_desc: str
    keywords: list[str]
    summary: dict   # {title, content}

    def index_text(self) -> str:
        return " ".join(self.keywords) + " " + (self.problem_desc or "")


@dataclass
class MergePair:
    """一对会话（a 先建 story，b 再加工），标注期望分支与是否并入同一 story。"""
    id: str
    relation: str               # duplicate | near_identical | distinct
    expected_branch: str        # merge_or_update | update | create
    expected_same_story: bool
    note: str
    a: SessionSpec
    b: SessionSpec


@dataclass
class SplitCase:
    """一条应触发分裂的合并场景：existing 先建 story，incoming 合并后应分裂。"""
    id: str
    relation: str
    expected_split: bool
    note: str
    existing: SessionSpec
    incoming: SessionSpec


@dataclass
class Benchmark:
    version: int
    description: str
    topics: list[Topic] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    merge_pairs: list[MergePair] = field(default_factory=list)
    split_cases: list[SplitCase] = field(default_factory=list)

    @property
    def query_pairs(self) -> list[dict[str, Any]]:
        """展开为 (query, variant, topic_id) 三元组列表，便于遍历统计。"""
        pairs = []
        for t in self.topics:
            for variant, q in t.queries.items():
                pairs.append({"query": q, "variant": variant, "topic_id": t.id})
        return pairs


def _session(d: dict) -> SessionSpec:
    return SessionSpec(
        problem_desc=d["problem_desc"],
        keywords=list(d["keywords"]),
        summary=dict(d["summary"]),
    )


def load_benchmark(path: Path | str = None) -> Benchmark:
    """加载 benchmark JSON 为结构化对象。路径缺省取 ``BENCHMARK_PATH``。"""
    path = Path(path) if path else BENCHMARK_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Benchmark(
        version=raw["version"],
        description=raw.get("description", ""),
        topics=[
            Topic(
                id=t["id"], domain=t["domain"], title=t["title"],
                problem_desc=t["problem_desc"], keywords=list(t["keywords"]),
                content=t["content"], queries=dict(t["queries"]),
            )
            for t in raw["topics"]
        ],
        negatives=list(raw.get("negatives", [])),
        merge_pairs=[
            MergePair(
                id=p["id"], relation=p["relation"],
                expected_branch=p["expected_branch"],
                expected_same_story=p["expected_branch"] != "create",
                note=p.get("note", ""),
                a=_session(p["a"]), b=_session(p["b"]),
            )
            for p in raw.get("merge_pairs", [])
        ],
        split_cases=[
            SplitCase(
                id=c["id"], relation=c["relation"],
                expected_split=c["expected_split"], note=c.get("note", ""),
                existing=_session(c["existing"]), incoming=_session(c["incoming"]),
            )
            for c in raw.get("split_cases", [])
        ],
    )
