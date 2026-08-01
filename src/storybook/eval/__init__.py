"""检索质量评测体系（benchmark + recall@k + 合并正确率 + 分裂质量）。

公开接口（``storybook.eval``）::

    from storybook.eval import run_retrieval_eval, run_processing_eval, run_all, load_benchmark

三个子评测：
  * **retrieval** —— 真实 embedding + 人工标注 story 语料，度量 recall@k / precision@k / MRR，
    以及 ``SIM_THRESHOLD_SEARCH`` 阈值敏感性曲线。对应验收点「重复 bug 检索准确率≥70%」。
  * **processing** —— 真实 embedding + 确定性 LLM 桩（人工关键词/摘要），度量 merge/update
    分支是否选对（create vs merge_or_update），以及 ``SIM_THRESHOLD_HIGH`` 阈值敏感性。
  * **split** —— 真实 embedding + 确定性 LLM 桩，度量分裂路径结构正确性（父向量移除、
    父子/兄弟边、子 story 可检索）。

数据集见 ``data/retrieval_benchmark.json``（人工 ground truth，可复现）。
"""
from .benchmark import load_benchmark, BENCHMARK_PATH, Topic, MergePair, SplitCase
from .metrics import (
    recall_at_k, precision_at_k, mrr, merge_branch_accuracy, threshold_sweep,
)
from .runner import (
    run_retrieval_eval, run_processing_eval, run_split_eval,
    run_embedding_ablation, run_all,
    format_report, EvalReport,
)

__all__ = [
    "load_benchmark", "BENCHMARK_PATH", "Topic", "MergePair", "SplitCase",
    "recall_at_k", "precision_at_k", "mrr", "merge_branch_accuracy", "threshold_sweep",
    "run_retrieval_eval", "run_processing_eval", "run_split_eval",
    "run_embedding_ablation", "run_all",
    "format_report", "EvalReport",
]
