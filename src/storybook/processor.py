"""
记忆加工模块 — 核心「做梦」流程

三分支逻辑：
  1. 新建：无匹配/低匹配 → 浓缩为 story 存入
  2. 合并：高匹配 → 合并到旧 story，必要时分裂
  3. 更新：仅补充细节 → 更新关键词/向量/权重
"""
import json
import logging
from typing import Optional

from . import config
from . import store
from . import embeddings
from . import llm
from . import story_v2

logger = logging.getLogger(__name__)


def _active_representation() -> str:
    state = store.get_embedding_index_state()
    return state.get("active_representation") or config.EMBED_REPRESENTATION


def process_session(session_id: int) -> Optional[int]:
    """
    处理单条会话：提取 → 检索 → 加工 → 存储
    返回新建/更新的 story_id，失败返回 None
    """
    session = store.get_session(session_id)
    if not session:
        logger.error("会话 #%d 不存在", session_id)
        return None

    if session["status"] == "processed":
        logger.info("会话 #%d 已处理过，跳过", session_id)
        return None

    raw_content = session["raw_content"]
    logger.info("🔄 开始处理会话 #%d: %s", session_id, session["problem_desc"][:50])

    # ── Step 1: LLM 提取关键词，并按独立结论形成 Story v2 ──
    keywords = llm.extract_keywords(raw_content)
    if not keywords:
        # fallback: 用 problem_desc 做关键词
        keywords = session["problem_desc"].split()[:5]
    logger.info("  关键词: %s", keywords)

    candidates = llm.form_stories(raw_content)
    candidates = [
        story_v2.normalize_story_payload(
            candidate,
            fallback_content=raw_content,
            source_session_ids=[session["id"]],
        )
        for candidate in candidates
    ]
    if not candidates:
        candidates = [story_v2.normalize_story_payload(
            {}, fallback_content=raw_content, source_session_ids=[session["id"]]
        )]

    # A legacy focused vector is retained only as the single-candidate
    # consolidation lookup. Persisted v2 vectors use the documented default
    # representation. Multi-candidate sessions use each candidate's own vector
    # so independent conclusions cannot collapse into one old Story.
    embed_text = " ".join(keywords) + " " + (session["problem_desc"] or "")
    lookup_vec = embeddings.embed(embed_text)
    if not lookup_vec:
        logger.error("  向量生成失败，跳过")
        store.update_session_status(session_id, "failed")
        return None

    prepared = []
    for candidate in candidates:
        candidate_keywords = list(dict.fromkeys(
            keywords + candidate.get("keywords", [])
        ))
        if candidate.get("_v2_explicit"):
            story_vec = embeddings.embed(story_v2.embedding_input(
                {**candidate, "keywords": candidate_keywords},
                _active_representation(),
            ))
            if not story_vec:
                logger.error("  Story v2 向量生成失败，跳过整条 Session")
                store.update_session_status(session_id, "failed")
                return None
        else:
            # Compatibility for deterministic legacy test/adaptor summaries.
            story_vec = lookup_vec
        prepared.append((candidate, candidate_keywords, story_vec))

    # ── Step 2: 逐独立 Story 检索并执行 create/merge/update ──
    story_ids = []
    for candidate, candidate_keywords, story_vec in prepared:
        candidate_lookup = story_vec if len(prepared) > 1 else lookup_vec
        matches = store.search_by_vector(
            candidate_lookup, top_k=config.TOP_K_RETRIEVAL
        )
        matches = [
            match for match in matches
            if match["story_id"] not in story_ids
        ]
        high_matches = [
            match for match in matches
            if match["similarity"] >= config.SIM_THRESHOLD_HIGH
        ]
        low_matches = [
            match for match in matches
            if config.SIM_THRESHOLD_LOW <= match["similarity"] < config.SIM_THRESHOLD_HIGH
        ]
        if high_matches:
            story_id = _handle_merge_or_update(
                session, high_matches[0], candidate_keywords, story_vec,
                matches, candidate,
            )
        else:
            story_id = _handle_create(
                session, candidate_keywords, story_vec, low_matches, candidate
            )
        if story_id is not None:
            story_ids.append(story_id)

    if not story_ids:
        store.update_session_status(session_id, "failed")
        return None

    # ── Step 3: 标记会话已处理 ──
    store.update_session_status(session_id, "processed")
    logger.info("✅ 会话 #%d 处理完成 → stories %s", session_id, story_ids)

    return story_ids[0]


def _handle_create(session, keywords: list[str], story_vec: list[float],
                   low_matches: list[dict], candidate: dict) -> int:
    """新建 story 分支"""
    story_id = store.add_story(
        title=candidate["title"],
        abstract=candidate["abstract"],
        content=candidate["content"],
        detail=candidate["detail"],
        sources=candidate["sources"],
        applicability=candidate["applicability"],
        keywords=keywords,
        embedding=story_vec,
        source_session_ids=[session["id"]],
    )

    # 与低匹配 story 建立弱关联边
    for match in low_matches:
        weight = match["similarity"]  # 用相似度作为初始权重
        store.add_or_update_edge(
            story_id,
            match["story_id"],
            weight,
            "semantic",
            provenance={
                "source": "memory_formation",
                "method": "embedding_similarity",
            },
        )
        logger.info("  建立关联: story#%d ↔ story#%d (weight=%.3f)",
                     story_id, match["story_id"], weight)

    return story_id


def _handle_merge_or_update(session, best_match: dict, keywords: list[str],
                            story_vec: list[float], all_matches: list[dict],
                            candidate: dict) -> int:
    """合并或更新分支"""
    old_story = store.get_story(best_match["story_id"])
    if not old_story:
        return _handle_create(session, keywords, story_vec, all_matches, candidate)

    # 判断是「仅补充细节」还是「需要合并内容」
    # 如果新会话内容与旧 story 高度相似(≥UPDATE_ONLY)，仅更新关键词和向量
    if best_match["similarity"] >= config.SIM_THRESHOLD_UPDATE_ONLY:
        logger.info("  → 更新分支 (sim=%.3f): 仅补充细节", best_match["similarity"])
        return _update_existing(old_story, session, keywords, story_vec, all_matches)
    else:
        logger.info("  → 合并分支 (sim=%.3f): 合并内容", best_match["similarity"])
        return _merge_into_existing(
            old_story, session, keywords, story_vec, all_matches, candidate
        )


def _update_existing(old_story: dict, session, keywords: list[str],
                     story_vec: list[float], all_matches: list[dict]) -> int:
    """仅补充细节：合并关键词、更新向量、强化关联权重"""
    # 合并关键词（去重）
    merged_kw = list(set(old_story.get("keywords", []) + keywords))

    new_vec = embeddings.embed(story_v2.embedding_input(
        {**old_story, "keywords": merged_kw}, _active_representation()
    )) or story_vec

    old_sources = old_story.get("source_session_ids", [])
    if session["id"] not in old_sources:
        old_sources.append(session["id"])

    store.update_story(
        old_story["id"],
        keywords=merged_kw,
        embedding=new_vec,
        sources=old_story.get("sources", []),
        source_session_ids=old_sources,
        event_type="update",
    )

    # 强化关联边权重
    for match in all_matches:
        if match["story_id"] != old_story["id"]:
            store.increment_edge_weight(old_story["id"], match["story_id"])

    return old_story["id"]


def _dedupe_values(*groups):
    result = []
    seen = set()
    for group in groups:
        for value in group or []:
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(value)
    return result


def _merged_payload(
    old_story: dict,
    candidate: dict,
    merged: dict,
    *,
    source_session_ids: list[int],
) -> dict:
    """Merge structured evidence without losing either Story's detail."""

    if not candidate.get("_v2_explicit"):
        return story_v2.normalize_story_payload(
            {
                "title": merged.get("title"),
                "content": merged.get("content", ""),
                "sources": _dedupe_values(
                    old_story.get("sources"), candidate.get("sources")
                ),
                "applicability": old_story.get("applicability", {}),
            },
            fallback_content=merged.get("content", ""),
            source_session_ids=source_session_ids,
        )

    old_detail = story_v2.normalize_detail(
        old_story.get("detail"), legacy_content=old_story.get("content", "")
    )
    new_detail = story_v2.normalize_detail(
        candidate.get("detail"), legacy_content=candidate.get("content", "")
    )
    old_app = old_detail["applicability"]
    new_app = new_detail["applicability"]
    applicability = {
        "applies_when": _dedupe_values(
            old_app.get("applies_when"), new_app.get("applies_when")
        ),
        "excludes_when": _dedupe_values(
            old_app.get("excludes_when"), new_app.get("excludes_when")
        ),
    }
    payload = {
        "title": merged.get("title") or old_story.get("title"),
        "abstract": "；".join(filter(None, (
            old_story.get("abstract"), candidate.get("abstract")
        ))),
        "detail": {
            "problem": "\n\n".join(filter(None, (
                old_detail.get("problem"), new_detail.get("problem")
            ))),
            "actions": _dedupe_values(
                old_detail.get("actions"), new_detail.get("actions")
            ),
            "outcome": "\n\n".join(filter(None, (
                old_detail.get("outcome"), new_detail.get("outcome")
            ))),
            "pitfalls": _dedupe_values(
                old_detail.get("pitfalls"), new_detail.get("pitfalls")
            ),
            "evidence": _dedupe_values(
                old_detail.get("evidence"), new_detail.get("evidence")
            ),
            "applicability": applicability,
        },
        "sources": _dedupe_values(
            old_story.get("sources"), candidate.get("sources")
        ),
        "applicability": applicability,
    }
    return story_v2.normalize_story_payload(
        payload,
        fallback_content=merged.get("content", ""),
        source_session_ids=source_session_ids,
    )


def _merge_into_existing(old_story: dict, session, keywords: list[str],
                         story_vec: list[float], all_matches: list[dict],
                         candidate: dict) -> int:
    """合并内容到旧 story，必要时分裂"""
    # LLM 合并旧 story + 新会话
    merged = llm.merge_stories(old_story["content"], candidate["content"])

    merged_text = merged["content"]

    # 判断是否需要分裂
    if llm.judge_split(merged_text):
        logger.info("  → 触发分裂！拆分子 story")
        return _split_and_store(
            old_story, session, merged_text, keywords, story_vec, all_matches
        )
    else:
        merged_kw = list(set(old_story.get("keywords", []) + keywords))
        old_sources = old_story.get("source_session_ids", [])
        if session["id"] not in old_sources:
            old_sources.append(session["id"])

        merged_payload = _merged_payload(
            old_story, candidate, merged, source_session_ids=old_sources
        )
        new_vec = embeddings.embed(
            story_v2.embedding_input(
                {**merged_payload, "keywords": merged_kw},
                _active_representation(),
            )
        ) or story_vec

        store.update_story(
            old_story["id"],
            title=merged_payload["title"],
            abstract=merged_payload["abstract"],
            content=merged_payload["content"],
            detail=merged_payload["detail"],
            sources=merged_payload["sources"],
            applicability=merged_payload["applicability"],
            source_session_ids=old_sources,
            keywords=merged_kw,
            embedding=new_vec,
            event_type="merge",
        )

        # 强化关联边权重
        for match in all_matches:
            if match["story_id"] != old_story["id"]:
                store.increment_edge_weight(old_story["id"], match["story_id"])

        return old_story["id"]


def _split_and_store(old_story: dict, session, merged_text: str,
                     keywords: list[str], story_vec: list[float],
                     all_matches: list[dict]) -> int:
    """分裂旧 story 为多个子 story"""
    sub_stories = llm.split_story(merged_text)

    # 把旧 story 标记为"已分裂"（保留但不再用于检索）
    # 实际上我们用 parent_id 来标识子 story，旧 story 保留不动

    # 合并父 story 关键词与当前会话关键词（去重保序）。
    # 否则父 story 的关键词（如中文"火山引擎 API"）会随分裂丢失，子 story 只剩触发会话的关键词，
    # 削弱跨语言/跨表述召回。
    parent_kw = old_story.get("keywords", []) or []
    merged_kw = list(dict.fromkeys(parent_kw + keywords))

    parent_id = old_story["id"]
    child_ids = []

    source_ids = list(dict.fromkeys(
        old_story.get("source_session_ids", []) + [session["id"]]
    ))
    for sub in sub_stories:
        payload = story_v2.normalize_story_payload(
            sub,
            fallback_content=sub.get("content", ""),
            source_session_ids=source_ids,
        )
        sub_vec = embeddings.embed(story_v2.embedding_input(
            {**payload, "keywords": merged_kw}, _active_representation()
        )) or story_vec

        child_id = store.add_story(
            title=payload["title"],
            abstract=payload["abstract"],
            content=payload["content"],
            detail=payload["detail"],
            sources=payload["sources"],
            applicability=payload["applicability"],
            keywords=merged_kw,
            embedding=sub_vec,
            parent_id=parent_id,
            source_session_ids=source_ids,
            event_type="split_child",
        )
        child_ids.append(child_id)

        # 父子边 weight=1.0
        store.add_or_update_edge(
            parent_id,
            child_id,
            config.WEIGHT_PARENT_CHILD,
            "parent_child",
            provenance={"source": "story_split", "method": "lineage"},
        )

    # 子 Story 共享一次 split 来源；用标准 semantic 边并在
    # provenance 中保留 sibling 语义，不再写 v0.1 的非标准类型。
    for i in range(len(child_ids)):
        for j in range(i + 1, len(child_ids)):
            store.add_or_update_edge(
                child_ids[i],
                child_ids[j],
                0.5,
                "semantic",
                provenance={
                    "source": "story_split",
                    "method": "shared_parent",
                    "relationship": "sibling",
                },
            )

    store.update_story(
        parent_id,
        sources=old_story.get("sources", []),
        source_session_ids=source_ids,
        embedding=old_story.get("embedding") or story_vec,
        event_type="split_source",
    )
    # 父 story 已拆分为子 story：从检索索引移除父向量，使其不再命中搜索
    # （保留 stories 行用于 parent_id 谱系引用）
    store.delete_story_vector(parent_id)

    logger.info("  分裂完成: %d 个子 story (parent=#%d)", len(child_ids), parent_id)
    return child_ids[0] if child_ids else parent_id


def process_all_pending(verbose: bool = True) -> dict:
    """处理所有 pending 状态的会话"""
    pending = store.get_pending_sessions()
    total = len(pending)
    success = 0
    failed = 0

    if verbose:
        print(f"\n🌙 开始「做梦」加工，共 {total} 条待处理会话\n")

    for i, session in enumerate(pending, 1):
        if verbose:
            print(f"  [{i}/{total}] 处理会话 #{session['id']}: {session['problem_desc'][:40]}...")

        try:
            result = process_session(session["id"])
            if result:
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("处理会话 #%d 异常: %s", session["id"], e)
            failed += 1
            store.update_session_status(session["id"], "failed")

    summary = {"total": total, "success": success, "failed": failed}

    if verbose:
        print(f"\n✅ 加工完成: {success} 成功, {failed} 失败, {total} 总计")
        stats = store.get_stats()
        print(f"   Story 库: {stats['stories']} 条记忆, {stats['edges']} 条关联")

    return summary


def run_dream_cycle():
    """定时触发入口：处理所有 pending 会话"""
    logging.info("🌙 梦境周期启动")
    return process_all_pending(verbose=False)
