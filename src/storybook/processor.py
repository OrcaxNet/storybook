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

logger = logging.getLogger(__name__)


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

    # ── Step 1: LLM 提取关键词和摘要 ──
    keywords = llm.extract_keywords(raw_content)
    if not keywords:
        # fallback: 用 problem_desc 做关键词
        keywords = session["problem_desc"].split()[:5]
    logger.info("  关键词: %s", keywords)

    # ── Step 2: 生成语义向量 ──
    # 用 关键词+问题描述 拼接做 embedding（比纯 raw_content 更聚焦）
    embed_text = " ".join(keywords) + " " + (session["problem_desc"] or "")
    query_vec = embeddings.embed(embed_text)
    if not query_vec:
        logger.error("  向量生成失败，跳过")
        store.update_session_status(session_id, "failed")
        return None

    # ── Step 3: 记忆检索 — 找相似 story ──
    matches = store.search_by_vector(query_vec, top_k=config.TOP_K_RETRIEVAL)
    high_matches = [m for m in matches if m["similarity"] >= config.SIM_THRESHOLD_HIGH]
    low_matches = [m for m in matches if config.SIM_THRESHOLD_LOW <= m["similarity"] < config.SIM_THRESHOLD_HIGH]

    logger.info("  检索结果: %d 高匹配, %d 低匹配, %d 总计",
                len(high_matches), len(low_matches), len(matches))

    # ── Step 4: 记忆处理（三分支） ──
    story_id = None

    if high_matches:
        # 高匹配 → 合并或更新
        best = high_matches[0]
        story_id = _handle_merge_or_update(session, best, keywords, query_vec, matches)
    else:
        # 无匹配/低匹配 → 新建
        story_id = _handle_create(session, keywords, query_vec, low_matches)

    # ── Step 5: 标记会话已处理 ──
    store.update_session_status(session_id, "processed")
    logger.info("✅ 会话 #%d 处理完成 → story #%d", session_id, story_id)

    return story_id


def _handle_create(session, keywords: list[str], query_vec: list[float],
                   low_matches: list[dict]) -> int:
    """新建 story 分支"""
    # LLM 浓缩为标准 story
    summary = llm.summarize_session(session["raw_content"])
    title = summary["title"]
    content = summary["content"]

    # 存入新 story
    story_id = store.add_story(
        title=title,
        content=content,
        keywords=keywords,
        embedding=query_vec,
        source_session_ids=[session["id"]],
    )

    # 与低匹配 story 建立弱关联边
    for match in low_matches:
        weight = match["similarity"]  # 用相似度作为初始权重
        store.add_or_update_edge(story_id, match["story_id"], weight, "semantic")
        logger.info("  建立关联: story#%d ↔ story#%d (weight=%.3f)",
                     story_id, match["story_id"], weight)

    return story_id


def _handle_merge_or_update(session, best_match: dict, keywords: list[str],
                            query_vec: list[float], all_matches: list[dict]) -> int:
    """合并或更新分支"""
    old_story = store.get_story(best_match["story_id"])
    if not old_story:
        return _handle_create(session, keywords, query_vec, all_matches)

    # 判断是「仅补充细节」还是「需要合并内容」
    # 如果新会话内容与旧 story 高度相似(≥UPDATE_ONLY)，仅更新关键词和向量
    if best_match["similarity"] >= config.SIM_THRESHOLD_UPDATE_ONLY:
        logger.info("  → 更新分支 (sim=%.3f): 仅补充细节", best_match["similarity"])
        return _update_existing(old_story, session, keywords, query_vec, all_matches)
    else:
        logger.info("  → 合并分支 (sim=%.3f): 合并内容", best_match["similarity"])
        return _merge_into_existing(old_story, session, keywords, query_vec, all_matches)


def _update_existing(old_story: dict, session, keywords: list[str],
                     query_vec: list[float], all_matches: list[dict]) -> int:
    """仅补充细节：合并关键词、更新向量、强化关联权重"""
    # 合并关键词（去重）
    merged_kw = list(set(old_story.get("keywords", []) + keywords))

    # 用旧 story 内容重新生成 embedding（保留原有语义 + 新关键词影响）
    embed_text = " ".join(merged_kw) + " " + old_story["content"]
    new_vec = embeddings.embed(embed_text) or query_vec

    store.update_story(
        old_story["id"],
        keywords=merged_kw,
        embedding=new_vec,
    )

    # 更新 source_session_ids
    old_sources = old_story.get("source_session_ids", [])
    if session["id"] not in old_sources:
        old_sources.append(session["id"])
        store.update_story_raw_sessions(old_story["id"], old_sources)

    # 强化关联边权重
    for match in all_matches:
        if match["story_id"] != old_story["id"]:
            store.increment_edge_weight(old_story["id"], match["story_id"])

    return old_story["id"]


def _merge_into_existing(old_story: dict, session, keywords: list[str],
                         query_vec: list[float], all_matches: list[dict]) -> int:
    """合并内容到旧 story，必要时分裂"""
    # LLM 合并旧 story + 新会话
    # 从 raw_content 提取新内容摘要
    new_summary = llm.summarize_session(session["raw_content"])
    merged = llm.merge_stories(old_story["content"], new_summary["content"])

    merged_text = merged["content"]

    # 判断是否需要分裂
    if llm.judge_split(merged_text):
        logger.info("  → 触发分裂！拆分子 story")
        return _split_and_store(old_story, session, merged_text, keywords, query_vec, all_matches)
    else:
        # 不分裂：直接更新旧 story
        merged_kw = list(set(old_story.get("keywords", []) + keywords))
        embed_text = " ".join(merged_kw) + " " + merged_text
        new_vec = embeddings.embed(embed_text) or query_vec

        # 更新 source_session_ids
        old_sources = old_story.get("source_session_ids", [])
        if session["id"] not in old_sources:
            old_sources.append(session["id"])

        store.update_story(
            old_story["id"],
            title=merged["title"],
            content=merged_text,
            keywords=merged_kw,
            embedding=new_vec,
        )

        # 如果有 source_session_ids 更新
        if session["id"] not in old_story.get("source_session_ids", []):
            store.update_story_raw_sessions(old_story["id"], old_sources)

        # 强化关联边权重
        for match in all_matches:
            if match["story_id"] != old_story["id"]:
                store.increment_edge_weight(old_story["id"], match["story_id"])

        return old_story["id"]


def _split_and_store(old_story: dict, session, merged_text: str,
                     keywords: list[str], query_vec: list[float],
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

    for sub in sub_stories:
        # 为每个子 story 生成 embedding
        sub_embed_text = " ".join(merged_kw) + " " + sub["content"]
        sub_vec = embeddings.embed(sub_embed_text) or query_vec

        child_id = store.add_story(
            title=sub["title"],
            content=sub["content"],
            keywords=merged_kw,
            embedding=sub_vec,
            parent_id=parent_id,
            source_session_ids=[session["id"]],
        )
        child_ids.append(child_id)

        # 父子边 weight=1.0
        store.add_or_update_edge(parent_id, child_id, config.WEIGHT_PARENT_CHILD, "parent_child")

    # 子 story 之间建立语义关联
    for i in range(len(child_ids)):
        for j in range(i + 1, len(child_ids)):
            store.add_or_update_edge(child_ids[i], child_ids[j], 0.5, "sibling")

    # 父 story 已拆分为子 story：从检索索引移除父向量，使其不再命中搜索
    # （保留 stories 行用于 parent_id 谱系引用）
    store.delete_story_vector(parent_id)

    # 更新旧 story 的 source_session_ids
    old_sources = old_story.get("source_session_ids", [])
    if session["id"] not in old_sources:
        old_sources.append(session["id"])
        store.update_story_raw_sessions(parent_id, old_sources)

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
