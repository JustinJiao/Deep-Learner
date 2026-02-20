from __future__ import annotations

from typing import Any, Dict, List


def clip_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""

    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} chars)"


def preview_messages(
    messages: List[Dict[str, Any]],
    max_items: int = 4,
    text_limit: int = 120,
) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    for msg in messages[-max_items:]:
        previews.append(
            {
                "role": msg.get("role", ""),
                "content_preview": clip_text(msg.get("content", ""), text_limit),
            }
        )
    return previews


def preview_turns(
    turns: List[Dict[str, Any]],
    max_items: int = 3,
    text_limit: int = 120,
) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    for turn in turns[:max_items]:
        previews.append(
            {
                "query_preview": clip_text(turn.get("query", ""), text_limit),
                "response_preview": clip_text(turn.get("response", ""), text_limit),
            }
        )
    return previews


def preview_docs(
    docs: List[Dict[str, Any]],
    max_items: int = 4,
    text_limit: int = 120,
) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    for doc in docs[:max_items]:
        score = doc.get("score")
        if isinstance(score, (int, float)):
            score = round(float(score), 4)

        previews.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "score": score,
                "content_preview": clip_text(doc.get("content", ""), text_limit),
            }
        )
    return previews


def preview_citations(
    citations: List[Dict[str, Any]],
    max_items: int = 4,
    text_limit: int = 120,
) -> List[Dict[str, Any]]:
    previews: List[Dict[str, Any]] = []
    for c in citations[:max_items]:
        score = c.get("score")
        if isinstance(score, (int, float)):
            score = round(float(score), 4)

        previews.append(
            {
                "id": c.get("id"),
                "title": c.get("title"),
                "score": score,
                "quote_preview": clip_text(c.get("quote", ""), text_limit),
            }
        )
    return previews
