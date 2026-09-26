"""Independent, prompt-based checks for Vietnamese book text revisions."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


logger = logging.getLogger(__name__)


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"
_REVIEW_PROMPTS = {
    "semantics": "vi_review_semantics.txt",
    "terminology": "vi_review_terminology.txt",
    "solution": "vi_review_solution.txt",
    "removal": "vi_review_removal.txt",
}
_ACTIONS = {"keep", "rewrite", "drop"}
_VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
            "required": ["passed", "feedback"],
            "additionalProperties": False,
        },
    },
}


def _parse_verdict(content: str, check: str) -> dict[str, object]:
    try:
        verdict = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{check} reviewer returned invalid JSON at character {exc.pos}") from exc
    if not isinstance(verdict, dict):
        raise ValueError(f"{check} reviewer must return an object, got {type(verdict).__name__}")
    if type(verdict.get("passed")) is not bool or not isinstance(verdict.get("feedback"), str):
        field_types = {key: type(value).__name__ for key, value in verdict.items()}
        raise ValueError(
            f"{check} reviewer requires passed:boolean and feedback:string; received {field_types}. "
            "Verify that the inference server supports response_format=json_schema."
        )
    if not verdict["passed"] and not verdict["feedback"].strip():
        raise ValueError(f"{check} reviewer rejected the candidate without explaining the failure")
    return {"passed": verdict["passed"], "feedback": verdict["feedback"]}


def review_candidate(
    client: OpenAI, model: str, source_text: str, candidate: dict[str, object]
) -> list[dict[str, object]]:
    """Review one candidate through four separately prompted checks."""
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("Review source_text must be a nonempty string")
    if not isinstance(candidate, dict):
        raise ValueError("Review candidate must be an object")
    action = candidate.get("action")
    if not isinstance(action, str) or action not in _ACTIONS:
        raise ValueError("Review candidate.action must be keep, rewrite, or drop")
    if not isinstance(candidate.get("text"), str):
        raise ValueError("Review candidate.text must be a string")
    answer = candidate.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("Review candidate.answer must be a string or null")
    explanation = candidate.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise ValueError("Review candidate.explanation must be a string or null")
    if not isinstance(candidate.get("reason"), str):
        raise ValueError("Review candidate.reason must be a string")

    payload = json.dumps(
        {"source_text": source_text, "candidate": candidate}, ensure_ascii=False
    )
    results: list[dict[str, object]] = []
    for check, prompt_file in _REVIEW_PROMPTS.items():
        started = perf_counter()
        logger.info("reviewer_started", extra={"details": {"check": check, "model": model}})
        prompt = (_PROMPT_DIR / prompt_file).read_text(encoding="utf-8")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": payload},
            ],
            temperature=0,
            response_format=_VERDICT_FORMAT,
        )
        if not completion.choices:
            raise ValueError(f"{check} reviewer returned no choices")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{check} reviewer returned an empty response")
        verdict = _parse_verdict(content, check)
        logger.info("reviewer_completed", extra={"details": {"check": check, "passed": verdict["passed"], "elapsed_seconds": round(perf_counter() - started, 3)}})
        results.append(
            {"check": check, "passed": verdict["passed"], "feedback": verdict["feedback"]}
        )
    return results
