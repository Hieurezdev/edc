"""Generate conservative Vietnamese text edits for corpus review."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from openai import OpenAI


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt_templates" / "vi_text_generator.txt"
logger = logging.getLogger(__name__)
_JSON_FENCE = re.compile(r"\A\s*```(?:json)?\s*\n(.*?)\n```\s*\Z", re.IGNORECASE | re.DOTALL)
_REQUIRED_KEYS = {"action", "text", "answer", "explanation", "reason"}
_ACTIONS = {"keep", "rewrite", "drop"}
_CANDIDATE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "text_candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["keep", "rewrite", "drop"]},
                "text": {"type": "string"},
                "answer": {"type": ["string", "null"]},
                "explanation": {"type": ["string", "null"], "maxLength": 300},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["action", "text", "answer", "explanation", "reason"],
            "additionalProperties": False,
        },
    },
}


class RetryFeedback(TypedDict):
    attempt: int
    candidate: dict[str, object]
    failed_checks: list[dict[str, object]]


class GeneratorTruncatedError(ValueError):
    """The server exhausted its output budget before completing a proposal."""


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _parse_candidate(content: str, source_text: str) -> dict[str, object]:
    match = _JSON_FENCE.fullmatch(content)
    payload = match.group(1) if match else content.strip()
    try:
        candidate = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generator response is not valid JSON at character {exc.pos}; "
            f"response length={len(payload)}; decoder error={exc.msg}"
        ) from exc

    if type(candidate) is not dict:
        raise ValueError(f"Generator response must be an object, got {type(candidate).__name__}")
    if set(candidate) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(candidate))
        unexpected_count = len(set(candidate) - _REQUIRED_KEYS)
        raise ValueError(
            "Generator response must contain exactly action, text, answer, explanation, and reason; "
            f"missing fields: {missing}; unexpected field count: {unexpected_count}. "
            "Verify that the inference server supports response_format=json_schema."
        )

    action = candidate["action"]
    text = candidate["text"]
    answer = candidate["answer"]
    explanation = candidate["explanation"]
    reason = candidate["reason"]
    if type(action) is not str or action not in _ACTIONS or type(text) is not str or type(reason) is not str or not reason.strip():
        raise ValueError("Generator response has invalid action, text, or reason")
    if answer is not None and (type(answer) is not str or not answer.strip() or answer != answer.strip()):
        raise ValueError("Generator answer must be null or a nonempty single-line string")
    if isinstance(answer, str) and ("\n" in answer or "\r" in answer):
        raise ValueError("Generator answer must fit on one line")
    if answer is None and explanation is not None:
        raise ValueError("Generator explanation requires an answer")
    if answer is not None and (
        type(explanation) is not str or not explanation.strip()
        or explanation != explanation.strip() or len(explanation) > 300
        or "\n" in explanation or "\r" in explanation
    ):
        raise ValueError("Generator explanation must be a short, nonempty single-line string")

    if action == "keep" and (text != source_text or answer is not None):
        # An edited proposal must undergo rewrite checks, never keep shortcuts.
        action = "rewrite"
        candidate["action"] = action
        logger.warning("generator_action_corrected", extra={"details": {"reported_action": "keep", "effective_action": action}})
    if action == "drop" and (text != "" or answer is not None):
        raise ValueError("A drop action must have empty text and no answer")
    if action == "rewrite" and not text.strip():
        raise ValueError("A rewrite action must contain nonempty text")
    if answer is not None:
        if action != "rewrite":
            raise ValueError("A new answer requires a rewrite action")
        lines = text.rstrip().splitlines()
        if lines and lines[-1].startswith("Giải thích:"):
            lines.pop()
        if lines and lines[-1].startswith("Đáp án:"):
            if lines[-1].removeprefix("Đáp án:").strip().rstrip(".") != answer:
                raise ValueError("Generator text has a conflicting answer")
            lines.pop()
        # Earlier questions may have their own answer blocks. Collapse only an
        # identical adjacent suffix, leaving those question-answer relations intact.
        expected_pair = [f"Đáp án: {answer}", f"Giải thích: {explanation}"]
        while len(lines) >= 2 and lines[-2:] == expected_pair:
            del lines[-2:]
        body = "\n".join(lines).rstrip()
        if not body:
            raise ValueError("Generator answer requires question text")
        candidate["text"] = body + f"\nĐáp án: {answer}\nGiải thích: {explanation}"
    if action == "rewrite" and candidate["text"] == source_text:
        # Classify the final text after answer formatting; existing answer lines
        # may already match the source and must not be reported as newly added.
        candidate["action"] = "keep"
        candidate["answer"] = None
        candidate["explanation"] = None
        logger.warning("generator_action_corrected", extra={"details": {"reported_action": "rewrite", "effective_action": "keep"}})

    return candidate


def generate_candidate(
    client: OpenAI,
    model: str,
    source_text: str,
    visolex_suggestions: list[dict[str, object]],
    rules: list[str],
    retry_feedback: list[RetryFeedback] | None = None,
    max_tokens: int = 8192,
    enable_thinking: bool | None = None,
) -> dict[str, object]:
    """Request an edit using rules and rejected attempts from the current chunk."""
    if not model.strip():
        raise ValueError("A model name is required for text generation")
    if not source_text.strip():
        raise ValueError("Source text must be nonempty for text generation")
    if max_tokens < 1:
        raise ValueError("Generator max_tokens must be positive")

    request = {
        "source_text": source_text,
        "visolex_suggestions": visolex_suggestions,
        "rules": rules,
        "retry_feedback": retry_feedback if retry_feedback is not None else [],
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}} if enable_thinking is not None else None,
        max_tokens=max_tokens,
        response_format=_CANDIDATE_FORMAT,
    )
    if not completion.choices or not isinstance(completion.choices[0].message.content, str):
        raise ValueError("Generator returned no text response")
    finish_reason = getattr(completion.choices[0], "finish_reason", None)
    if finish_reason == "length":
        raise GeneratorTruncatedError(
            f"Generator response was truncated (finish_reason=length, max_tokens={max_tokens}). "
            "Increase --generation-max-tokens or reduce chunk size using a fresh state directory. "
            "Also check the inference server output and context limits."
        )
    return _parse_candidate(completion.choices[0].message.content, source_text)
