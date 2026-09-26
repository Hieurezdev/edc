"""Derive reusable writing rules from rejected edits and an accepted revision."""

from __future__ import annotations

from collections import Counter
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt_templates" / "vi_feedback_rules.txt"
_RETRY_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt_templates" / "vi_retry_rules.txt"
_CATEGORIES = {"spelling", "terminology", "math", "answer", "removal", "semantics"}
_RULE_TEXT_SCHEMA = {
    "type": "string", "minLength": 1, "maxLength": 250,
    "pattern": r"^\S(?:[^\r\n]*\S)?$",
}
_RULES_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "feedback_rules",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "enum": sorted(_CATEGORIES)},
                            "rule": _RULE_TEXT_SCHEMA,
                            "evidence": {"type": "string", "minLength": 1},
                        },
                        "required": ["category", "rule", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["rules"],
            "additionalProperties": False,
        },
    },
}
_ACTIONS = {"keep", "rewrite", "drop"}
_MATH = re.compile(r"\$\$.*?\$\$|(?<!\$)\$(?!\$)[^$\n]*\$(?!\$)", re.DOTALL)
_MATH_RULE = {
    "category": "math",
    "rule": "Khi sửa văn bản, giữ nguyên từng chuỗi công thức LaTeX $...$ hoặc $$...$$ từ nguồn.",
    "evidence": "Bản bị từ chối đã mất công thức LaTeX; bản được duyệt giữ nguyên công thức nguồn.",
}
_CATEGORIES_BY_CHECK = {
    "protected_math": {"math"},
    "semantics": {"semantics", "spelling"},
    "terminology": {"terminology", "spelling"},
    "solution": {"answer"},
    "removal": {"removal"},
}
_ANSWER_REMOVAL = re.compile(
    r"(?:\b(?:xóa|xoá|bỏ|loại|gỡ)\b.{0,24}\bđáp\s*án\b|"
    r"\bđáp\s*án\b.{0,24}\b(?:xóa|xoá|bỏ|loại|gỡ)\b)",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:không|tránh|đừng|chớ)\b", re.IGNORECASE)
_CHANGE_VERB = re.compile(r"\b(?:xóa|xoá|bỏ|loại|gỡ|thay|đổi|chuyển)\b", re.IGNORECASE)
_ANSWER_LABEL = re.compile(r"(?m)^Đáp án:\s*\S+", re.IGNORECASE)
_PROTECTED_CHANGE = re.compile(
    r"(?:\b(?:xóa|xoá|bỏ|loại|gỡ|thay|đổi|chuyển)\b.{0,28}"
    r"(?:\b(?:LaTeX|công\s*thức|từ\s*địa\s*phương|thuật\s*ngữ)\b|\$)|"
    r"(?:\b(?:LaTeX|công\s*thức|từ\s*địa\s*phương|thuật\s*ngữ)\b|\$).{0,28}"
    r"\b(?:xóa|xoá|bỏ|loại|gỡ|thay|đổi|chuyển)\b)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _retry_system_prompt() -> str:
    return _RETRY_PROMPT_PATH.read_text(encoding="utf-8")


def _candidate_fields(value: object, label: str) -> tuple[str, str, str | None]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    action, text, answer = value.get("action"), value.get("text"), value.get("answer")
    if not isinstance(action, str) or action not in _ACTIONS or not isinstance(text, str):
        raise ValueError(f"{label} must contain a valid action and text")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError(f"{label}.answer must be a nonempty string or null")
    return action, text, answer


def _validate_rejected_attempts(
    rejected_attempts: list[dict[str, object]],
    accepted_fields: tuple[str, str, str | None],
) -> list[dict[str, object]]:
    if not isinstance(rejected_attempts, list):
        raise ValueError("rejected_attempts must be a list")
    useful: list[dict[str, object]] = []
    for index, attempt in enumerate(rejected_attempts):
        if not isinstance(attempt, dict) or attempt.get("accepted") is not False:
            raise ValueError(f"rejected_attempts[{index}] must be a rejected attempt")
        candidate = attempt.get("candidate")
        fields = _candidate_fields(candidate, f"rejected_attempts[{index}].candidate")
        checks = attempt.get("checks")
        if not isinstance(checks, list):
            raise ValueError(f"rejected_attempts[{index}].checks must be a list")
        failed: list[dict[str, str]] = []
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get("check"), str) or type(check.get("passed")) is not bool or not isinstance(check.get("feedback"), str):
                raise ValueError(f"rejected_attempts[{index}] contains an invalid check")
            if check["passed"] is False and check["feedback"].strip():
                failed.append({"check": check["check"], "feedback": check["feedback"]})
        if failed and fields != accepted_fields:
            useful.append({"candidate": candidate, "failed_checks": failed})
    return useful


def _is_unsafe_rule(rule: str, accepted_answer_present: bool) -> bool:
    patterns = [_PROTECTED_CHANGE]
    if accepted_answer_present:
        patterns.append(_ANSWER_REMOVAL)
    for pattern in patterns:
        for match in pattern.finditer(rule):
            for verb in _CHANGE_VERB.finditer(rule, match.start(), match.end()):
                if not _NEGATION.search(rule[max(0, verb.start() - 14):verb.start()]):
                    return True
    return False


def _parse_rules(
    content: str, accepted_answer_present: bool,
    existing: set[tuple[str, str]], allowed_categories: set[str],
) -> list[dict[str, object]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Feedback-rule agent returned invalid JSON") from exc
    if type(payload) is not dict or set(payload) != {"rules"} or type(payload["rules"]) is not list:
        raise ValueError("Feedback-rule agent must return an object containing only a rules array")
    rules: list[dict[str, object]] = []
    seen = set(existing)
    for index, item in enumerate(payload["rules"]):
        if type(item) is not dict or set(item) != {"category", "rule", "evidence"}:
            raise ValueError(f"Feedback rule {index} must contain exactly category, rule, evidence")
        category, rule, evidence = item["category"], item["rule"], item["evidence"]
        if type(category) is not str or category not in _CATEGORIES or type(rule) is not str or type(evidence) is not str:
            raise ValueError(f"Feedback rule {index} has invalid field types or category")
        # Evidence is audit data, not a writing instruction; preserve it verbatim.
        if not evidence.strip():
            raise ValueError(f"Feedback rule {index}.evidence must contain nonempty evidence; length={len(evidence)}")
        if not rule.strip() or rule != rule.strip() or len(rule) > 250 or "\n" in rule or "\r" in rule:
            raise ValueError(
                f"Feedback rule {index}.rule must be 1-250 characters, trimmed, and single-line; "
                f"length={len(rule)}, blank={not rule.strip()}, "
                f"outer_whitespace={rule != rule.strip()}, multiline={chr(10) in rule or chr(13) in rule}. "
                "Verify that the inference server enforces the feedback_rules JSON schema."
            )
        if category not in allowed_categories or _is_unsafe_rule(rule, accepted_answer_present):
            continue
        identity = (category, rule.casefold())
        if identity not in seen:
            seen.add(identity)
            rules.append({"category": category, "rule": rule, "evidence": evidence})
    return rules


def derive_rules(
    client: OpenAI,
    model: str,
    source_text: str,
    rejected_attempts: list[dict[str, object]],
    accepted_candidate: dict[str, object],
    existing_rules: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Learn only general lessons supported by a rejected-to-accepted revision."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model name is required for feedback-rule generation")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a nonempty string")
    accepted_fields = _candidate_fields(accepted_candidate, "accepted_candidate")
    if not isinstance(existing_rules, list):
        raise ValueError("existing_rules must be a list")
    existing: set[tuple[str, str]] = set()
    for index, rule in enumerate(existing_rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("category"), str) or rule["category"] not in _CATEGORIES or not isinstance(rule.get("rule"), str):
            raise ValueError(f"existing_rules[{index}] has an invalid category or rule")
        existing.add((rule["category"], rule["rule"].casefold()))
    useful = _validate_rejected_attempts(rejected_attempts, accepted_fields)
    if not useful:
        return []
    allowed_categories = {
        category
        for attempt in useful
        for check in attempt["failed_checks"]
        for category in _CATEGORIES_BY_CHECK.get(check["check"], set())
    }
    if not allowed_categories:
        return []

    confirmed_rules: list[dict[str, object]] = []
    accepted_math = Counter(_MATH.findall(accepted_fields[1]))
    source_math = Counter(_MATH.findall(source_text))
    math_failure = any(
        check["check"] == "protected_math"
        and bool(source_math - Counter(_MATH.findall(attempt["candidate"]["text"])))
        for attempt in useful
        for check in attempt["failed_checks"]
    )
    math_rule_key = (_MATH_RULE["category"], _MATH_RULE["rule"].casefold())
    if math_failure and not (source_math - accepted_math) and math_rule_key not in existing:
        confirmed_rules.append(dict(_MATH_RULE))
        existing.add(math_rule_key)
    allowed_categories.discard("math")
    if not allowed_categories:
        return confirmed_rules

    request = {
        "source_text": source_text,
        "rejected_attempts": useful,
        "accepted_candidate": accepted_candidate,
        "existing_rules": existing_rules,
        "allowed_categories": sorted(allowed_categories),
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        temperature=0,
        response_format=_RULES_FORMAT,
    )
    if not completion.choices or not isinstance(completion.choices[0].message.content, str):
        raise ValueError("Feedback-rule agent returned no text response")
    answer_present = accepted_fields[2] is not None or bool(_ANSWER_LABEL.search(accepted_fields[1]))
    return confirmed_rules + _parse_rules(
        completion.choices[0].message.content, answer_present, existing, allowed_categories,
    )


def derive_retry_rules(
    client: OpenAI,
    model: str,
    source_text: str,
    rejected_candidate: dict[str, object],
    checks: list[dict[str, object]],
    existing_rules: list[dict[str, object]],
) -> list[str]:
    """Convert current failed checks into provisional rules for one retry."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model name is required for retry-rule generation")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must be a nonempty string")
    _, candidate_text, answer = _candidate_fields(rejected_candidate, "rejected_candidate")
    if not isinstance(checks, list):
        raise ValueError("checks must be a list")
    failed_checks: list[dict[str, str]] = []
    allowed_categories: set[str] = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or not isinstance(check.get("check"), str) or type(check.get("passed")) is not bool or not isinstance(check.get("feedback"), str):
            raise ValueError(f"checks[{index}] is invalid")
        if check["passed"] is False and check["feedback"].strip():
            failed_checks.append({"check": check["check"], "feedback": check["feedback"]})
            allowed_categories.update(_CATEGORIES_BY_CHECK.get(check["check"], set()))
    if not failed_checks:
        return []

    retry_rules: list[str] = []
    source_math = Counter(_MATH.findall(source_text))
    if any(check["check"] == "protected_math" for check in failed_checks) and source_math - Counter(_MATH.findall(candidate_text)):
        retry_rules.append(_MATH_RULE["rule"])
    allowed_categories.discard("math")
    if not allowed_categories:
        return retry_rules

    existing: set[tuple[str, str]] = set()
    for index, rule in enumerate(existing_rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("category"), str) or rule["category"] not in _CATEGORIES or not isinstance(rule.get("rule"), str):
            raise ValueError(f"existing_rules[{index}] has an invalid category or rule")
        existing.add((rule["category"], rule["rule"].casefold()))
    request = {
        "source_text": source_text,
        "rejected_candidate": rejected_candidate,
        "failed_checks": failed_checks,
        "allowed_categories": sorted(allowed_categories),
        "existing_rules": existing_rules,
    }
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _retry_system_prompt()},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ],
        temperature=0,
        response_format=_RULES_FORMAT,
    )
    if not completion.choices or not isinstance(completion.choices[0].message.content, str):
        raise ValueError("Retry-rule agent returned no text response")
    answer_present = answer is not None or bool(_ANSWER_LABEL.search(candidate_text))
    learned = _parse_rules(
        completion.choices[0].message.content, answer_present, existing, allowed_categories,
    )
    return retry_rules + [str(rule["rule"]) for rule in learned]
