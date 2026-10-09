"""Coordinate text generation, independent reviews, and reusable feedback."""

from __future__ import annotations

from collections import Counter
import json
import logging
from time import perf_counter
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, TypedDict

from edc.feedback_rules import RuleOutputError, derive_retry_rules, derive_rules
from edc.text_generator import RetryFeedback, generate_candidate
from edc.text_review import review_candidate
from edc.visolex_model import ViSoLexSuggestion

if TYPE_CHECKING:
    from openai import OpenAI


logger = logging.getLogger(__name__)


_MATH = re.compile(r"\$\$.*?\$\$|(?<!\$)\$(?!\$)[^$\n]*\$(?!\$)", re.DOTALL)


class FeedbackEntry(TypedDict):
    record_id: str
    attempt: int
    source: str
    candidate: dict[str, object]
    checks: list[dict[str, object]]
    accepted: bool
    retry_rules: list[str]


class TextAgentResult(TypedDict):
    text: str
    status: Literal["accepted", "dropped", "review_rejected", "model_failed"]
    action: str
    history: list[FeedbackEntry]


class TextAgentWorkflow:
    def __init__(
        self, client: OpenAI, model: str, review_model: str, rule_model: str,
        cheatsheet_path: Path, max_rounds: int, entries_log_path: Path | None = None,
        generation_max_tokens: int = 8192,
        enable_thinking: bool | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if generation_max_tokens < 1:
            raise ValueError("generation_max_tokens must be positive")
        self.generation_max_tokens = generation_max_tokens
        self.enable_thinking = enable_thinking
        self.client = client
        self.model = model
        self.review_model = review_model
        self.rule_model = rule_model
        self.cheatsheet_path = cheatsheet_path
        self.entries_log_path = entries_log_path
        self.max_rounds = max_rounds
        self.entries, self.rules = self._load_cheatsheet()
        if not cheatsheet_path.exists():
            self._save_entries()

    def _load_cheatsheet(self) -> tuple[list[FeedbackEntry], list[dict[str, object]]]:
        if not self.cheatsheet_path.exists():
            return [], []
        data = json.loads(self.cheatsheet_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("entries"), list):
            raise ValueError(f"Invalid text-agent cheatsheet: {self.cheatsheet_path}")
        expected_log = str(self.entries_log_path.resolve()) if self.entries_log_path else None
        if data.get("entries_log") != expected_log:
            raise ValueError(f"Cheatsheet feedback log does not match: {self.cheatsheet_path}")
        entries = data["entries"]
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("checks"), list):
                raise ValueError(f"Invalid cheatsheet entry {index} in {self.cheatsheet_path}")
        rules = data.get("rules", [])
        if not isinstance(rules, list) or any(
            not isinstance(rule, dict) or not isinstance(rule.get("rule"), str)
            for rule in rules
        ):
            raise ValueError(f"Invalid text-agent rules in {self.cheatsheet_path}")
        return entries, rules

    def _save_entries(self) -> None:
        self.cheatsheet_path.parent.mkdir(parents=True, exist_ok=True)
        if self.entries_log_path is not None:
            self.entries_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.entries:
                with self.entries_log_path.open("a", encoding="utf-8") as log:
                    for entry in self.entries:
                        log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self.entries.clear()
        temporary = self.cheatsheet_path.with_name(self.cheatsheet_path.name + ".tmp")
        temporary.write_text(
            json.dumps({
                "version": 1, "entries": self.entries, "rules": self.rules,
                "entries_log": str(self.entries_log_path.resolve()) if self.entries_log_path else None,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.cheatsheet_path)

    def process(
        self, record_id: str, source_text: str,
        visolex_suggestions: list[ViSoLexSuggestion],
    ) -> TextAgentResult:
        retry_rules: list[str] = []
        history: list[FeedbackEntry] = []
        for attempt in range(1, self.max_rounds + 1):
            context = {"record_id": record_id, "attempt": attempt}
            started = perf_counter()
            retry_feedback: list[RetryFeedback] = [
                {
                    "attempt": entry["attempt"],
                    "candidate": entry["candidate"],
                    "failed_checks": [check for check in entry["checks"] if not check["passed"]],
                }
                for entry in history if not entry["accepted"]
            ]
            logger.info("generation_started", extra={"details": {**context, "feedback_attempts": len(retry_feedback)}})
            confirmed_rules = [str(rule["rule"]) for rule in self.rules[-12:]]
            candidate = generate_candidate(
                self.client, self.model, source_text,
                visolex_suggestions, list(dict.fromkeys([*confirmed_rules, *retry_rules])),
                retry_feedback=retry_feedback,
                max_tokens=self.generation_max_tokens,
                enable_thinking=self.enable_thinking,
            )
            logger.info("generation_completed", extra={"details": {**context, "action": candidate["action"], "elapsed_seconds": round(perf_counter() - started, 3)}})
            checks = [*review_candidate(self.client, self.review_model, source_text, candidate, enable_thinking=self.enable_thinking)]
            if candidate["action"] == "rewrite":
                candidate_text = candidate["text"]
                if not isinstance(candidate_text, str):
                    raise ValueError("Text generator rewrite must contain string text")
                missing_formulas = Counter(_MATH.findall(source_text)) - Counter(_MATH.findall(candidate_text))
                if missing_formulas:
                    checks.append({
                        "check": "protected_math", "passed": False,
                        "feedback": f"Restore these original LaTeX spans exactly: {dict(missing_formulas)}",
                    })
            accepted = all(check["passed"] for check in checks)
            logger.info("review_completed", extra={"details": {**context, "accepted": accepted, "failed_checks": [check["check"] for check in checks if not check["passed"]]}})
            entry: FeedbackEntry = {
                "record_id": record_id, "attempt": attempt, "source": source_text,
                "candidate": candidate, "checks": checks, "accepted": accepted,
                "retry_rules": [],
            }
            self.entries.append(entry)
            history.append(entry)
            if accepted:
                self._save_entries()
                if len(history) > 1:
                    logger.info("confirmed_rules_started", extra={"details": context})
                    try:
                        new_rules = derive_rules(
                            self.client, self.rule_model, source_text,
                            history[:-1], candidate, self.rules,
                            enable_thinking=self.enable_thinking,
                        )
                    except RuleOutputError as exc:
                        logger.warning("confirmed_rules_unavailable", extra={"details": {**context, "error_type": type(exc).__name__}})
                        new_rules = []
                    logger.info("confirmed_rules_completed", extra={"details": {**context, "rules": len(new_rules)}})
                    existing = {(str(rule.get("category", "")), str(rule["rule"]).casefold().strip()) for rule in self.rules}
                    for rule in new_rules:
                        key = str(rule["category"]), str(rule["rule"]).casefold().strip()
                        if key not in existing:
                            self.rules.append({**rule, "source_record_id": record_id})
                            existing.add(key)
                    self._save_entries()
                if candidate["action"] == "drop":
                    return {"text": "", "status": "dropped", "action": "drop", "history": history}
                return {
                    "text": str(candidate["text"]), "status": "accepted",
                    "action": str(candidate["action"]), "history": history,
                }
            if attempt < self.max_rounds:
                logger.info("retry_rules_started", extra={"details": context})
                try:
                    new_retry_rules = derive_retry_rules(
                        self.client, self.rule_model, source_text, candidate, checks, self.rules,
                        enable_thinking=self.enable_thinking,
                    )
                except RuleOutputError as exc:
                    logger.warning("retry_rules_unavailable", extra={"details": {**context, "error_type": type(exc).__name__}})
                    new_retry_rules = []
                logger.info("retry_rules_completed", extra={"details": {**context, "rules": len(new_retry_rules)}})
                entry["retry_rules"] = new_retry_rules
                retry_rules = list(dict.fromkeys([*retry_rules, *new_retry_rules]))
            self._save_entries()
        logger.warning("review_rounds_exhausted", extra={"details": {"record_id": record_id, "attempts": self.max_rounds}})
        return {"text": "", "status": "review_rejected", "action": "none", "history": history}
