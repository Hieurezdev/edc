"""Use the local ViSoLex student checkpoint for conservative lexical correction."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TypedDict

from edc.vietnamese_preprocess import split_text


class ViSoLexSuggestion(TypedDict):
    source: str
    replacement: str
    confidence: float


class ViSoLexCorrector:
    def __init__(self, checkpoint: Path, tokenizer_source: str, max_tokens: int = 768) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ViSoLex checkpoint does not exist: {checkpoint}")

        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install requirements-visolex.txt in the project environment") from exc

        visolex_dir = Path(__file__).resolve().parents[1] / "visolex"
        if str(visolex_dir) not in sys.path:
            sys.path.insert(0, str(visolex_dir))
        from normalizer.model_construction.bartpho import get_bartpho_normalizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self._tokenizer.add_tokens(["<space>"])
        base_config = AutoConfig.from_pretrained(tokenizer_source)
        self._model = get_bartpho_normalizer(
            len(self._tokenizer), checkpoint_dir=str(checkpoint),
            mask_n_predictor=True, nsw_detector=True, base_config=base_config,
        )
        self._model.eval()
        self._max_tokens = max_tokens

    def suggest(self, text: str) -> tuple[list[ViSoLexSuggestion], str]:
        suggestions, status = self._suggest_single(text)
        if status not in {"skipped_token_limit", "skipped_tokenizer_alignment"} or len(text) <= 500:
            return suggestions, status

        parts = split_text(text, 500)
        if len(parts) == 1:
            return suggestions, status
        suggestions = []
        partial = False
        for part in parts:
            part_suggestions, part_status = self._suggest_single(part)
            suggestions.extend(part_suggestions)
            partial |= part_status.startswith("skipped_")
        unique: dict[tuple[str, str], ViSoLexSuggestion] = {}
        for suggestion in suggestions:
            key = suggestion["source"], suggestion["replacement"]
            if key not in unique or suggestion["confidence"] > unique[key]["confidence"]:
                unique[key] = suggestion
        status = "partial" if partial else ("suggested" if unique else "unchanged")
        return list(unique.values())[:12], status

    def _suggest_single(self, text: str) -> tuple[list[ViSoLexSuggestion], str]:
        tokens = self._tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"]
        if input_ids.shape[1] > self._max_tokens:
            return [], "skipped_token_limit"

        source_tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        special_tokens = set(self._tokenizer.all_special_tokens)
        source_content = [token for token in source_tokens if token not in special_tokens]
        if self._tokenizer.convert_tokens_to_string(source_content).strip() != text.strip():
            return [], "skipped_tokenizer_alignment"

        with self._torch.inference_mode():
            _, logits, _ = self._model(input_ids, tokens["attention_mask"])
            probabilities, predictions = self._torch.softmax(logits["logits_norm"][0], dim=-1).max(dim=-1)
            nonstandard = logits["logits_nsw_detection"][0].argmax(dim=-1)

        suggestions: list[ViSoLexSuggestion] = []
        groups: list[list[int]] = []
        for index, token in enumerate(source_tokens):
            if token in special_tokens:
                continue
            if token.startswith("▁") or not groups:
                groups.append([])
            groups[-1].append(index)

        for group in groups:
            if not any(int(nonstandard[index]) == 1 for index in group):
                continue
            source_word = self._tokenizer.convert_tokens_to_string(
                [source_tokens[index] for index in group]
            ).strip()
            predicted_tokens = [self._tokenizer.convert_ids_to_tokens(int(predictions[index])) for index in group]
            replacement_tokens = [token for token in predicted_tokens if token != "<space>"]
            if not replacement_tokens:
                continue
            replacement_word = self._tokenizer.convert_tokens_to_string(replacement_tokens).strip()
            confidence = min(float(probabilities[index]) for index in group)
            if (
                replacement_word != source_word
                and len(source_word) >= 3 and confidence >= 0.9
                and source_word.isalpha() and source_word.islower()
                and replacement_word.isalpha() and replacement_word.islower()
                and all(token not in special_tokens for token in replacement_tokens)
            ):
                suggestions.append({
                    "source": source_word, "replacement": replacement_word,
                    "confidence": round(confidence, 4),
                })
        return suggestions, "suggested" if suggestions else "unchanged"
