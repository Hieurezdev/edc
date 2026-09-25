"""Conservative cleanup of OCR and crawled Vietnamese before extraction."""

from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher


_DETAILS = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\(https?://[^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BOOK_TAG = re.compile(
    r"</?(a|article|b|br|details|div|em|h[1-6]|i|img|li|nl|ol|p|section|span|strong|sub|summary|sup|table|tbody|td|th|thead|tr|ul)(?=[ \t/>])[^>\n]*>",
    re.IGNORECASE,
)
_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_HORIZONTAL_RULE = re.compile(r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}$")
_FIGURE_CAPTION = re.compile(
    r"^[ \t]*(?:[▲△▶►●■•][ \t]*)?(?:hình|hinh)[ \t]+\d+(?:\.\d+)*(?:[a-z])?(?P<tail>.*)$",
    re.IGNORECASE,
)
_BOOK_MATH = re.compile(
    r"\$\$(?:(?!\n[ \t]*\n)[^$]){0,2000}\$\$|(?<!\$)\$(?!\$)[^$\n]{0,2000}\$(?!\$)",
    re.DOTALL,
)
_TAG = re.compile(
    r"</?(?:p|div|span|strong|em|br|ul|ol|li|h[1-6]|sup|sub|a|img|section|article|tbody|thead|table|tr|td|th|details|summary)\b[^>]*>",
    re.IGNORECASE,
)
_SPACE = re.compile(r"\s+")
_LATEX = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.DOTALL)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"(?u)\b[^\W\d_]+\b")
_OCR_REPLACEMENTS = (
    (re.compile(r"\bgọi tất\b", re.IGNORECASE), "gọi tắt"),
    (re.compile(r"\bviết tất\b", re.IGNORECASE), "viết tắt"),
    (re.compile(r"\bgóc ở tâm chấn cung\b", re.IGNORECASE), "góc ở tâm chắn cung"),
    (re.compile(r"\bsố do radian\b", re.IGNORECASE), "số đo radian"),
    (re.compile(r"\bbao nhiều đỏ\b", re.IGNORECASE), "bao nhiêu độ"),
)


def clean_book_markdown(raw_text: str) -> str:
    """Remove image markup and captions without leaving deletion gaps."""
    text = unicodedata.normalize("NFC", raw_text.replace("\r\n", "\n").replace("\r", "\n"))
    original_lines = text.split("\n")

    def blank(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    def clean_tag(match: re.Match[str]) -> str:
        tag = match.group(1).lower()
        closing = match.group(0).startswith("</")
        if tag == "sup":
            replacement = "}" if closing else "^{"
        elif tag == "sub":
            replacement = "}" if closing else "_{"
        elif closing and tag in {"td", "th", "tr"}:
            replacement = " ; "
        elif tag in {"br", "li", "p", "div"}:
            replacement = " "
        else:
            replacement = ""
        return replacement + blank(match)

    def clean_plain(plain: str) -> str:
        plain = _BOOK_TAG.sub(clean_tag, plain)
        plain = _LINK.sub(r"\1", plain)
        plain = html.unescape(plain)
        plain = _HEADING.sub("", plain)
        plain = _HORIZONTAL_RULE.sub("", plain)
        plain = "".join(
            " " if char == "\ufffd" or unicodedata.category(char) == "Zs" else char
            for char in plain
            if char in "\n\t" or unicodedata.category(char)[0] != "C"
        )
        return re.sub(r"[ \t]+(?=\n)", "", plain)

    text = _DETAILS.sub(blank, text)
    text = _IMAGE.sub(blank, text)
    text = _HTML_COMMENT.sub(blank, text)
    parts: list[str] = []
    previous_end = 0
    for match in _BOOK_MATH.finditer(text):
        parts.append(clean_plain(text[previous_end:match.start()]))
        parts.append(match.group(0))
        previous_end = match.end()
    parts.append(clean_plain(text[previous_end:]))
    cleaned_lines = "".join(parts).split("\n")
    if len(cleaned_lines) != len(original_lines):
        raise ValueError("Book cleanup changed line alignment before compacting whitespace")

    kept_lines: list[str] = []
    paragraph_break = False
    deleted_gap = False
    for original, line in zip(original_lines, cleaned_lines):
        if _is_figure_caption(line):
            deleted_gap = True
            continue
        if not line.strip():
            if original.strip():
                deleted_gap = True
            elif kept_lines:
                paragraph_break = True
            continue
        if paragraph_break and not deleted_gap:
            kept_lines.append("")
        paragraph_break = False
        deleted_gap = False
        kept_lines.append(line)
    result = "\n".join(kept_lines)
    return result + ("\n" if text.endswith("\n") and kept_lines else "")


def _is_figure_caption(line: str) -> bool:
    match = _FIGURE_CAPTION.match(line)
    if match is None:
        return False
    tail = match.group("tail").strip()
    return not tail.strip(".,:;()–—-") or tail[0] in ".:–—-" or tail[0].isupper()


def clean_vietnamese_text(raw_text: str) -> str:
    """Remove markup and apply only high confidence, context bound OCR fixes."""
    text = unicodedata.normalize("NFC", html.unescape(raw_text))
    math_segments: list[str] = []

    def save_math(match: re.Match[str]) -> str:
        math_segments.append(match.group(0))
        return f"\ue000{len(math_segments) - 1}\ue001"

    text = _LATEX.sub(save_math, text)
    text = _DETAILS.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = re.sub(r"</?(?:table|tr)\b[^>]*>", " . ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:td|th)\b[^>]*>", " | ", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = text.replace("\\-", "-")
    text = re.sub(r"(?<!\\)#\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    for pattern, replacement in _OCR_REPLACEMENTS:
        text = pattern.sub(
            lambda match: replacement.capitalize() if match.group(0)[0].isupper() else replacement,
            text,
        )
    text = _SPACE.sub(" ", text).strip()
    for index, formula in enumerate(math_segments):
        text = text.replace(f"\ue000{index}\ue001", formula)
    return text


def strip_image_descriptions_preserving_lines(text: str) -> str:
    """Remove crawled image descriptions without changing source line numbers."""
    had_final_newline = text.endswith("\n")
    text = _DETAILS.sub(lambda match: "\n" * match.group(0).count("\n"), text)
    text = _IMAGE.sub(" ", text)
    if not had_final_newline and text.endswith("\n"):
        text += " "
    return text


def split_text(text: str, max_chars: int) -> list[str]:
    """Split long OCR chunks near sentence or word boundaries."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    pieces: list[str] = []
    while len(text) > max_chars:
        window = text[: max_chars + 1]
        math_spans = [(match.start(), match.end()) for match in _LATEX.finditer(text)]

        def outside_math(position: int) -> bool:
            return all(not (start < position < end) for start, end in math_spans)

        sentence_boundaries = [
            match.end() for match in re.finditer(r"[.!?]\s+", window)
            if match.end() >= max_chars // 2 and outside_math(match.end())
        ]
        word_boundaries = [
            match.end() for match in re.finditer(r"\s+", window)
            if outside_math(match.end())
        ]
        boundary = sentence_boundaries[-1] if sentence_boundaries else (word_boundaries[-1] if word_boundaries else 0)
        if not boundary:
            later_boundaries = (
                max_chars + match.end() for match in re.finditer(r"\s+", text[max_chars:])
            )
            boundary = next((position for position in later_boundaries if outside_math(position)), len(text))
        pieces.append(text[:boundary].strip())
        text = text[boundary:].strip()
    if text:
        pieces.append(text)
    return pieces


def correction_is_safe(original: str, corrected: str) -> bool:
    """Reject model rewrites that alter protected facts or much of the source."""
    if not original or not corrected or not (0.6 <= len(corrected) / len(original) <= 1.4):
        return False

    def capitalized_words(text: str) -> list[str]:
        return [match.group() for match in _WORD.finditer(text) if match.group()[0].isupper()]

    return (
        _LATEX.findall(original) == _LATEX.findall(corrected)
        and _NUMBER.findall(original) == _NUMBER.findall(corrected)
        and capitalized_words(original) == capitalized_words(corrected)
        and SequenceMatcher(None, original, corrected).ratio() >= 0.8
    )
