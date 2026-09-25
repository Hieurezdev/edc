"""Vietnamese, source-grounded GraphJudge inference for EDC triples."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


Triple = tuple[str, str, str]
_LABEL = re.compile(r"\s*(YES|NO|UNSURE)\s*[.!]?\s*", re.IGNORECASE)


def parse_judgment(response: str) -> str:
    match = _LABEL.fullmatch(response)
    return match.group(1).upper() if match else "UNSURE"


def judge_triple(client: OpenAI, model: str, source_text: str, triple: Triple) -> tuple[str, str]:
    """Return a strict label and raw response; only YES is accepted downstream."""
    subject, relation, object_ = triple
    prompt = (
        "Bạn là người kiểm chứng bộ ba của knowledge graph. Chỉ dùng văn bản nguồn. "
        "Trả lời đúng một nhãn: YES nếu văn bản hỗ trợ trực tiếp bộ ba, "
        "NO nếu văn bản mâu thuẫn, UNSURE nếu thiếu bằng chứng hoặc không rõ. "
        "Không suy diễn từ kiến thức bên ngoài.\n\n"
        f"Văn bản nguồn: {source_text}\n"
        f"Bộ ba: [{subject!r}, {relation!r}, {object_!r}]\n"
        "Nhãn:"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    raw = completion.choices[0].message.content or ""
    return parse_judgment(raw), raw
