"""Run Vietnamese corpus cleanup, EDC extraction, and GraphJudge verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Iterator, TextIO

if TYPE_CHECKING:
    from openai import OpenAI

from edc.vietnamese_preprocess import (
    clean_vietnamese_text,
    correction_is_safe,
    split_text,
    strip_image_descriptions_preserving_lines,
)
from edc.text_agent_pipeline import TextAgentWorkflow
from edc.visolex_model import ViSoLexCorrector, ViSoLexSuggestion
from GraphJudge.graph_judger.verify_triples import Triple, judge_triple


ROOT = Path(__file__).resolve().parent


def write_json_line(file: TextIO, record: dict[str, object]) -> None:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_client(base_url: str) -> OpenAI:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and not base_url:
        raise ValueError("An API key or an OpenAI-compatible base URL is required")
    return OpenAI(api_key=api_key or "local", base_url=base_url)


def correct_text(
    client: OpenAI, model: str, text: str,
    suggestions: list[ViSoLexSuggestion],
) -> tuple[str, str]:
    suggestion_text = json.dumps(suggestions, ensure_ascii=False) if suggestions else "không có"
    prompt = (
        "Sửa lỗi chính tả/OCR tiếng Việt trong đoạn sau. Giữ nguyên mọi sự kiện, "
        "tên riêng, số liệu, công thức LaTeX và thứ tự câu. Không thêm thông tin. "
        "Gợi ý ViSoLex có thể sai với thuật ngữ, tiếng địa phương hoặc từ tiếng Anh; "
        "chỉ dùng nếu đúng trong ngữ cảnh. Không thay ký hiệu, tên riêng, "
        "chữ viết tắt chuyên môn hoặc từ vốn đã đúng. "
        "Chỉ trả lại văn bản đã sửa, không giải thích.\n"
        f"Dự đoán của mô hình ViSoLex: {suggestion_text}\n\nVăn bản: {text}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    candidate = (completion.choices[0].message.content or "").strip()
    if correction_is_safe(text, candidate):
        return candidate, "accepted"
    return text, "rejected"


def markdown_records(source_dir: Path, max_chars: int) -> Iterator[dict[str, object]]:
    paths = sorted(source_dir.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown books found in {source_dir}")
    for path in paths:
        relative_path = path.relative_to(source_dir)
        source_id = "book:" + hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]
        original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        filtered = strip_image_descriptions_preserving_lines("".join(original_lines))
        filtered_lines = filtered.splitlines(keepends=True)
        if len(original_lines) != len(filtered_lines):
            raise ValueError(f"Image cleanup changed line numbering in {path}")

        blocks: list[tuple[int, int, str]] = []
        block_start: int | None = None
        for line_number, line in enumerate(filtered_lines, start=1):
            if line.strip():
                if block_start is None:
                    block_start = line_number
            elif block_start is not None:
                cleaned = clean_vietnamese_text("".join(filtered_lines[block_start - 1:line_number - 1]))
                if cleaned:
                    blocks.append((block_start, line_number - 1, cleaned))
                block_start = None
        if block_start is not None:
            cleaned = clean_vietnamese_text("".join(filtered_lines[block_start - 1:]))
            if cleaned:
                blocks.append((block_start, len(filtered_lines), cleaned))

        segment_parts: list[str] = []
        segment_start = 0
        segment_end = 0
        segment_length = 0
        segment_number = 0

        def make_record() -> dict[str, object]:
            return {
                "id": f"{source_id}::seg_{segment_number:05d}",
                "source_id": source_id,
                "source_line": segment_start,
                "text": "\n\n".join(segment_parts),
                "raw_text": "".join(original_lines[segment_start - 1:segment_end]),
                "metadata": {
                    "doc_id": source_id,
                    "source_file_name": path.name,
                    "source_path": relative_path.as_posix(),
                    "start_line": segment_start,
                    "end_line": segment_end,
                },
            }

        for start_line, end_line, block in blocks:
            for piece in split_text(block, max_chars):
                additional = len(piece) + (2 if segment_parts else 0)
                if segment_parts and segment_length + additional > max_chars:
                    segment_number += 1
                    yield make_record()
                    segment_parts = []
                    segment_length = 0
                if not segment_parts:
                    segment_start = start_line
                segment_parts.append(piece)
                segment_end = end_line
                segment_length += len(piece) + (2 if len(segment_parts) > 1 else 0)
        if segment_parts:
            segment_number += 1
            yield make_record()


def input_records(source_path: Path, max_chars: int) -> Iterator[tuple[int, dict[str, object]]]:
    if source_path.is_dir():
        for index, record in enumerate(markdown_records(source_path, max_chars), start=1):
            yield index, record
    elif source_path.is_file():
        with source_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {source_path} at line {line_number}") from exc
                yield line_number, item
    else:
        raise FileNotFoundError(f"Input does not exist: {source_path}")


def prepare_corpus(args: argparse.Namespace, output_dir: Path) -> int:
    text_agent_model = getattr(args, "text_agent_model", None)
    if text_agent_model and args.correction_model:
        raise ValueError("Use either --text-agent-model or --correction-model, not both")
    correction_client = create_client(args.api_base_url) if args.correction_model else None
    text_agents = (
        TextAgentWorkflow(
            create_client(args.api_base_url), text_agent_model,
            getattr(args, "review_model", None) or text_agent_model,
            getattr(args, "rule_model", None) or text_agent_model,
            getattr(args, "cheatsheet_path", None) or output_dir / "cheatsheet.json",
            getattr(args, "agent_rounds", 3),
        )
        if text_agent_model else None
    )
    visolex_corrector = (
        ViSoLexCorrector(args.visolex_checkpoint, args.visolex_tokenizer)
        if args.visolex_checkpoint else None
    )
    input_path = output_dir / "edc_input.txt"
    prepared_path = output_dir / "prepared.jsonl"
    seen_ids: set[str] = set()
    count = 0
    with prepared_path.open("w", encoding="utf-8") as prepared, input_path.open("w", encoding="utf-8") as edc_input:
        for record_number, item in input_records(args.corpus, args.max_chars):
            if args.limit and record_number > args.limit:
                break
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("text"), str):
                raise ValueError(f"Expected string id and text in {args.corpus} at record {record_number}")
            if not isinstance(item.get("metadata", {}), dict):
                raise ValueError(f"Expected object metadata in {args.corpus} at record {record_number}")
            source_line = item.get("source_line", record_number)
            if not isinstance(source_line, int) or source_line < 1:
                raise ValueError(f"Invalid source line in {args.corpus} at record {record_number}")
            record_id = item["id"]
            if record_id in seen_ids:
                raise ValueError(f"Duplicate input id {record_id!r} at record {record_number}")
            seen_ids.add(record_id)
            cleaned = clean_vietnamese_text(item["text"])
            pieces = split_text(cleaned, args.max_chars) if cleaned else []
            if not pieces:
                write_json_line(prepared, {
                    "id": record_id, "source_line": source_line, "metadata": item.get("metadata", {}),
                    "raw_text": item.get("raw_text", item["text"]), "clean_text": "",
                    "visolex_suggestions": [], "visolex_status": "skipped_empty_after_cleanup",
                    "status": "skipped_empty_after_cleanup",
                })
                continue
            for part_number, piece in enumerate(pieces):
                normalized_text = piece
                segment_id = record_id if len(pieces) == 1 else f"{record_id}::part{part_number + 1}"
                visolex_suggestions: list[ViSoLexSuggestion] = []
                visolex_status = "not_requested"
                if visolex_corrector is not None:
                    visolex_suggestions, visolex_status = visolex_corrector.suggest(piece)
                correction_status = "not_requested"
                if correction_client is not None:
                    piece, correction_status = correct_text(
                        correction_client, args.correction_model, piece, visolex_suggestions,
                    )
                agent_status = "not_requested"
                agent_action = "none"
                agent_history: list[dict[str, object]] = []
                if text_agents is not None:
                    agent_result = text_agents.process(segment_id, piece, visolex_suggestions)
                    piece = agent_result["text"]
                    agent_status = agent_result["status"]
                    agent_action = agent_result["action"]
                    agent_history = agent_result["history"]
                status = "ready" if agent_status not in {"dropped", "review_rejected"} else f"skipped_{agent_status}"
                write_json_line(prepared, {
                    "id": segment_id, "source_id": item.get("source_id", record_id), "source_line": source_line,
                    "part": part_number + 1, "metadata": item.get("metadata", {}),
                    "raw_text": item.get("raw_text", item["text"]), "normalized_text": normalized_text,
                    "clean_text": piece,
                    "visolex_suggestions": visolex_suggestions, "visolex_status": visolex_status,
                    "correction_status": correction_status, "agent_status": agent_status,
                    "agent_action": agent_action, "agent_history": agent_history, "status": status,
                })
                if status == "ready":
                    edc_input.write(piece.replace("\n", " ") + "\n")
                    count += 1
    return count


def run_edc(args: argparse.Namespace, output_dir: Path) -> Path:
    edc_dir = output_dir / "edc"
    api_model = f"api:{args.model}"
    command = [
        args.python, str(ROOT / "run.py"),
        "--input_text_file_path", str(output_dir / "edc_input.txt"),
        "--output_dir", str(edc_dir),
        "--oie_llm", api_model,
        "--sd_llm", api_model,
        "--sc_llm", api_model,
        "--sc_embedder", args.embedder,
        "--oie_prompt_template_file_path", str(ROOT / "prompt_templates/vi_oie_template.txt"),
        "--oie_few_shot_example_file_path", str(ROOT / "few_shot_examples/vi/oie.txt"),
        "--sd_prompt_template_file_path", str(ROOT / "prompt_templates/vi_sd_template.txt"),
        "--sd_few_shot_example_file_path", str(ROOT / "few_shot_examples/vi/sd.txt"),
        "--sc_prompt_template_file_path", str(ROOT / "prompt_templates/vi_sc_template.txt"),
        "--target_schema_path", str(ROOT / "schemas/vi_empty_schema.csv"),
        "--enrich_schema",
    ]
    environment = os.environ.copy()
    environment["EDC_OPENAI_BASE_URL"] = args.api_base_url
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return edc_dir / "iter0/result_at_each_stage.json"


def get_triples(stage_result: dict[str, object]) -> list[Triple]:
    triples = stage_result.get("schema_canonicalizaiton")
    if not isinstance(triples, list):
        raise ValueError("EDC result is missing schema_canonicalizaiton")
    valid: list[Triple] = []
    seen: set[Triple] = set()
    for triple in triples:
        if triple is None:
            continue
        if not isinstance(triple, list) or len(triple) != 3 or not all(isinstance(value, str) and value.strip() for value in triple):
            raise ValueError(f"EDC emitted an invalid triple: {triple!r}")
        normalized = (triple[0].strip(), triple[1].strip(), triple[2].strip())
        if normalized not in seen:
            valid.append(normalized)
            seen.add(normalized)
    return valid


def verify_graph(args: argparse.Namespace, output_dir: Path, edc_results_path: Path) -> None:
    results = json.loads(edc_results_path.read_text(encoding="utf-8"))
    with (output_dir / "prepared.jsonl").open(encoding="utf-8") as prepared_file:
        prepared = [json.loads(line) for line in prepared_file]
    ready = [record for record in prepared if record["status"] == "ready"]
    if len(results) != len(ready):
        raise ValueError(f"EDC returned {len(results)} entries for {len(ready)} prepared segments")
    client = create_client(args.api_base_url)
    with (output_dir / "kg.jsonl").open("w", encoding="utf-8") as kg_file, (output_dir / "triples.jsonl").open("w", encoding="utf-8") as triples_file:
        for index, (record, result) in enumerate(zip(ready, results)):
            expected_input = record["clean_text"].replace("\n", " ").strip()
            if result.get("index") != index or result.get("input_text", "").strip() != expected_input:
                raise ValueError(f"EDC result at index {index} does not match prepared segment {record['id']}")
            candidates = get_triples(result)
            judgments: list[dict[str, object]] = []
            for triple in candidates:
                label, raw_response = judge_triple(client, args.judge_model or args.model, record["clean_text"], triple)
                judgment = {"triple": list(triple), "label": label, "response": raw_response}
                judgments.append(judgment)
                if label == "YES":
                    write_json_line(triples_file, {
                        "source_id": record["source_id"], "segment_id": record["id"],
                        "source_line": record["source_line"], "metadata": record["metadata"],
                        "triple": list(triple),
                    })
            write_json_line(kg_file, {
                "id": record["id"], "source_id": record["source_id"],
                "source_line": record["source_line"], "metadata": record["metadata"],
                "clean_text": record["clean_text"], "judgments": judgments,
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--corpus", dest="corpus", type=Path, default=ROOT / "crawled_books")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10, help="Markdown chunks or JSONL rows to process; 0 means all")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--api-base-url", default="http://localhost:5000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--correction-model", default=None)
    parser.add_argument("--text-agent-model", default=None)
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--rule-model", default=None)
    parser.add_argument("--agent-rounds", type=int, default=3)
    parser.add_argument("--cheatsheet-path", type=Path, default=None)
    parser.add_argument("--visolex-checkpoint", type=Path, default=None)
    parser.add_argument("--visolex-tokenizer", default=str(ROOT / "checkpoints/bartpho-tokenizer"))
    parser.add_argument("--embedder", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.agent_rounds < 1:
        parser.error("--agent-rounds must be at least 1")
    if args.text_agent_model and args.correction_model:
        parser.error("Use either --text-agent-model or --correction-model, not both")
    if args.review_model and not args.text_agent_model:
        parser.error("--review-model requires --text-agent-model")
    if args.rule_model and not args.text_agent_model:
        parser.error("--rule-model requires --text-agent-model")
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    count = prepare_corpus(args, args.output_dir)
    print(f"Prepared {count} segments in {args.output_dir}")
    if args.prepare_only or count == 0:
        return
    results_path = run_edc(args, args.output_dir)
    verify_graph(args, args.output_dir, results_path)
    print(f"Verified graph written to {args.output_dir / 'kg.jsonl'}")


if __name__ == "__main__":
    main()
