"""Review cleaned Markdown books in resumable batches and write matching Markdown files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from time import perf_counter
from typing import TypedDict

from edc.text_agent_pipeline import TextAgentWorkflow
from edc.visolex_model import ViSoLexCorrector
from kg_pipeline import create_client


logger = logging.getLogger("edc.books")


class ProgressFormatter(logging.Formatter):
    """Write metadata-only events as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "time": self.formatTime(record), "level": record.levelname,
            "event": record.getMessage(), **getattr(record, "details", {}),
        }, ensure_ascii=False)


def configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_logger = logging.getLogger("edc")
    for handler in pipeline_logger.handlers[:]:
        pipeline_logger.removeHandler(handler)
        handler.close()
    pipeline_logger.setLevel(level)
    pipeline_logger.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(ProgressFormatter())
        pipeline_logger.addHandler(handler)


ROOT = Path(__file__).resolve().parent
_PARAGRAPH_GAP = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
_WHITESPACE = re.compile(r"\s+")
_MATH = re.compile(r"\$\$.*?\$\$|(?<!\$)\$(?!\$)[^$\n]*\$(?!\$)", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int


class SavedChunk(TypedDict):
    source_path: str
    source_sha256: str
    index: int
    start: int
    end: int
    status: str
    text: str


def _paragraph_spans(text: str) -> list[Chunk]:
    spans: list[Chunk] = []
    cursor = 0
    for gap in _PARAGRAPH_GAP.finditer(text):
        if text[cursor:gap.start()].strip():
            spans.append(Chunk(cursor, gap.start()))
        cursor = gap.end()
    if text[cursor:].strip():
        spans.append(Chunk(cursor, len(text)))
    return spans


def _split_long_span(text: str, span: Chunk, max_chars: int) -> list[Chunk]:
    pieces: list[Chunk] = []
    cursor = span.start
    math_spans = [(span.start + match.start(), span.start + match.end()) for match in _MATH.finditer(text[span.start:span.end])]
    while span.end - cursor > max_chars:
        boundaries = [
            match for match in _WHITESPACE.finditer(text, cursor, span.end)
            if match.start() > cursor and match.end() < span.end
            and all(not (start < match.start() < end or start < match.end() < end) for start, end in math_spans)
        ]
        before_limit = [match for match in boundaries if match.start() - cursor <= max_chars]
        boundary = before_limit[-1] if before_limit else (boundaries[0] if boundaries else None)
        if boundary is None:
            break
        pieces.append(Chunk(cursor, boundary.start()))
        cursor = boundary.end()
    pieces.append(Chunk(cursor, span.end))
    return pieces


def chunk_markdown(text: str, max_chars: int) -> list[Chunk]:
    """Find bounded spans while keeping original whitespace outside each span."""
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    pieces = [piece for span in _paragraph_spans(text) for piece in _split_long_span(text, span, max_chars)]
    chunks: list[Chunk] = []
    for piece in pieces:
        if chunks and piece.end - chunks[-1].start <= max_chars:
            chunks[-1] = Chunk(chunks[-1].start, piece.end)
        else:
            chunks.append(piece)
    return chunks


def _read_saved(path: Path) -> dict[tuple[str, int], SavedChunk]:
    saved: dict[tuple[str, int], SavedChunk] = {}
    if not path.exists():
        return saved
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid checkpoint JSON at {path}:{line_number}") from exc
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("source_path"), str)
                or not isinstance(item.get("source_sha256"), str)
                or type(item.get("index")) is not int
                or type(item.get("start")) is not int
                or type(item.get("end")) is not int
                or item.get("status") not in {"accepted", "dropped", "review_rejected"}
                or not isinstance(item.get("text"), str)
            ):
                raise ValueError(f"Invalid checkpoint record at {path}:{line_number}")
            key = item["source_path"], item["index"]
            if key in saved:
                raise ValueError(f"Duplicate checkpoint chunk {key} at {path}:{line_number}")
            saved[key] = item
    return saved


def _assembled_text(source: str, chunks: list[Chunk], records: list[SavedChunk]) -> str:
    parts: list[str] = []
    cursor = 0
    dropped = False
    for chunk, record in zip(chunks, records):
        parts.append(source[cursor:chunk.start])
        parts.append(source[chunk.start:chunk.end] if record["status"] == "review_rejected" else record["text"])
        dropped |= record["status"] == "dropped"
        cursor = chunk.end
    parts.append(source[cursor:])
    result = "".join(parts)
    if dropped:
        result = re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", result)
        result = result.strip("\n") + ("\n" if source.endswith("\n") and result.strip() else "")
    elif source.endswith("\n") and result and not result.endswith("\n"):
        result += "\n"
    return result


def _validate_directories(input_dir: Path, output_dir: Path, state_dir: Path) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Book input directory does not exist: {input_dir}")
    source = input_dir.resolve()
    for label, destination in (("output", output_dir), ("state", state_dir)):
        resolved = destination.resolve()
        if resolved == source or source in resolved.parents:
            raise ValueError(f"{label} directory must be outside the input directory")
    if output_dir.resolve() == state_dir.resolve() or output_dir.resolve() in state_dir.resolve().parents or state_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output and state directories must be separate")


def rewrite_books(
    input_dir: Path,
    output_dir: Path,
    state_dir: Path,
    workflow: TextAgentWorkflow,
    max_chars: int,
    batch_size: int,
    max_chunks: int,
    visolex: ViSoLexCorrector | None = None,
) -> tuple[int, int]:
    """Process Markdown chunks, checkpoint each result, and publish complete books."""
    _validate_directories(input_dir, output_dir, state_dir)
    if batch_size < 1 or max_chunks < 0:
        raise ValueError("batch_size must be positive and max_chunks cannot be negative")
    paths = sorted(input_dir.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown books found in {input_dir}")
    relative_names = {path.relative_to(input_dir).as_posix() for path in paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "config.json"
    config = {"version": 1, "input_dir": str(input_dir.resolve()), "max_chars": max_chars}
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Input directory or chunk size changed; use a fresh state directory: {state_dir}")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checkpoint_path = state_dir / "chunks.jsonl"
    saved = _read_saved(checkpoint_path)
    stale_paths = {name for name, _ in saved} - relative_names
    if stale_paths:
        raise ValueError(f"Checkpoint refers to missing books: {sorted(stale_paths)}")
    logger.info("run_started", extra={"details": {"books": len(paths), "saved_chunks": len(saved), "batch_size": batch_size}})
    completed_books = 0
    processed_chunks = 0
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for path in paths:
            relative = path.relative_to(input_dir)
            relative_name = relative.as_posix()
            source = path.read_text(encoding="utf-8")
            source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
            chunks = chunk_markdown(source, max_chars)
            book_saved = {index: record for (name, index), record in saved.items() if name == relative_name}
            if any(index >= len(chunks) for index in book_saved):
                raise ValueError(f"Checkpoint has extra chunks for {relative_name}; use a fresh state directory")
            for index, record in book_saved.items():
                chunk = chunks[index]
                if record["source_sha256"] != source_sha or (record["start"], record["end"]) != (chunk.start, chunk.end):
                    raise ValueError(f"Source or chunking changed for {relative_name}; use a fresh state directory")
            logger.info("book_started", extra={"details": {"book": relative_name, "chunks": len(chunks), "saved_chunks": len(book_saved)}})
            destination = output_dir / relative
            if destination.exists() and len(book_saved) != len(chunks):
                raise FileExistsError(f"Output exists without a complete checkpoint: {destination}")

            for batch_start in range(0, len(chunks), batch_size):
                for index in range(batch_start, min(batch_start + batch_size, len(chunks))):
                    if index in book_saved:
                        logger.debug("chunk_resumed", extra={"details": {"book": relative_name, "chunk": index + 1}})
                        continue
                    if max_chunks and processed_chunks >= max_chunks:
                        logger.info("chunk_limit_reached", extra={"details": {"processed_chunks": processed_chunks}})
                        return completed_books, processed_chunks
                    chunk = chunks[index]
                    chunk_text = source[chunk.start:chunk.end]
                    suggestions = visolex.suggest(chunk_text)[0] if visolex is not None else []
                    record_id = f"book:{hashlib.sha256(relative_name.encode('utf-8')).hexdigest()[:16]}::chunk_{index + 1:06d}"
                    started = perf_counter()
                    context = {"book": relative_name, "chunk": index + 1, "total_chunks": len(chunks), "record_id": record_id}
                    logger.info("chunk_started", extra={"details": context})
                    try:
                        result = workflow.process(record_id, chunk_text, suggestions)
                    except Exception as exc:
                        # Response bodies and exception messages can contain private source text.
                        logger.error("chunk_failed", extra={"details": {**context, "error_type": type(exc).__name__, "elapsed_seconds": round(perf_counter() - started, 3)}})
                        raise
                    record: SavedChunk = {
                        "source_path": relative_name, "source_sha256": source_sha,
                        "index": index, "start": chunk.start, "end": chunk.end,
                        "status": result["status"], "text": result["text"],
                    }
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    saved[relative_name, index] = record
                    book_saved[index] = record
                    processed_chunks += 1
                    logger.info("chunk_checkpointed", extra={"details": {**context, "status": result["status"], "attempts": len(result["history"]), "elapsed_seconds": round(perf_counter() - started, 3)}})
                logger.info("batch_completed", extra={"details": {"book": relative_name, "first_chunk": batch_start + 1, "last_chunk": min(batch_start + batch_size, len(chunks)), "total_chunks": len(chunks)}})

            assembled = _assembled_text(source, chunks, [book_saved[index] for index in range(len(chunks))])
            if destination.exists():
                if destination.read_text(encoding="utf-8") != assembled:
                    raise FileExistsError(f"Output differs from checkpoint: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                temporary.write_text(assembled, encoding="utf-8")
                temporary.replace(destination)
            completed_books += 1
            logger.info("book_completed", extra={"details": {"book": relative_name, "completed_books": completed_books}})
    return completed_books, processed_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "cleaned_books")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/reviewed_books")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=0, help="New chunks to process in this run; 0 means all")
    parser.add_argument("--plan", action="store_true", help="Count books and chunks without model calls or writes")
    parser.add_argument("--api-base-url", default="http://localhost:5000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--review-model")
    parser.add_argument("--rule-model")
    parser.add_argument("--agent-rounds", type=int, default=3)
    parser.add_argument("--visolex-checkpoint", type=Path)
    parser.add_argument("--visolex-tokenizer", default=str(ROOT / "checkpoints/bartpho-tokenizer"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-file", type=Path, help="Append progress logs here; default: state directory/run.log")
    args = parser.parse_args()
    state_dir = args.state_dir or args.output_dir.with_name(args.output_dir.name + "_state")
    _validate_directories(args.input_dir, args.output_dir, state_dir)
    if args.plan:
        paths = sorted(args.input_dir.rglob("*.md"))
        if not paths:
            raise ValueError(f"No Markdown books found in {args.input_dir}")
        chunks = sum(len(chunk_markdown(path.read_text(encoding="utf-8"), args.max_chars)) for path in paths)
        print(f"Plan: {len(paths)} books, {chunks} chunks at max_chars={args.max_chars}; batch_size={args.batch_size}.")
        return
    configure_logging(args.log_file or state_dir / "run.log", args.log_level)
    started = perf_counter()
    workflow = TextAgentWorkflow(
        create_client(args.api_base_url), args.model,
        args.review_model or args.model, args.rule_model or args.model,
        state_dir / "cheatsheet.json", args.agent_rounds,
        entries_log_path=state_dir / "feedback.jsonl",
    )
    visolex = (
        ViSoLexCorrector(args.visolex_checkpoint, args.visolex_tokenizer)
        if args.visolex_checkpoint else None
    )
    books, chunks = rewrite_books(
        args.input_dir, args.output_dir, state_dir, workflow,
        args.max_chars, args.batch_size, args.max_chunks, visolex,
    )
    logger.info("run_completed", extra={"details": {"books": books, "new_chunks": chunks, "output_dir": str(args.output_dir), "state_dir": str(state_dir), "elapsed_seconds": round(perf_counter() - started, 3)}})


if __name__ == "__main__":
    main()
