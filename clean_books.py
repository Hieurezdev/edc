"""Write cleaned Markdown books to a separate directory without changing originals."""

from __future__ import annotations

import argparse
from pathlib import Path

from edc.vietnamese_preprocess import clean_book_markdown


ROOT = Path(__file__).resolve().parent


def clean_books(input_dir: Path, output_dir: Path) -> tuple[int, int, int]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Book input directory does not exist: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Cleaned book directory already exists: {output_dir}")
    if input_dir.resolve() == output_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output directory must be outside the book input directory")

    paths = sorted(input_dir.rglob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown books found in {input_dir}")

    input_chars = 0
    output_chars = 0
    output_dir.mkdir(parents=True)
    for source in paths:
        original = source.read_text(encoding="utf-8")
        cleaned = clean_book_markdown(original)
        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(cleaned, encoding="utf-8")
        input_chars += len(original)
        output_chars += len(cleaned)
    return len(paths), input_chars, output_chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "crawled_books")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cleaned_books")
    args = parser.parse_args()
    count, input_chars, output_chars = clean_books(args.input_dir, args.output_dir)
    print(f"Cleaned {count} books into {args.output_dir} ({input_chars - output_chars:,} characters removed)")


if __name__ == "__main__":
    main()
