import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from edc.text_generator import OutputTruncatedError
from rewrite_books import ProgressFormatter, chunk_markdown, rewrite_books


class RewriteBooksTests(unittest.TestCase):
    def test_failure_logs_chunk_context_and_propagates_without_source_text(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            source = "Private source paragraph."
            (input_dir / "book.md").write_text(source, encoding="utf-8")
            workflow = Mock()
            failure = ValueError("Private model response")
            workflow.process.side_effect = failure
            with self.assertLogs("edc.books", level="INFO") as captured:
                with self.assertRaises(ValueError) as raised:
                    rewrite_books(input_dir, root / "output", root / "state", workflow, 100, 1, 0)
            self.assertIs(raised.exception, failure)
            event = json.loads(ProgressFormatter().format(captured.records[-1]))
            self.assertEqual(event["event"], "chunk_failed")
            self.assertEqual(event["chunk"], 1)
            self.assertEqual(event["error_type"], "ValueError")
            self.assertIn("record_id", event)
            serialized = "\n".join(ProgressFormatter().format(record) for record in captured.records)
            self.assertNotIn(source, serialized)
            self.assertNotIn(str(failure), serialized)
            self.assertEqual((root / "state/chunks.jsonl").read_text(), "")

    def test_chunking_preserves_original_text_and_latex_boundaries(self):
        formula = "$" + "x + " * 40 + "y$"
        source = "Mở đầu. " + formula + " Kết thúc.\n\nĐoạn tiếp theo " + "từ " * 50
        chunks = chunk_markdown(source, 100)
        self.assertGreater(len(chunks), 1)
        rebuilt = []
        cursor = 0
        for chunk in chunks:
            rebuilt.extend((source[cursor:chunk.start], source[chunk.start:chunk.end]))
            cursor = chunk.end
        rebuilt.append(source[cursor:])
        self.assertEqual("".join(rebuilt), source)
        self.assertEqual(sum(formula in source[chunk.start:chunk.end] for chunk in chunks), 1)

    def test_batches_resume_and_publish_matching_markdown_only_when_complete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "cleaned_books"
            source_path = input_dir / "nested" / "book.md"
            source_path.parent.mkdir(parents=True)
            first = "Câu hỏi gáp? " + "a" * 75
            second = "Đoạn thứ hai " + "b" * 75
            source = first + "\n\n" + second + "\n"
            source_path.write_text(source, encoding="utf-8")
            output_dir = root / "reviewed_books"
            state_dir = root / "reviewed_books_state"
            first_workflow = Mock()
            first_workflow.process.return_value = {"status": "accepted", "text": first.replace("gáp", "gấp"), "action": "rewrite", "history": []}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rewrite_books(input_dir, output_dir, state_dir, first_workflow, 100, 1, 1), (0, 1))
            self.assertFalse((output_dir / "nested" / "book.md").exists())
            self.assertEqual(first_workflow.process.call_count, 1)

            second_workflow = Mock()
            second_workflow.process.return_value = {"status": "review_rejected", "text": "", "action": "none", "history": []}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rewrite_books(input_dir, output_dir, state_dir, second_workflow, 100, 1, 0), (1, 1))
            self.assertEqual(second_workflow.process.call_count, 1)
            self.assertEqual((output_dir / "nested" / "book.md").read_text(encoding="utf-8"), first.replace("gáp", "gấp") + "\n\n" + second + "\n")

            with redirect_stdout(io.StringIO()):
                self.assertEqual(rewrite_books(input_dir, output_dir, state_dir, Mock(), 100, 2, 0), (1, 0))
            with self.assertRaisesRegex(ValueError, "chunk size changed"):
                rewrite_books(input_dir, output_dir, state_dir, Mock(), 120, 1, 0)
            source_path.write_text(source + "Đổi nguồn.", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source or chunking changed"):
                rewrite_books(input_dir, output_dir, state_dir, Mock(), 100, 1, 0)

    def test_truncated_chunk_splits_and_checkpoints_one_combined_result(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "input"
            source_dir.mkdir()
            source = "Đoạn đầu gáp " + "a" * 150 + "\n\nĐoạn sau gáp " + "b" * 150 + "\n"
            (source_dir / "book.md").write_text(source, encoding="utf-8")
            workflow = Mock()
            def process(record_id, text, suggestions):
                if len(text) > 200:
                    raise OutputTruncatedError("output budget exhausted")
                return {"text": text.replace("gáp", "gấp"), "status": "accepted", "action": "rewrite", "history": []}
            workflow.process.side_effect = process
            with self.assertLogs("edc.books", level="WARNING") as captured:
                self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", workflow, 1000, 1, 0), (1, 1))
            self.assertTrue(any(record.getMessage() == "chunk_split_after_truncation" for record in captured.records))
            self.assertGreater(workflow.process.call_count, 1)
            self.assertEqual((root / "output/book.md").read_text(), source.replace("gáp", "gấp"))
            checkpoints = (root / "state/chunks.jsonl").read_text().splitlines()
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(json.loads(checkpoints[0])["status"], "accepted")
            resumed = Mock()
            self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", resumed, 1000, 1, 0), (1, 0))
            resumed.process.assert_not_called()

    def test_unsplittable_truncation_keeps_source_and_can_be_reprocessed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "input"
            source_dir.mkdir()
            source = "x" * 300
            (source_dir / "book.md").write_text(source, encoding="utf-8")
            workflow = Mock()
            workflow.process.side_effect = OutputTruncatedError("output budget exhausted")
            with self.assertLogs("edc.books", level="ERROR"):
                self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", workflow, 1000, 1, 0), (1, 1))
            destination = root / "output/book.md"
            self.assertEqual(destination.read_text(), source)
            checkpoint = root / "state/chunks.jsonl"
            self.assertEqual(json.loads(checkpoint.read_text().splitlines()[0])["status"], "model_failed")
            resumed = Mock()
            self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", resumed, 1000, 1, 0), (1, 0))
            resumed.process.assert_not_called()
            resumed.process.return_value = {"text": "Đã sửa.", "status": "accepted", "action": "rewrite", "history": []}
            self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", resumed, 1000, 1, 0, retry_model_failed=True), (1, 1))
            self.assertEqual(destination.read_text(), "Đã sửa.")
            self.assertEqual([json.loads(line)["status"] for line in checkpoint.read_text().splitlines()], ["model_failed", "accepted"])
            backups = list((root / "output").glob("book.md.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), source)
            self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", Mock(), 1000, 1, 0), (1, 0))

    def test_repeated_truncation_stops_after_bounded_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "input"
            source_dir.mkdir()
            source = "Một đoạn văn có nghĩa. " * 50
            (source_dir / "book.md").write_text(source, encoding="utf-8")
            workflow = Mock()
            workflow.process.side_effect = OutputTruncatedError("output budget exhausted")
            with self.assertLogs("edc.books", level="ERROR"):
                self.assertEqual(rewrite_books(source_dir, root / "output", root / "state", workflow, 2000, 1, 0), (1, 1))
            self.assertLessEqual(workflow.process.call_count, 4)
            self.assertEqual((root / "output/book.md").read_text(), source)
            self.assertEqual(json.loads((root / "state/chunks.jsonl").read_text())["status"], "model_failed")

    def test_repair_output_preserves_backup_and_reuses_checkpoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            (source / "book.md").write_text("Nguồn gáp.\n", encoding="utf-8")
            workflow = Mock()
            workflow.process.return_value = {"status": "accepted", "text": "Nguồn gấp.", "history": []}
            output, state = root / "output", root / "state"
            rewrite_books(source, output, state, workflow, 100, 1, 0)
            destination = output / "book.md"
            changed = b"Manually modified content\r\n"
            destination.write_bytes(changed)
            resumed = Mock()
            with self.assertRaises(FileExistsError):
                rewrite_books(source, output, state, resumed, 100, 1, 0)
            self.assertEqual(destination.read_bytes(), changed)
            with self.assertLogs("edc.books", level="WARNING"):
                self.assertEqual(rewrite_books(source, output, state, resumed, 100, 1, 0, repair_output=True), (1, 0))
            resumed.process.assert_not_called()
            self.assertEqual(destination.read_text(), "Nguồn gấp.\n")
            backups = list(output.glob("book.md.bak-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), changed)
            self.assertEqual(rewrite_books(source, output, state, resumed, 100, 1, 0, repair_output=True), (1, 0))
            self.assertEqual(len(list(output.glob("book.md.bak-*"))), 1)

    def test_dropped_chunk_closes_paragraph_gap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "cleaned_books"
            input_dir.mkdir()
            paragraphs = [label * 65 for label in "ABC"]
            source = "\n\n".join(paragraphs) + "\n"
            (input_dir / "book.md").write_text(source, encoding="utf-8")
            workflow = Mock()
            workflow.process.side_effect = [
                {"status": "accepted", "text": paragraphs[0], "action": "keep", "history": []},
                {"status": "dropped", "text": "", "action": "drop", "history": []},
                {"status": "accepted", "text": paragraphs[2] + "\n", "action": "keep", "history": []},
            ]
            with redirect_stdout(io.StringIO()):
                books, chunks = rewrite_books(input_dir, root / "reviewed", root / "state", workflow, 100, 2, 0)
            self.assertEqual((books, chunks), (1, 3))
            self.assertEqual((root / "reviewed" / "book.md").read_text(encoding="utf-8"), paragraphs[0] + "\n\n" + paragraphs[2] + "\n")

    def test_accepted_rewrite_keeps_source_final_newline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "cleaned_books"
            input_dir.mkdir()
            (input_dir / "book.md").write_text("Một lỗi gáp.\n", encoding="utf-8")
            workflow = Mock()
            workflow.process.return_value = {"status": "accepted", "text": "Một lỗi gấp.", "action": "rewrite", "history": []}
            with redirect_stdout(io.StringIO()):
                rewrite_books(input_dir, root / "reviewed", root / "state", workflow, 100, 1, 0)
            self.assertEqual((root / "reviewed" / "book.md").read_text(encoding="utf-8"), "Một lỗi gấp.\n")


if __name__ == "__main__":
    unittest.main()
