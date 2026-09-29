from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from clean_books import clean_books
from edc.vietnamese_preprocess import clean_book_markdown, clean_vietnamese_text, correction_is_safe, split_text
from edc.text_agent_pipeline import TextAgentWorkflow
from edc.feedback_rules import _parse_rules, derive_retry_rules, derive_rules
from edc.text_generator import OutputTruncatedError, _parse_candidate, generate_candidate
from edc.text_review import _parse_verdict, review_candidate
from GraphJudge.graph_judger.verify_triples import parse_judgment
from kg_pipeline import create_client, markdown_records, prepare_corpus, verify_graph


class VietnamesePipelineTests(unittest.TestCase):
    def test_client_passes_configured_timeout_without_changing_other_callers(self):
        with patch("openai.OpenAI") as openai_client:
            create_client("http://localhost:8000/v1", timeout=1800)
            self.assertEqual(openai_client.call_args.kwargs["timeout"], 1800)
            create_client("http://localhost:8000/v1")
            self.assertNotIn("timeout", openai_client.call_args.kwargs)
            with self.assertRaisesRegex(ValueError, "timeout must be positive"):
                create_client("http://localhost:8000/v1", timeout=0)
            self.assertEqual(openai_client.call_count, 2)

    def test_book_cleanup_removes_image_markup_and_preserves_math(self):
        original = (
            "# Định lí\n![](images/a.png)\n"
            "<details><summary>natural_image</summary>\nAn illustration\n</details>\n"
            "<details><summary>text_image</summary>\nA ≤ B\n</details>\n"
            "<table><tr><td>x</td><td>√2</td></tr></table>\n"
            "$x < 5$ và a ≤ b, 2**3**2, H<sup>2</sup>O; &lt; vẫn là dấu so sánh. <iostream>\u200b\n"
        )
        cleaned = clean_book_markdown(original)
        self.assertNotIn("\n\n", cleaned)
        self.assertNotIn("![](", cleaned)
        self.assertNotIn("<details>", cleaned)
        self.assertNotIn("An illustration", cleaned)
        self.assertNotIn("A ≤ B", cleaned)
        self.assertIn("a ≤ b", cleaned)
        self.assertIn("√2", cleaned)
        self.assertIn("$x < 5$", cleaned)
        self.assertIn("2**3**2", cleaned)
        self.assertIn("H^{2}O", cleaned)
        self.assertIn("<iostream>", cleaned)
        self.assertNotIn("\u200b", cleaned)

    def test_book_cleanup_writes_separate_folder(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "crawled_books"
            source_dir.mkdir()
            source = source_dir / "sach.md"
            source.write_text("# Sách\n![](images/a.png)\n$2 + 2 = 4$\n", encoding="utf-8")
            output_dir = root / "cleaned_books"
            count, _, _ = clean_books(source_dir, output_dir)
            self.assertEqual(count, 1)
            self.assertIn("![](", source.read_text(encoding="utf-8"))
            self.assertEqual((output_dir / "sach.md").read_text(encoding="utf-8"), "Sách\n$2 + 2 = 4$\n")

    def test_book_cleanup_removes_numbered_figure_captions_only(self):
        original = (
            "Hình 1.1. Sơ đồ $x^2$\n"
            "▲Hình 2.11: Mô hình\n"
            "Hinh 3 Một số ví dụ\n"
            "Hình 4\n"
            "Hình 5.2,\n"
            "Hinh 6.3a)\n"
            "Hình 7.1 là biểu đồ của $y = x^2$.\n"
            "\n"
            "Quan sát Hình 1.1 và tính $x + 1$.\n"
        )
        cleaned = clean_book_markdown(original)
        self.assertEqual(cleaned, "Hình 7.1 là biểu đồ của $y = x^2$.\n\nQuan sát Hình 1.1 và tính $x + 1$.\n")

    def test_book_cleanup_closes_gaps_around_deleted_content(self):
        original = "Đoạn một.\n\n![](figure.png)\n\nĐoạn hai.\n\nHình 1.1. Ảnh\n\nĐoạn ba.\n\nĐoạn bốn.\n"
        self.assertEqual(
            clean_book_markdown(original),
            "Đoạn một.\nĐoạn hai.\nĐoạn ba.\n\nĐoạn bốn.\n",
        )

    def test_cleanup_preserves_formula_and_removes_image_markup(self):
        raw = "Góc ở tâm chấn cung 1 rad. ![](images/a.jpg) <details>photo description</details> $2 \\pi   R$"
        self.assertEqual(clean_vietnamese_text(raw), "Góc ở tâm chắn cung 1 rad. $2 \\pi   R$")
        self.assertFalse(correction_is_safe("Góc 1 rad $2\\pi$", "Góc 2 rad $2\\pi$"))
        self.assertFalse(correction_is_safe("Hà Nội là thủ đô.", "Paris là thủ đô."))
        self.assertTrue(correction_is_safe("Một dieu kì lạ.", "Một điều kì lạ."))
        self.assertEqual(parse_judgment("Yes, probably"), "UNSURE")

    def test_split_keeps_formula_intact(self):
        formula = "$" + "x + " * 40 + "1$"
        parts = split_text("Mở đầu. " * 12 + formula + " Kết thúc.", 100)
        self.assertEqual(sum(formula in part for part in parts), 1)

    def test_markdown_books_keep_source_lines_and_remove_image_descriptions(self):
        with TemporaryDirectory() as directory:
            book = Path(directory) / "sach.md"
            book.write_text(
                "# Bài học\n\nHà Nội là thủ đô của Việt Nam.\n\n"
                "![](images/a.jpg)\n<details>\n<summary>natural_image</summary>\nẢnh minh họa\n</details>\n\n"
                "Paris là thủ đô của Pháp.\n",
                encoding="utf-8",
            )
            records = list(markdown_records(Path(directory), 100))
            self.assertEqual(len(records), 1)
            self.assertIn("Hà Nội", records[0]["text"])
            self.assertIn("Paris", records[0]["text"])
            self.assertNotIn("Ảnh minh họa", records[0]["text"])
            self.assertEqual(records[0]["metadata"]["source_file_name"], "sach.md")
            self.assertEqual(records[0]["metadata"]["start_line"], 1)
            self.assertEqual(records[0]["metadata"]["end_line"], 11)

    def test_verification_preserves_alignment_and_filters_non_yes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            prepared = [
                {"id": "skip", "status": "skipped_empty_after_cleanup"},
                {"id": "part", "source_id": "source", "source_line": 2,
                 "metadata": {"doc_id": "doc_1"}, "clean_text": "Hà Nội là thủ đô Việt Nam.", "status": "ready"},
            ]
            (output / "prepared.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in prepared), encoding="utf-8"
            )
            stage_path = output / "stage.json"
            stage_path.write_text(json.dumps([{
                "index": 0,
                "input_text": "Hà Nội là thủ đô Việt Nam.\n",
                "schema_canonicalizaiton": [
                    ["Hà Nội", "là thủ đô của", "Việt Nam"],
                    ["Hà Nội", "là thủ đô của", "Pháp"],
                ]
            }]), encoding="utf-8")
            args = Namespace(api_base_url="http://localhost:5000/v1", judge_model=None, model="judge")
            with patch("kg_pipeline.create_client"), patch(
                "kg_pipeline.judge_triple", side_effect=[("YES", "YES"), ("NO", "NO")]
            ):
                verify_graph(args, output, stage_path)
            with (output / "kg.jsonl").open(encoding="utf-8") as graph_file:
                graph = [json.loads(line) for line in graph_file]
            with (output / "triples.jsonl").open(encoding="utf-8") as triples_file:
                triples = [json.loads(line) for line in triples_file]
            self.assertEqual(graph[0]["id"], "part")
            self.assertEqual([item["label"] for item in graph[0]["judgments"]], ["YES", "NO"])
            self.assertEqual(len(triples), 1)
            self.assertEqual(triples[0]["source_id"], "source")
            self.assertEqual(triples[0]["triple"], ["Hà Nội", "là thủ đô của", "Việt Nam"])

    def test_verification_accepts_answer_line_flattened_for_edc(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prepared.jsonl").write_text(
                json.dumps({
                    "id": "mcq", "source_id": "book", "source_line": 1,
                    "metadata": {}, "clean_text": "Câu hỏi?\nĐáp án: B\nGiải thích: Vì 2 + 2 = 4.", "status": "ready",
                }, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            stage = root / "stage.json"
            stage.write_text(json.dumps([{
                "index": 0, "input_text": "Câu hỏi? Đáp án: B Giải thích: Vì 2 + 2 = 4.\n",
                "schema_canonicalizaiton": [],
            }], ensure_ascii=False), encoding="utf-8")
            args = Namespace(api_base_url="http://localhost:5000/v1", judge_model=None, model="judge")
            with patch("kg_pipeline.create_client"):
                verify_graph(args, root, stage)
            self.assertEqual(json.loads((root / "kg.jsonl").read_text(encoding="utf-8"))["id"], "mcq")

    def test_visolex_suggestions_preserve_source_until_contextual_correction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"id": "sample", "text": "Tôi khum biết."}, ensure_ascii=False) + "\n")
            args = Namespace(
                corpus=source, max_chars=1800, limit=0, correction_model=None,
                api_base_url="http://localhost:5000/v1",
                visolex_checkpoint=root / "checkpoint.pt", visolex_tokenizer="local-tokenizer",
            )
            suggestion = {"source": "khum", "replacement": "không", "confidence": 0.9999}
            with patch("kg_pipeline.ViSoLexCorrector") as corrector:
                corrector.return_value.suggest.return_value = ([suggestion], "suggested")
                self.assertEqual(prepare_corpus(args, root), 1)
            prepared = json.loads((root / "prepared.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prepared["clean_text"], "Tôi khum biết.")
            self.assertEqual(prepared["visolex_suggestions"], [suggestion])
            self.assertEqual(prepared["visolex_status"], "suggested")

    def test_contextual_correction_receives_visolex_suggestion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"id": "sample", "text": "Tôi khum biết."}, ensure_ascii=False) + "\n")
            args = Namespace(
                corpus=source, max_chars=1800, limit=0, correction_model="local-model",
                api_base_url="http://localhost:5000/v1", visolex_checkpoint=root / "checkpoint.pt",
                visolex_tokenizer="local-tokenizer",
            )
            suggestion = {"source": "khum", "replacement": "không", "confidence": 0.9999}
            completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Tôi không biết."))])
            with patch("kg_pipeline.ViSoLexCorrector") as corrector, patch("kg_pipeline.create_client") as create_client:
                corrector.return_value.suggest.return_value = ([suggestion], "suggested")
                create_client.return_value.chat.completions.create.return_value = completion
                self.assertEqual(prepare_corpus(args, root), 1)
                prompt = create_client.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            prepared = json.loads((root / "prepared.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prepared["clean_text"], "Tôi không biết.")
            self.assertEqual(prepared["correction_status"], "accepted")
            self.assertIn('"source": "khum"', prompt)

    def test_text_agents_retry_with_feedback_and_persist_cheatsheet(self):
        with TemporaryDirectory() as directory:
            cheatsheet = Path(directory) / "cheatsheet.json"
            source = "Bài toán có $x^2 = 4$."
            candidates = [
                {"action": "rewrite", "text": "Bài toán có x² = 4.", "answer": None, "reason": "Sửa định dạng"},
                {"action": "keep", "text": source, "answer": None, "reason": "Giữ công thức"},
            ]
            passed = [{"check": name, "passed": True, "feedback": ""} for name in (
                "semantics", "terminology", "solution", "removal",
            )]
            failed = [{"check": "semantics", "passed": False, "feedback": "Restore the original LaTeX formula"}, *passed[1:]]
            rule = {"category": "math", "rule": "Giữ nguyên công thức LaTeX.", "evidence": "Bản được duyệt giữ công thức."}
            with patch("edc.text_agent_pipeline.generate_candidate", side_effect=candidates) as generate, patch(
                "edc.text_agent_pipeline.review_candidate", side_effect=[failed, passed]
            ), patch("edc.text_agent_pipeline.derive_retry_rules", return_value=["Giữ nguyên công thức LaTeX."]) as retry, patch(
                "edc.text_agent_pipeline.derive_rules", return_value=[rule]
            ) as derive:
                workflow = TextAgentWorkflow(object(), "writer", "checker", "rule-agent", cheatsheet, 2, enable_thinking=False)
                result = workflow.process("book::1", source, [])
            self.assertEqual(result["status"], "accepted")
            for call in generate.call_args_list:
                self.assertFalse(call.kwargs["enable_thinking"])
            self.assertFalse(retry.call_args.kwargs["enable_thinking"])
            self.assertFalse(derive.call_args.kwargs["enable_thinking"])
            self.assertEqual(result["text"], source)
            self.assertEqual(generate.call_args_list[0].args[4], [])
            self.assertEqual(generate.call_args_list[1].args[4], ["Giữ nguyên công thức LaTeX."])
            self.assertEqual(generate.call_args_list[0].kwargs["retry_feedback"], [])
            feedback = generate.call_args_list[1].kwargs["retry_feedback"]
            self.assertEqual(feedback[0]["attempt"], 1)
            self.assertEqual(feedback[0]["candidate"], candidates[0])
            self.assertEqual([check["check"] for check in feedback[0]["failed_checks"]], ["semantics", "protected_math"])
            self.assertEqual(feedback[0]["failed_checks"][0]["feedback"], failed[0]["feedback"])
            self.assertEqual(retry.call_args.args[3], candidates[0])
            entries = json.loads(cheatsheet.read_text(encoding="utf-8"))["entries"]
            self.assertEqual([entry["accepted"] for entry in entries], [False, True])
            self.assertEqual(entries[0]["record_id"], "book::1")
            self.assertEqual(entries[0]["retry_rules"], ["Giữ nguyên công thức LaTeX."])
            reloaded = TextAgentWorkflow(object(), "writer", "checker", "rule-agent", cheatsheet, 2)
            self.assertEqual(len(reloaded.entries), 2)
            self.assertEqual(reloaded.rules[0]["rule"], rule["rule"])
            self.assertEqual(derive.call_args.args[3], result["history"][:-1])
            with patch("edc.text_agent_pipeline.generate_candidate", return_value=candidates[1]) as regenerate, patch(
                "edc.text_agent_pipeline.review_candidate", return_value=passed,
            ):
                reloaded.process("book::2", source, [])
            self.assertEqual(regenerate.call_args.args[4], [rule["rule"]])
            self.assertEqual(regenerate.call_args.kwargs["retry_feedback"], [])

    def test_text_agents_append_feedback_log_for_large_book_runs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = "Một đoạn sách."
            candidate = {"action": "keep", "text": source, "answer": None, "explanation": None, "reason": "Đúng"}
            checks = [{"check": name, "passed": True, "feedback": ""} for name in (
                "semantics", "terminology", "solution", "removal",
            )]
            with patch("edc.text_agent_pipeline.generate_candidate", return_value=candidate), patch(
                "edc.text_agent_pipeline.review_candidate", return_value=checks,
            ):
                workflow = TextAgentWorkflow(
                    object(), "writer", "checker", "rule-agent",
                    root / "cheatsheet.json", 2, root / "feedback.jsonl",
                )
                workflow.process("book::1", source, [])
            cheat = json.loads((root / "cheatsheet.json").read_text(encoding="utf-8"))
            self.assertEqual(cheat["entries"], [])
            self.assertEqual(cheat["rules"], [])
            log = [json.loads(line) for line in (root / "feedback.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([entry["record_id"] for entry in log], ["book::1"])
            reloaded = TextAgentWorkflow(
                object(), "writer", "checker", "rule-agent",
                root / "cheatsheet.json", 2, root / "feedback.jsonl",
            )
            self.assertEqual(reloaded.entries, [])

    def test_generator_and_four_review_prompts_use_validated_json(self):
        source = "Hà Nội là thủ đô Việt Nam."
        responses = [json.dumps({"action": "keep", "text": source, "answer": None, "explanation": None, "reason": "Đúng"}, ensure_ascii=False)]
        responses.extend(json.dumps({"passed": True, "feedback": ""}) for _ in range(4))
        completions = Mock()
        completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
            for response in responses
        ]
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        rule = {"category": "math", "rule": "Giữ nguyên LaTeX.", "evidence": "Một lỗi trước đó."}
        retry_feedback = [{
            "attempt": 1,
            "candidate": {"action": "rewrite", "text": "Sai", "answer": None, "explanation": None, "reason": "Sửa"},
            "failed_checks": [{"check": "semantics", "passed": False, "feedback": "Giữ đúng dữ kiện nguồn"}],
        }]
        candidate = generate_candidate(client, "writer", source, [], [rule["rule"]], retry_feedback, enable_thinking=False)
        checks = review_candidate(client, "checker", source, candidate, enable_thinking=False)
        self.assertEqual([check["check"] for check in checks], ["semantics", "terminology", "solution", "removal"])
        request = json.loads(completions.create.call_args_list[0].kwargs["messages"][1]["content"])
        self.assertEqual(request["rules"], [rule["rule"]])
        self.assertEqual(set(request), {"source_text", "visolex_suggestions", "rules", "retry_feedback"})
        self.assertEqual(request["retry_feedback"], retry_feedback)
        generator_format = completions.create.call_args_list[0].kwargs["response_format"]
        self.assertEqual(generator_format["type"], "json_schema")
        schema = generator_format["json_schema"]["schema"]
        self.assertEqual(set(schema["required"]), {"action", "text", "answer", "explanation", "reason"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["answer"]["type"], ["string", "null"])
        for call in completions.create.call_args_list:
            self.assertEqual(call.kwargs["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        for call in completions.create.call_args_list[1:]:
            review_payload = json.loads(call.kwargs["messages"][1]["content"])
            self.assertEqual(review_payload["source_text"], source)
            self.assertEqual(review_payload["candidate"]["text"], candidate["text"])
            self.assertNotIn("reason", review_payload["candidate"])
        prompts = [call.kwargs["messages"][0]["content"] for call in completions.create.call_args_list]
        self.assertEqual(len(set(prompts)), 5)
        for call in completions.create.call_args_list[1:]:
            format_spec = call.kwargs["response_format"]
            self.assertEqual(format_spec["type"], "json_schema")
            self.assertEqual(format_spec["json_schema"]["schema"]["properties"]["passed"], {"type": "boolean"})

    def test_reviewer_incomplete_json_requests_safe_chunk_split(self):
        with self.assertRaisesRegex(OutputTruncatedError, "semantics reviewer returned incomplete JSON"):
            _parse_verdict('{"passed": false, "feedback": "Chưa rõ', "semantics")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            _parse_verdict('{"passed":, "feedback": "Sai"}', "semantics")

        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"passed": true, "feedback": ""}'),
            finish_reason="length",
        )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        candidate = {"action": "keep", "text": "Nguồn", "answer": None, "explanation": None, "reason": "Giữ"}
        with self.assertRaisesRegex(OutputTruncatedError, "finish_reason=length"):
            review_candidate(client, "checker", "Nguồn", candidate)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 2048)

    def test_review_verdict_ignores_unrelated_metadata(self):
        self.assertEqual(
            _parse_verdict('{"passed": false, "feedback": "Sai dữ kiện", "score": 0.2}', "semantics"),
            {"passed": False, "feedback": "Sai dữ kiện"},
        )

    def test_review_verdict_does_not_coerce_string_boolean(self):
        with self.assertRaisesRegex(ValueError, "passed:boolean"):
            _parse_verdict('{"passed": "false", "feedback": "Sai"}', "semantics")
        with self.assertRaisesRegex(ValueError, "feedback:string"):
            _parse_verdict('{"passed": true, "feedback": null}', "semantics")
        with self.assertRaisesRegex(ValueError, "without explaining"):
            _parse_verdict('{"passed": false, "feedback": ""}', "semantics")

    def test_generator_places_supported_answer_and_short_explanation_on_separate_lines(self):
        source = "2 + 2 = ? A. 3 B. 4 C. 5 D. 6"
        response = json.dumps({
            "action": "rewrite", "text": source, "answer": "B",
            "explanation": "Vì 2 + 2 = 4, tương ứng phương án B.", "reason": "Thêm đáp án đã kiểm tra",
        }, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        candidate = generate_candidate(client, "writer", source, [], [])
        self.assertEqual(candidate["text"], source + "\nĐáp án: B\nGiải thích: Vì 2 + 2 = 4, tương ứng phương án B.")

    def test_generator_rejects_answer_without_explanation(self):
        source = "2 + 2 = ? A. 3 B. 4"
        response = json.dumps({
            "action": "rewrite", "text": source, "answer": "B", "explanation": None,
            "reason": "2 + 2 = 4",
        }, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        with self.assertRaisesRegex(ValueError, "explanation"):
            generate_candidate(client, "writer", source, [], [])

    def test_mislabeled_keep_is_reviewed_as_rewrite(self):
        source = "Nguồn có thông tin."
        proposed = "Nguồn có thông tin mới."
        response = json.dumps({"action": "keep", "text": proposed, "answer": None, "explanation": None, "reason": "Giữ"})
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        for passed, status in ((True, "accepted"), (False, "review_rejected")):
            with self.subTest(passed=passed), TemporaryDirectory() as directory:
                workflow = TextAgentWorkflow(client, "writer", "checker", "rules", Path(directory) / "cheatsheet.json", 1)
                with patch("edc.text_agent_pipeline.review_candidate", return_value=[{
                    "check": "semantics", "passed": passed, "feedback": "" if passed else "Bổ sung dữ kiện",
                }]) as review, self.assertLogs("edc.text_generator", level="WARNING") as logs:
                    result = workflow.process("book::2", source, [])
                self.assertEqual(result["status"], status)
                self.assertEqual(review.call_args.args[3]["action"], "rewrite")
                self.assertEqual(review.call_args.args[3]["text"], proposed)
                self.assertEqual(result["history"][0]["candidate"]["action"], "rewrite")
                self.assertEqual(logs.records[0].getMessage(), "generator_action_corrected")

    def test_unchanged_rewrite_becomes_keep_and_is_still_reviewed(self):
        cases = [
            ("Nguồn không có lỗi.", None, None),
            ("2 + 2? A. 3 B. 4\nĐáp án: B\nGiải thích: 2 + 2 = 4.", "B", "2 + 2 = 4."),
        ]
        for source, answer, explanation in cases:
            with self.subTest(answer=answer), TemporaryDirectory() as directory:
                response = json.dumps({"action": "rewrite", "text": source, "answer": answer,
                                       "explanation": explanation, "reason": "Đã kiểm tra"})
                completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
                client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
                workflow = TextAgentWorkflow(client, "writer", "checker", "rules", Path(directory) / "cheatsheet.json", 1)
                with patch("edc.text_agent_pipeline.review_candidate", return_value=[{
                    "check": "semantics", "passed": True, "feedback": "",
                }]) as review, self.assertLogs("edc.text_generator", level="WARNING") as logs:
                    result = workflow.process("book::35", source, [])
                self.assertEqual(result["status"], "accepted")
                self.assertEqual(result["text"], source)
                self.assertEqual(result["action"], "keep")
                reviewed = review.call_args.args[3]
                self.assertEqual(reviewed["action"], "keep")
                self.assertIsNone(reviewed["answer"])
                self.assertIsNone(reviewed["explanation"])
                self.assertEqual(logs.records[0].details["effective_action"], "keep")

    def test_generator_reports_truncation_before_json_parsing(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"text": "Private unfinished'), finish_reason="length",
        )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        with self.assertRaisesRegex(ValueError, "finish_reason=length") as raised:
            generate_candidate(client, "writer", "Question", [], [], max_tokens=16384)
        self.assertNotIn("Private", str(raised.exception))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 16384)

    def test_generator_reports_missing_fields_without_response_contents(self):
        response = json.dumps({
            "action": "rewrite", "text": "Private source", "answer": "B",
            "reason": "Private reason", "Private extra field": "secret",
        })
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        with self.assertRaises(ValueError) as raised:
            generate_candidate(client, "writer", "Question", [], [])
        self.assertIn("missing fields: ['explanation']", str(raised.exception))
        self.assertIn("unexpected field count: 1", str(raised.exception))
        self.assertNotIn("Private", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_generator_preserves_separate_answers_for_multiple_questions(self):
        source = "Câu 1: 2 + 2? A. 4 B. 5\n\nCâu 2: 1 + 1? A. 1 B. 2"
        first = "Câu 1: 2 + 2? A. 4 B. 5\nĐáp án: A\nGiải thích: 2 + 2 = 4."
        body = first + "\n\nCâu 2: 1 + 1? A. 1 B. 2"
        suffix = "\nĐáp án: B\nGiải thích: 1 + 1 = 2."
        for text in (body, body + suffix, body + suffix + suffix):
            candidate = _parse_candidate(json.dumps({
                "action": "rewrite", "text": text, "answer": "B", "explanation": "1 + 1 = 2.", "reason": "Thêm đáp án",
            }), source)
            self.assertEqual(candidate["text"], body + suffix)

    def test_generator_does_not_duplicate_existing_answer_and_explanation_lines(self):
        source = "2 + 2 = ? A. 3 B. 4"
        explanation = "Vì 2 + 2 = 4."
        expected = source + "\nĐáp án: B\nGiải thích: " + explanation
        response = json.dumps({
            "action": "rewrite", "text": expected, "answer": "B",
            "explanation": explanation, "reason": "Đã tính lại",
        }, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        candidate = generate_candidate(client, "writer", source, [], [])
        self.assertEqual(candidate["text"], expected)

    def test_generator_uses_structured_explanation_when_text_differs(self):
        source = "2 + 2 = ? A. 3 B. 4"
        response = json.dumps({
            "action": "rewrite", "text": source + "\nĐáp án: B\nGiải thích: Tổng bằng bốn.",
            "answer": "B", "explanation": "Vì 2 + 2 = 4.", "reason": "Đã tính lại",
        }, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        candidate = generate_candidate(client, "writer", source, [], [])
        self.assertEqual(candidate["text"], source + "\nĐáp án: B\nGiải thích: Vì 2 + 2 = 4.")

    def test_rule_agent_keeps_only_rules_about_confirmed_feedback(self):
        source = "Có công thức $x^2$."
        rejected = [{
            "accepted": False,
            "candidate": {"action": "rewrite", "text": "Có công thức x².", "answer": None, "reason": "Đổi dạng"},
            "checks": [{"check": "protected_math", "passed": False, "feedback": "Restore $x^2$"}],
        }]
        accepted = {"action": "keep", "text": source, "answer": None, "reason": "Giữ công thức"}
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock()))
        rules = derive_rules(client, "rule-agent", source, rejected, accepted, [])
        self.assertEqual([rule["category"] for rule in rules], ["math"])
        self.assertIn("LaTeX", rules[0]["rule"])
        client.chat.completions.create.assert_not_called()

    def test_retry_rule_agent_restores_missing_latex_without_raw_feedback_to_writer(self):
        source = "Có công thức $x^2$."
        candidate = {"action": "rewrite", "text": "Có công thức x².", "answer": None}
        checks = [{"check": "protected_math", "passed": False, "feedback": "Restore $x^2$"}]
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock()))
        rules = derive_retry_rules(client, "rule-agent", source, candidate, checks, [])
        self.assertEqual(len(rules), 1)
        self.assertIn("LaTeX", rules[0])
        client.chat.completions.create.assert_not_called()

    def test_retry_rule_agent_discards_instruction_to_remove_answer(self):
        source = "2 + 2 = ? A. 3 B. 4"
        candidate = {"action": "rewrite", "text": source + "\nĐáp án: B", "answer": "B"}
        checks = [{"check": "semantics", "passed": False, "feedback": "Remove the answer"}]
        response = json.dumps({"rules": [{
            "category": "semantics", "rule": "Xóa dòng đáp án.",
            "evidence": "Reviewer yêu cầu xóa đáp án.",
        }]}, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        self.assertEqual(derive_retry_rules(client, "rule-agent", source, candidate, checks, []), [])

    def test_retry_rule_agent_converts_current_feedback_to_one_rule(self):
        source = "khí hiếm Ne"
        candidate = {"action": "rewrite", "text": "khí hiểm Ne", "answer": None}
        checks = [{"check": "terminology", "passed": False, "feedback": "Giữ thuật ngữ khí hiếm."}]
        response = json.dumps({"rules": [{
            "category": "terminology", "rule": "Giữ thuật ngữ khí hiếm khi sửa OCR.",
            "evidence": "Nguồn có khí hiếm; bản nháp đổi thành khí hiểm.",
        }]}, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        rules = derive_retry_rules(client, "rule-agent", source, candidate, checks, [])
        self.assertEqual(rules, ["Giữ thuật ngữ khí hiếm khi sửa OCR."])
        self.assertEqual(client.chat.completions.create.call_args.kwargs["response_format"]["type"], "json_schema")
        request = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(request["failed_checks"], [{"check": "terminology", "feedback": "Giữ thuật ngữ khí hiếm."}])

    def test_rule_agent_discards_categories_not_in_failed_checks(self):
        source = "khí hiểm Ne"
        rejected = [{
            "accepted": False,
            "candidate": {"action": "rewrite", "text": "khí nguy hiểm Ne", "answer": None, "reason": "Sửa thuật ngữ"},
            "checks": [{"check": "terminology", "passed": False, "feedback": "Use khí hiếm"}],
        }]
        accepted = {"action": "rewrite", "text": "khí hiếm Ne", "answer": None, "reason": "Sửa OCR"}
        response = json.dumps({"rules": [
            {"category": "terminology", "rule": "Dựa vào ngữ cảnh để sửa thuật ngữ OCR.", "evidence": "Bản cuối dùng khí hiếm."},
            {"category": "removal", "rule": "Bỏ tiêu đề ở cuối đoạn.", "evidence": "Một tiêu đề bị bỏ."},
        ]}, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        rules = derive_rules(client, "rule-agent", source, rejected, accepted, [])
        self.assertEqual([rule["category"] for rule in rules], ["terminology"])
        format_spec = client.chat.completions.create.call_args.kwargs["response_format"]
        self.assertEqual(format_spec["type"], "json_schema")
        properties = format_spec["json_schema"]["schema"]["properties"]["rules"]["items"]["properties"]
        self.assertNotIn("maxLength", properties["rule"])
        self.assertNotIn("maxLength", properties["evidence"])
        self.assertIn("pattern", properties["rule"])

        request = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(request["allowed_categories"], ["spelling", "terminology"])

    def test_rule_validation_reports_safe_field_details(self):
        for value in ("Private\ncontent", " padded ", ""):
            with self.subTest(value_length=len(value)):
                response = json.dumps({"rules": [{
                    "category": "semantics", "rule": value, "evidence": "Bằng chứng.",
                }]})
                with self.assertRaises(ValueError) as raised:
                    _parse_rules(response, False, set(), {"semantics"})
                self.assertIn("0.rule", str(raised.exception))
                self.assertIn(f"length={len(value)}", str(raised.exception))
                self.assertNotIn("Private", str(raised.exception))
        self.assertEqual(_parse_rules('{"rules": []}', False, set(), {"semantics"}), [])

    def test_rule_agent_preserves_long_rules_without_truncation(self):
        rule = "Luôn đối chiếu bản sửa với văn bản nguồn và giữ đầy đủ điều kiện của câu hỏi. " * 7
        rule = rule.strip()
        self.assertGreater(len(rule), 476)
        payload = {"rules": [{"category": "semantics", "rule": rule, "evidence": "Bản sửa đã giữ đúng dữ kiện."}]}
        parsed = _parse_rules(json.dumps(payload, ensure_ascii=False), False, set(), {"semantics"})
        self.assertEqual(parsed, payload["rules"])
        self.assertEqual(_parse_rules(json.dumps(payload), False, {("semantics", rule.casefold())}, {"semantics"}), [])

    def test_rule_agent_preserves_long_evidence_in_persistent_cheatsheet(self):
        evidence = "Bằng chứng cần đối chiếu. " * 20 + "\nChi tiết tiếp theo."
        source = "khí hiểm Ne"
        first = {"action": "rewrite", "text": "khí nguy hiểm Ne", "answer": None, "reason": "Sửa"}
        accepted = {"action": "rewrite", "text": "khí hiếm Ne", "answer": None, "reason": "Sửa OCR"}
        response = json.dumps({"rules": [{
            "category": "terminology", "rule": "Giữ thuật ngữ khí hiếm.", "evidence": evidence,
        }]}, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        failed = [{"check": "terminology", "passed": False, "feedback": "Giữ thuật ngữ"}]
        passed = [{"check": "terminology", "passed": True, "feedback": ""}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cheatsheet.json"
            workflow = TextAgentWorkflow(client, "writer", "checker", "rules", path, 2)
            with patch("edc.text_agent_pipeline.generate_candidate", side_effect=[first, accepted]), patch(
                "edc.text_agent_pipeline.review_candidate", side_effect=[failed, passed]
            ), patch("edc.text_agent_pipeline.derive_retry_rules", return_value=[]):
                result = workflow.process("book::19", source, [])
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(json.loads(path.read_text())["rules"][0]["evidence"], evidence)
        with self.assertRaisesRegex(ValueError, "nonempty evidence"):
            _parse_rules(json.dumps({"rules": [{"category": "semantics", "rule": "Giữ nghĩa.", "evidence": " "}]}), False, set(), {"semantics"})

    def test_rule_agent_does_not_learn_to_delete_accepted_answer(self):
        source = "2 + 2 = ? A. 3 B. 4"
        rejected = [{
            "accepted": False,
            "candidate": {"action": "rewrite", "text": source, "answer": None, "reason": "Chưa trả lời"},
            "checks": [{"check": "semantics", "passed": False, "feedback": "Remove the answer"}],
        }]
        accepted = {"action": "rewrite", "text": source + "\nĐáp án: B", "answer": "B", "reason": "2 + 2 = 4"}
        response = json.dumps({"rules": [
            {"category": "semantics", "rule": "Xóa dòng đáp án sau câu hỏi.", "evidence": "Reviewer từng yêu cầu."},
        ]}, ensure_ascii=False)
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=Mock(create=Mock(return_value=completion))))
        self.assertEqual(derive_rules(client, "rule-agent", source, rejected, accepted, []), [])

    def test_text_agent_drop_is_excluded_from_edc_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"id": "noise", "text": "zzx ### qqq"}) + "\n", encoding="utf-8")
            args = Namespace(
                corpus=source, max_chars=1800, limit=0, correction_model=None,
                text_agent_model="writer", review_model="checker", agent_rounds=2,
                cheatsheet_path=None, api_base_url="http://localhost:5000/v1",
                visolex_checkpoint=None, visolex_tokenizer="unused",
            )
            with patch("kg_pipeline.create_client"), patch(
                "edc.text_agent_pipeline.generate_candidate",
                return_value={"action": "drop", "text": "", "answer": None, "reason": "OCR vô nghĩa"},
            ), patch("edc.text_agent_pipeline.review_candidate", return_value=[
                {"check": name, "passed": True, "feedback": ""}
                for name in ("semantics", "terminology", "solution", "removal")
            ]):
                self.assertEqual(prepare_corpus(args, root), 0)
            prepared = json.loads((root / "prepared.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prepared["status"], "skipped_dropped")
            self.assertEqual((root / "edc_input.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(len(json.loads((root / "cheatsheet.json").read_text(encoding="utf-8"))["entries"]), 1)

    def test_text_agent_rejection_never_reaches_edc(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"id": "question", "text": "2 + 2 = ?"}) + "\n", encoding="utf-8")
            args = Namespace(
                corpus=source, max_chars=1800, limit=0, correction_model=None,
                text_agent_model="writer", review_model="checker", agent_rounds=2,
                cheatsheet_path=None, api_base_url="http://localhost:5000/v1",
                visolex_checkpoint=None, visolex_tokenizer="unused",
            )
            with patch("kg_pipeline.create_client"), patch(
                "edc.text_agent_pipeline.generate_candidate",
                return_value={"action": "rewrite", "text": "2 + 2 = 5", "answer": None, "reason": "Sai"},
            ), patch("edc.text_agent_pipeline.review_candidate", return_value=[
                {"check": "semantics", "passed": False, "feedback": "Incorrect arithmetic"},
                {"check": "terminology", "passed": True, "feedback": ""},
                {"check": "solution", "passed": True, "feedback": ""},
                {"check": "removal", "passed": True, "feedback": ""},
            ]), patch("edc.text_agent_pipeline.derive_retry_rules", return_value=["Kiểm tra lại phép tính."]):
                self.assertEqual(prepare_corpus(args, root), 0)
            prepared = json.loads((root / "prepared.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(prepared["status"], "skipped_review_rejected")
            self.assertEqual(len(prepared["agent_history"]), 2)
            self.assertEqual((root / "edc_input.txt").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
