"""Grounding + refusal eval for CiteRAG.

These make the core promise testable: in-scope questions are answered from the
right document with a citation, and out-of-scope questions are refused rather
than answered with a confident guess.

Run from the project root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.corpus import load_chunks            # noqa: E402
from app.index import TfidfIndex              # noqa: E402
from app.service import answer_question       # noqa: E402


class CiteRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = TfidfIndex().fit(load_chunks())

    def test_corpus_chunked(self):
        self.assertGreaterEqual(len(self.index.chunks), 8)

    def test_return_window_answer(self):
        res = answer_question("What is the return window for items?", self.index)
        self.assertFalse(res["refused"])
        self.assertIn("30 days", res["answer"])
        self.assertEqual(res["sources"][0]["doc"], "Returns & Refund Policy")

    def test_restocking_fee_answer(self):
        res = answer_question("Is there a restocking fee?", self.index)
        self.assertFalse(res["refused"])
        self.assertIn("15%", res["answer"])

    def test_shipping_free_threshold(self):
        res = answer_question("When is shipping free?", self.index)
        self.assertFalse(res["refused"])
        self.assertIn("$50", res["answer"])

    def test_every_in_scope_sample_is_answered_with_citation(self):
        in_scope = [
            "What is the return window for items?",
            "Is there a restocking fee?",
            "Can I return clearance items?",
            "When is shipping free?",
            "What does the warranty cover?",
        ]
        for q in in_scope:
            res = answer_question(q, self.index)
            self.assertFalse(res["refused"], f"unexpectedly refused: {q}")
            self.assertTrue(res["sources"], f"no sources cited: {q}")

    def test_out_of_scope_is_refused(self):
        for q in [
            "What is the company's annual revenue?",
            "What is the capital of France?",
            "How do I bake chocolate chip cookies?",
        ]:
            res = answer_question(q, self.index)
            self.assertTrue(res["refused"], f"should have refused: {q}")

    def test_refusal_text_has_no_fabricated_citation(self):
        res = answer_question("What is the weather forecast for tomorrow?", self.index)
        self.assertTrue(res["refused"])
        self.assertIn("couldn't find", res["answer"].lower())


if __name__ == "__main__":
    unittest.main()
