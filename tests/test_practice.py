import random
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from m3_edu_memory.practice import _make_question, _template, generate_practice, requested_count
from m3_edu_memory.server import MemoryApi


REPORT = {
    "request": "帮我收集分析过去3个月的错题",
    "categories": [
        {"error_type": "algebraic_manipulation", "label": "代数变形错误", "count": 8,
         "top_knowledge_points": [{"name": "链式法则", "error_count": 5}]},
        {"error_type": "arithmetic", "label": "计算错误", "count": 6,
         "top_knowledge_points": [{"name": "一元一次方程", "error_count": 4}]},
    ],
    "cases": [
        {"attempt_id": "old-1", "error_type": "algebraic_manipulation",
         "domain_code": "clc", "domain_label": "微积分",
         "knowledge_points": ["链式法则"], "question": "求旧函数的导数"},
        {"attempt_id": "old-2", "error_type": "arithmetic",
         "domain_code": "alg", "domain_label": "代数",
         "knowledge_points": ["一元一次方程"], "question": "解旧方程"},
    ],
}


class PracticeTests(unittest.TestCase):
    def test_five_new_questions_are_evidence_linked_and_repeatable(self):
        request = "根据我的错题分析记录再出5题来练手巩固"
        first = generate_practice(REPORT, request=request)
        second = generate_practice(REPORT, request=request)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 5)
        self.assertEqual(len({item["question"] for item in first["questions"]}), 5)
        self.assertEqual({item["source_attempt_id"] for item in first["questions"]},
                         {"old-1", "old-2"})
        self.assertEqual(first["questions"][0]["knowledge_point"], "链式法则")
        self.assertEqual(first["questions"][0]["related_error_count"], 5)
        self.assertTrue(all(item["question"] not in {"求旧函数的导数", "解旧方程"}
                            for item in first["questions"]))

    def test_count_validation_and_missing_evidence(self):
        self.assertEqual(requested_count("再出五题"), 5)
        self.assertEqual(requested_count("出题练手"), 5)
        with self.assertRaises(ValueError):
            requested_count("再出11题")
        with self.assertRaises(ValueError):
            requested_count("再出101题")
        with self.assertRaises(ValueError):
            generate_practice({"request": "空", "cases": [], "categories": []},
                              request="再出5题")

    def test_linear_equations_have_a_unique_solution(self):
        for seed in range(30):
            question = _make_question("linear_equation", random.Random(seed), seed)
            match = re.search(r"\\\((\d+)\(x\+(\d+)\)=(\d+)x([+-]\d+)\\\)", question)
            self.assertIsNotNone(match)
            self.assertNotEqual(int(match.group(1)), int(match.group(3)))

    def test_geometry_distance_variants_match_the_knowledge_point(self):
        self.assertEqual(_template("点到直线的距离公式", "mgm"), "point_line_distance")
        self.assertEqual(_template("空间两点间距离公式", "mgm"), "distance_3d")

    def test_api_uses_current_analysis_and_rejects_excessive_count(self):
        api = MemoryApi(Path("unused.db"), default_model="seed")
        with patch("m3_edu_memory.server.analyze_errors", return_value=REPORT) as analyze:
            status, payload = api.dispatch(
                "POST", "/v1/memory/practice",
                {"analysis_request": REPORT["request"], "request": "再出5题"},
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(payload["questions"]), 5)
            analyze.assert_called_once()
            status, payload = api.dispatch(
                "POST", "/v1/memory/practice",
                {"analysis_request": REPORT["request"], "request": "再出11题"},
            )
            self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
