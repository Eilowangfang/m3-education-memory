import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m3_edu_memory.error_analysis import analyze_errors


def _attempt(attempt_id, *, wrong, error_type=None, point="一元方程", reviewed=False):
    return {
        "attempt_id": attempt_id,
        "Time": "2026-09-10T12:00:00+08:00",
        "domain_code": "alg",
        "has_error_effective": int(wrong),
        "error_type_effective": error_type,
        "knowledge_points": [point],
        "first_error_step_effective": 0 if wrong else None,
        "steps": [{"step_index": 0, "transcription": "x+1=3", "bbox": [0, 0, 1, 1]}],
        "error_bbox_effective": [0, 0, 1, 1] if wrong else None,
        "question_transcription": "解方程",
        "orig_q": "解方程",
        "error_explanation": "移项出错",
        "confidence": 0.8,
        "display_source": "teacher_override" if reviewed else "vlm",
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_full_cohort_counts_and_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            db = Path(root) / "memory.db"
            connection = sqlite3.connect(db)
            connection.execute("CREATE TABLE attempts(domain_code TEXT, Time TEXT)")
            connection.executemany(
                "INSERT INTO attempts VALUES (?,?)",
                [("alg", "2026-09-10T12:00:00+08:00")] * 4,
            )
            connection.commit()
            connection.close()
            diagnosed = [
                _attempt("one", wrong=True, error_type="arithmetic"),
                _attempt("two", wrong=True, error_type="notation", reviewed=True),
                _attempt("three", wrong=False),
            ]
            with patch("m3_edu_memory.error_analysis.resolve_recent_vlm_window",
                       return_value=("2026-07-01T00:00:00+08:00", "2026-09-23T23:59:59+08:00")), \
                 patch("m3_edu_memory.error_analysis.query_vlm_memories",
                       return_value={"attempts": diagnosed}):
                report = analyze_errors(db, request="分析过去3个月的代数错题", model="seed")

        self.assertEqual(report["coverage"]["cohort_attempts"], 4)
        self.assertEqual(report["coverage"]["missing_diagnosis"], 1)
        self.assertEqual(report["coverage"]["error_attempts"], 2)
        self.assertEqual(sum(item["count"] for item in report["categories"]), 2)
        self.assertEqual(sum(item["count"] for item in report["domains"]), 2)
        self.assertEqual(report["domains"][0]["attempt_count"], 3)
        self.assertEqual(report["domains"][0]["model_error_rate"], 0.6667)
        summary = report["frequency_summary"]
        self.assertLessEqual(len(summary), 500)
        self.assertLess(summary.index("分析："), summary.index("建议："))
        self.assertIn("计算错误1道", summary)
        self.assertIn("各方向作答量不同", summary)
        self.assertIn("④隔几天重新独立完成", summary)
        self.assertTrue(summary.endswith("。"))
        for category in report["categories"]:
            self.assertEqual(
                sum(domain["count"] for domain in category["domains"]),
                category["count"],
            )
        self.assertEqual(report["weaknesses"][0]["attempt_count"], 3)
        self.assertEqual(report["weaknesses"][0]["error_rate"], 0.6667)
        self.assertEqual({item["attempt_id"] for item in report["cases"]}, {"one", "two"})
        self.assertEqual(
            next(item for item in report["cases"] if item["attempt_id"] == "two")["display_source"],
            "teacher_override",
        )


if __name__ == "__main__":
    unittest.main()
