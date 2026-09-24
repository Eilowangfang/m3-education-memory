from __future__ import annotations

import unittest

from m3_edu_memory.knowledge import (
    canonicalize_knowledge_point,
    canonicalize_knowledge_points,
)


class KnowledgeNormalizationTests(unittest.TestCase):
    def test_chinese_and_english_aliases_share_one_curriculum_label(self):
        self.assertEqual(canonicalize_knowledge_point("chain rule"), "链式法则")
        self.assertEqual(
            canonicalize_knowledge_point("复合函数求导时的链式法则"), "链式法则"
        )

    def test_aliases_are_deduplicated_without_erasing_unknown_points(self):
        self.assertEqual(
            canonicalize_knowledge_points(
                ["Matrix Multiplication", "矩阵乘法", "特殊矩阵性质"],
                domain_code="alg",
            ),
            ["矩阵乘法", "特殊矩阵性质"],
        )


if __name__ == "__main__":
    unittest.main()
