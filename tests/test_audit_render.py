from __future__ import annotations

import unittest

from m3_edu_memory.audit_render import _latex_to_display_math


class AuditRenderTests(unittest.TestCase):
    def test_inline_limit_remains_readable(self):
        rendered = _latex_to_display_math(
            r"应用$\lim_{x\to1}\frac{x^{15}-1}{x-1}=15\cdot1^{14}$"
        )

        self.assertIn("lim x→1", rendered)
        self.assertIn("x¹⁵", rendered)
        self.assertIn("15·1¹⁴", rendered)
        self.assertNotIn("lim_x", rendered)
        self.assertNotIn("\\", rendered)

    def test_letter_exponents_use_portable_notation(self):
        rendered = _latex_to_display_math(r"n\cdot1^{n-1}")

        self.assertEqual(rendered, "n·1^(n−1)")
        self.assertNotIn("ⁿ", rendered)


if __name__ == "__main__":
    unittest.main()
