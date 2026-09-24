import unittest

from m3_edu_memory.time_simulation import GRADE_TO_MONTH_OFFSET, simulated_time


class TimeSimulationTests(unittest.TestCase):
    def test_all_grades_map_to_seven_months(self):
        self.assertEqual(GRADE_TO_MONTH_OFFSET, {
            "c07": -6,
            "c08": -5,
            "c09": -4,
            "c10": -3,
            "c11": -2,
            "c12": -1,
            "c00": 0,
        })


    def test_time_is_deterministic_and_within_bucket(self):
        kwargs = dict(
            grade="c10",
            dataset_revision="revision",
            shard="train-00000.parquet",
            row_index=12,
            anchor_date="2026-09-23",
        )
        first = simulated_time(**kwargs)
        second = simulated_time(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first.year, 2026)
        self.assertEqual(first.month, 6)
        self.assertIsNotNone(first.tzinfo)


    def test_different_rows_receive_different_time(self):
        common = dict(
            grade="c07",
            dataset_revision="revision",
            shard="train-00000.parquet",
            anchor_date="2026-09-23",
        )
        self.assertNotEqual(
            simulated_time(row_index=1, **common),
            simulated_time(row_index=2, **common),
        )

    def test_current_month_never_exceeds_anchor_day(self):
        value = simulated_time(
            grade="c00", dataset_revision="revision",
            shard="train-00000.parquet", row_index=99,
            anchor_date="2026-09-23",
        )
        self.assertEqual((value.year, value.month), (2026, 9))
        self.assertLessEqual(value.day, 23)


if __name__ == "__main__":
    unittest.main()
