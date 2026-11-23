"""Check the confusion-matrix solver against cases verified by hand."""
from implied_confusion import solve


def test_spanish_row_has_unique_solution():
    """Paper Table 10, Spanish: 97.00 / 97.55 / 96.37 -> P=248, TP=239, FP=6, TN=246, FN=9."""
    assert solve(97.00, 97.55, 96.37) == [(248, 239, 6, 246, 9)]


def test_chinese_paper_row_implies_37_errors():
    """Paper Table 10, Chinese: 92.60 / 92.71 / 92.34 -> FN=19, FP=18 (37 errors)."""
    sol = solve(92.60, 92.71, 92.34)
    assert sol == [(248, 229, 18, 234, 19)]
