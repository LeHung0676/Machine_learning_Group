import numpy as np
from src.evaluation.Evaluation import Evaluator


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    m = Evaluator.regression_metrics(y, y)
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1
