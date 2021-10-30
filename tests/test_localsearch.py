import numpy as np
from pyTTP import objective
from pyTTP.localsearch import satisfies_home_away_stands

D_nl4 = np.array([[0,  745,  665,  929],
                  [745,   0,   80,  337],
                  [665,  80,   0,  380],
                  [929, 337,  380,    0]], dtype=np.int32)
sol_nl4 = np.array([[-4, -2, -3,  2,  4,  3],
                    [3,  1, -4, -1, -3,  4],
                    [-2, -4,  1,  4,  2, -1],
                    [1,  3,  2, -3, -1, -2]], dtype=np.int32)


def test_objective_nl4():
    assert objective(sol_nl4, D_nl4) == 8276


def test_home_away_stands_nl4():
    assert satisfies_home_away_stands(sol_nl4, 3) is True
    assert satisfies_home_away_stands(sol_nl4, 2) is False
