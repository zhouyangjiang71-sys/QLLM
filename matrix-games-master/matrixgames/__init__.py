from gymnasium import register
import numpy as np

for k in (0, 25, 50, 75, 100):
    _payoff = np.array(
        2
        * [
            [[-k, 0, 10], [0, 2, 0], [10, 0, -k]],
        ]
    )
    register(
        f"penalty-{k}-nostate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": False,
        },
    )
    register(
        f"penalty-{k}-laststate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": True,
        },
    )

_payoff_trivial = np.array(
    2
    * [
        [
            [11, 8, 0],
            [8, 7, 0],
            [0, 0, 0],
        ]
    ]
)
register(
    f"trivial-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_trivial,
        "ep_length": 1000,
        "last_action_state": False,
    },
)
_payoff_linear = np.array(
    2
    * [
        [
            [1, 4, 2],
            [0, 3, 1],
            [2, 5, 3],
        ]
    ]
)
register(
    f"linear-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_linear,
        "ep_length": 1000,
        "last_action_state": False,
    },
)

    # +0 | +3 | +1
# -----------------
# +1 | 1 |  4 |  2 
# +0 | 0 |  3 |  1
# +2 | 2 |  5 |  3

_payoff_non_linear = np.array(
    2
    * [
        [
            [0, 3, 1],
            [0, 0, 0],
            [0, 6, 2],
        ]
    ]
)
register(
    f"nonlinear-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_non_linear,
        "ep_length": 1000,
        "last_action_state": False,
    },
)


    # +0 | +3
# -----------
# +1 | 1 |  4
# +2 | 2 |  5

    #  +0 | +3
# -------------
# +1 |  0 |  3
# +2 |  0 |  6
_payoff_linear_2p = np.array(
    2
    * [
        [
            [1, 5],
            [5, 9],
        ]
    ]
)
register(
    f"linear-2p-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_linear_2p,
        "ep_length": 1000,
        "last_action_state": False,
    },
)

_payoff_nonlinear_2p = np.array(
    2
    * [
        [
            [0, 0],
            [0, 10],
        ]
    ]
)
register(
    f"nonlinear-2p-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_nonlinear_2p,
        "ep_length": 1000,
        "last_action_state": False,
    },
)


register(
    f"twostep-2p-v0",
    entry_point="matrixgames.games:TwoStep",
    kwargs={
        "payoff_matrix1": _payoff_linear_2p,
        "payoff_matrix2": _payoff_nonlinear_2p,
    },
)

_payoff_climbing = np.array(
    2
    * [
        [
            [11, -30, 0],
            [-30, 7, 0],
            [0, 6, 5],
        ]
    ]
)
register(
    f"climbing-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1000,
        "last_action_state": False,
    },
)

register(
    f"climbing-laststate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing,
        "ep_length": 1000,
        "last_action_state": True,
    },
)

_payoff_pdilemma = np.array(
    [
        [
            [-1, -3],
            [-3, -2],
        ],
        [
            [-1, -3],
            [-3, -2],
        ],
    ]
)

register(
    f"pdilemma-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_pdilemma,
        "ep_length": 1000,
        "last_action_state": False,
    },
)

register(
    f"pdilemma-laststate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_pdilemma,
        "ep_length": 1000,
        "last_action_state": True,
    },
)


_payoff_climbing_3_players = np.array(
    3
    * [
        [
            [[11, -30, 0], [-30, 0, 0], [0, 0, 0]],
            [
                [-30, 0, 0],
                [0, 7, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 6, 5],
            ],
        ]
    ]
)

register(
    f"climbing-3p-nostate-v0",
    entry_point="matrixgames.games:MatrixGame",
    kwargs={
        "payoff_matrix": _payoff_climbing_3_players,
        "ep_length": 1000,
        "last_action_state": False,
    },
)


import matrixgames.aamas2012

for i in range(1, 22):
    _arr = getattr(matrixgames.aamas2012, f"nc{i}")
    _payoff = np.array([
        [[_arr[0], _arr[2]], [_arr[4], _arr[6]]],
        [[_arr[1], _arr[3]], [_arr[5], _arr[7]]],
    ])
    register(
        f"nonconflict-{i}-nostate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": False,
        }, 
    )

for i in range(1, 58):
    _arr = getattr(matrixgames.aamas2012, f"c{i}")
    _payoff = np.array([
        [[_arr[0], _arr[2]], [_arr[4], _arr[6]]],
        [[_arr[1], _arr[3]], [_arr[5], _arr[7]]],
    ])
    register(
        f"conflict-{i}-nostate-v0",
        entry_point="matrixgames.games:MatrixGame",
        kwargs={
            "payoff_matrix": _payoff,
            "ep_length": 1000,
            "last_action_state": False,
        }, 
    )
