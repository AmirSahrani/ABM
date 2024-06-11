from agents import Nomad
import numpy as np


def fighting_game(agent1: Nomad, agent2: Nomad):
    total_spice = agent1.spice + agent2.spice
    return np.array(
        [
            [           # Game player 1
                [agent1.spice, 0],
                [total_spice, 0.8 * total_spice]
            ],
            [           # Game player 2
                [agent2.spice, total_spice],
                [0, 0.8.total_spice]
            ]
        ]
    )


def pure_eq(game: np.ndarray):
    best_response = np.argmax(game, axis=1), np.argmax(game, axis=2)
    return game[0, best_response[0], best_response[1]], game[1, best_response[0], best_response[1]]