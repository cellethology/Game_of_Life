"""Conway's Game of Life simulation utilities."""

import numpy as np
from typing import Tuple


def random_board(size: int, p_alive: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a random (size x size) 0/1 board with Bernoulli(p_alive)."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.random((size, size)) < p_alive


def count_neighbors(board: np.ndarray) -> np.ndarray:
    """Count living neighbors for each cell using toroidal boundary conditions."""
    # Use np.roll to shift the board in all 8 directions
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    neighbor_count = np.zeros_like(board, dtype=int)
    for dx, dy in shifts:
        neighbor_count += np.roll(np.roll(board, dx, axis=0), dy, axis=1)

    return neighbor_count


def step(board: np.ndarray) -> np.ndarray:
    """Compute one Game of Life step on a toroidal grid and return the new board."""
    neighbor_count = count_neighbors(board)

    # Game of Life rules:
    # 1. Any live cell with 2 or 3 live neighbors survives
    # 2. Any dead cell with exactly 3 live neighbors becomes alive
    # 3. All other cells die or remain dead

    # Rule 1: Cell survives if it was alive and has 2 or 3 neighbors
    survives = (board == 1) & ((neighbor_count == 2) | (neighbor_count == 3))

    # Rule 2: Cell is born if it was dead and has exactly 3 neighbors
    born = (board == 0) & (neighbor_count == 3)

    # Apply rules
    new_board = (survives | born).astype(np.uint8)

    return new_board


def run_trajectory(
    num_boards: int,
    size: int,
    num_steps: int,
    p_alive: float = 0.5,
    burn_in: int = 50,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate trajectories for multiple independent boards.

    Returns an array of shape (num_boards, T, size, size),
    where T = num_steps (post burn-in).

    The function should:
    - For each board:
      - Sample a random initial board.
      - Run for `burn_in` steps but do NOT store those.
      - Then run for `num_steps` more steps, storing each time step.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize storage
    trajectories = np.zeros((num_boards, num_steps, size, size), dtype=np.uint8)

    for i in range(num_boards):
        # Create initial random board
        board = random_board(size, p_alive, rng)

        # Run burn-in steps
        for _ in range(burn_in):
            board = step(board)

        # Run and store num_steps
        for t in range(num_steps):
            board = step(board)
            trajectories[i, t] = board

    return trajectories