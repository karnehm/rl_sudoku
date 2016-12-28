"""
Some terribly hacky implementations of sudoku generation code.
"""

import random

import numpy as np

import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import sudoku_maker


##############################################################################
# 2x2 sudoku stuff
##############################################################################

BASE1 = np.array(
    [[1,2,3,4],
     [3,4,1,2],
     [2,1,4,3],
     [4,3,2,1]])

BASE2 = np.array(
    [[1,2,3,4],
     [3,4,1,2],
     [2,3,4,1],
     [4,1,2,3]])

BASE3 = np.array(
    [[1,2,3,4],
     [3,4,2,1],
     [2,1,4,3],
     [4,3,1,2]])

BASES = [BASE1, BASE2, BASE3]

##############################################################################


SUDOKU_SIZE = 9


def flatten(grid):
    mask = (grid.ravel() + SUDOKU_SIZE * np.arange(SUDOKU_SIZE**2) - 1)[(np.where(grid.ravel() != 0))]
    flat = np.zeros(SUDOKU_SIZE**3)
    flat[mask] = 1
    return flat.astype(int)


def unflatten(grid):
    return (grid.reshape((SUDOKU_SIZE, SUDOKU_SIZE, SUDOKU_SIZE)).argmax(axis=2) + 1) * \
            grid.reshape((SUDOKU_SIZE, SUDOKU_SIZE, SUDOKU_SIZE)).max(axis=2).astype(int)


def _swap_cols(grid, col1, col2):
    grid[:, col1], grid[:, col2] = grid[:, col2], grid[:, col1].copy()


def _swap_rows(grid, row1, row2):
    grid[row1, :], grid[row2, :] = grid[row2, :], grid[row1, :].copy()


def _swap_box_cols(grid):
    old_grid = grid.copy()
    grid[:, [0,1]], grid[:, [2,3]] = old_grid[:, [2,3]], old_grid[:, [0,1]]


def _swap_box_rows(grid):
    old_grid = grid.copy()
    grid[[0,1], :], grid[[2,3], :] = old_grid[[2,3], :], old_grid[[0,1], :]


def _permute_numbers(grid):
    perm = np.random.permutation(np.arange(1, SUDOKU_SIZE + 1))
    old_grid = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i][j] = perm[old_grid[i][j]-1]


def permute(grid):
    if random.random() > 0.5: _swap_rows(grid, 0, 1)
    if random.random() > 0.5: _swap_cols(grid, 0, 1)
    if random.random() > 0.5: _swap_cols(grid, 2, 3)
    if random.random() > 0.5: _swap_box_cols(grid)
    if random.random() > 0.5: _swap_box_rows(grid)
    if random.random() > 0.5: grid = grid.T
    _permute_numbers(grid)

    return grid


def _check_rows(grid):
    for row in grid:
        counts = np.bincount(row)[1:]
        if len(counts) > 0 and max(counts) > 1:
            return False
    return True


def _check_cols(grid):
    return _check_rows(grid.T)


def _check_boxes(grid):
    for i,j in [(0,0), (0,1), (1,0), (1,1)]:
        box = grid[[2*i, 2*i+1], [2*j, 2*j+1]]
        counts = np.bincount(np.ravel(box))[1:]
        if len(counts) > 0 and max(counts) > 1:
            return False
    return True


def check_valid(grid):
    return _check_rows(grid) and _check_cols(grid) and _check_boxes(grid)


def unique_solution(grid):
    def rec(grid, num_solutions):
        # Find empty squares
        gaps = np.where(grid==0)
        if len(gaps[0]) == 0:
            # All filled in
            return num_solutions + 1
        else:
            gap = (gaps[0][0], gaps[1][0])
            for added_entry in range(1, SUDOKU_SIZE + 1):
                new_grid = grid.copy()
                new_grid[gap] = added_entry
                if check_valid(new_grid):
                    num_solutions = rec(new_grid, num_solutions)
                    if num_solutions > 1:
                        return num_solutions
        return num_solutions
    
    num_solutions = 0
    num_solutions = rec(grid, num_solutions)
    return num_solutions == 1


def dig(grid):
    """Dig holes until lose uniqueness of solution."""
    ddug_grid = grid.copy()
    while unique_solution(ddug_grid):
        dug_grid = ddug_grid.copy()
        holes = np.where(grid != 0)
        new_hole = (np.random.choice(holes[0]), np.random.choice(holes[1]))
        ddug_grid[new_hole] = 0
    return dug_grid


def generate_grid(flat=False):
    if SUDOKU_SIZE == 4:
        base = BASES[np.random.randint(0,3)]
        grid = dig(base)
    elif SUDOKU_SIZE == 9:
        grid = np.array(sudoku_maker.make())
        for _ in range(50):
            i = np.random.choice(np.arange(SUDOKU_SIZE))
            j = np.random.choice(np.arange(SUDOKU_SIZE))
            grid[i][j] = 0

    if flat:
        return flatten(grid)
    else:
        return grid


if __name__ == '__main__':
    print(generate_grid())
