import logging

import sudoku
import numpy as np


def flatten(grid):
    mask = (grid.ravel() + 4*np.arange(16) - 1)[(np.where(grid.ravel() != 0))]
    flat = np.zeros(64)
    flat[mask] = 1
    return flat.astype(int)


def unflatten(grid):
    return (grid.reshape((4,4,4)).argmax(axis=2) + 1) * grid.reshape((4,4,4)).max(axis=2).astype(int)


class Environment:
    def __init__(self):
        self.num_actions = 64
        self.start_grid = self.new_grid()
        self.current_grid = self.start_grid.copy()
    
    def new_grid(self):
        grid = sudoku.generate_grid()
        logging.debug("Creating new grid\n%s", grid)
        self.start_grid = flatten(grid)
        self.current_grid = self.start_grid.copy()
        return self.current_grid

    def reset_grid(self):
        self.current_grid = self.start_grid.copy()
        return self.current_grid

    def act(self, action):
        new_grid = unflatten(self.current_grid)
        if new_grid[action//16][(action%16)//4] != 0:
            # This square already contains an entry.
            reward = -1
            terminal = 0 
        else:
            new_grid[action//16][(action%16)//4] = action%4 + 1
            is_valid = sudoku.check_valid(new_grid)
            if is_valid:
                if np.min(new_grid) > 0:
                    # Have solved the grid.
                    self.current_grid = None
                    print("\nSudoku solved!\n")
                    reward = 10
                    terminal = 1
                else:
                    self.current_grid = flatten(new_grid)
                    reward = 1
                    terminal = 0
            else:
                self.current_grid = None
                reward = -1
                terminal = 1
        
        return self.current_grid, reward, terminal

