"""
Sudoku grid environment.
"""

import logging

import sudoku
import numpy as np


SUDOKU_SIZE = 9


class Environment:
    def __init__(self):
        """
        Sudoku solving environment.
        """
        self.num_actions = SUDOKU_SIZE**3
        self.start_grid = self.new_grid()
        self.current_grid = self.start_grid.copy()
    
    def new_grid(self):
        """
        Initialise a new sudoku grid for this environment.

        :return: The new grid
        """
        self.start_grid = sudoku.generate_grid(flat=True)
        logging.debug("Creating new grid\n%s", self.start_grid)
        self.current_grid = self.start_grid.copy()

        return self.current_grid

#    def reset_grid(self):
#        self.current_grid = self.start_grid.copy()
#        return self.current_grid

    def act(self, action):
        """
        Perform an action within the current grid.

        :param action: The action to perform
        :return: A (grid, reward, terminal) tuple containing the new grid,
                 the reward for the given action, and whether the game is now over.
        """
        new_grid = sudoku.unflatten(self.current_grid)
        row_idx = action // (SUDOKU_SIZE**2)
        col_idx = (action % (SUDOKU_SIZE**2)) // SUDOKU_SIZE
        entry = action % SUDOKU_SIZE + 1

        if new_grid[row_idx][col_idx] != 0:
            # This square already contains an entry.
            reward = -1
            terminal = 1
        else:
            new_grid[row_idx][col_idx] = entry
            is_valid = sudoku.check_valid(new_grid)
            if is_valid:
                if np.min(new_grid) > 0:
                    # Have solved the grid.
                    self.current_grid = sudoku.flatten(new_grid)
                    print("\nSudoku solved!\n")
                    reward = 1
                    terminal = 1
                else:
                    self.current_grid = sudoku.flatten(new_grid)
                    reward = 0
                    terminal = 0
            else:
                self.current_grid = None
                reward = -1
                terminal = 1
        
        return self.current_grid, reward, terminal
