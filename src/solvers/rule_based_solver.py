from .base_solver import BaseSolver
from src.utils.sudoku_utils import is_valid_move, is_valid_grid
from src.utils.stats_collector import StatsCollector
import logging
from typing import List, Optional, Tuple
import psutil
import os
import numpy as np

# Configure logger
logger = logging.getLogger("sudoku_ai.rule_based_solver")

class RuleBasedSolver(BaseSolver):
    """
    Rule-based solver using backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    def __init__(self):
        """Initialize the rule-based solver with counters for steps and rules triggered."""
        self.steps = 0
        self.rules_triggered = 0
        self.invalid_attempts = 0

    def solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Solve the Sudoku puzzle using backtracking.

        Args:
            puzzle (List[List[int]]): 9x9 grid representing the Sudoku puzzle (0 for empty cells).

        Returns:
            Optional[List[List[int]]]: Solved 9x9 grid or None if no solution exists.
        """
        try:
            logger.info("Starting rule-based solver")
            if not is_valid_grid(puzzle):
                logger.error("Invalid initial puzzle")
                raise ValueError("Invalid initial puzzle")

            grid = [row[:] for row in puzzle]
            self.steps = 0
            self.rules_triggered = 0
            self.invalid_attempts = 0
            if self._backtrack(grid):
                if any(0 in row for row in grid):
                    logger.warning("Solution contains empty cells")
                    return None
                if not is_valid_grid(grid):
                    logger.warning("Solution is invalid")
                    return None
                logger.info("Puzzle solved successfully")
                return grid
            logger.warning("No solution found")
            return None
        except Exception as e:
            logger.error(f"Error solving puzzle: {str(e)}")
            raise ValueError(f"Error solving puzzle: {str(e)}")

    def _get_mrv_cell(self, grid: List[List[int]]) -> Optional[Tuple[int, int, List[int]]]:
        """
        Find the cell with the minimum remaining valid values.

        Args:
            grid (List[List[int]]): 9x9 grid.

        Returns:
            Optional[Tuple[int, int, List[int]]]: (row, col, valid_values) or None if no empty cells or invalid state.
        """
        min_values = float('inf')
        best_cell = None
        best_valid_nums = None

        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    valid_nums = [num for num in range(1, 10) if is_valid_move(grid, row, col, num)]
                    if len(valid_nums) < min_values:
                        min_values = len(valid_nums)
                        best_cell = (row, col)
                        best_valid_nums = valid_nums
                    if min_values == 0:
                        self.invalid_attempts += 1
                        return None

        if best_cell is None:
            return None
        return best_cell[0], best_cell[1], best_valid_nums

    def _backtrack(self, grid: List[List[int]]) -> bool:
        """
        Recursive backtracking algorithm with MRV heuristic.

        Args:
            grid (List[List[int]]): 9x9 grid to solve.

        Returns:
            bool: True if a solution is found, False otherwise.
        """
        self.steps += 1
        cell = self._get_mrv_cell(grid)
        if cell is None:
            return not any(0 in row for row in grid)  # Check if grid is fully filled

        row, col, valid_nums = cell
        if not valid_nums:
            self.invalid_attempts += 1
            return False

        for num in valid_nums:
            grid[row][col] = num
            self.rules_triggered += 1
            logger.debug(f"Trying {num} at ({row}, {col}), step {self.steps}")
            if self._backtrack(grid):
                return True
            grid[row][col] = 0

        return False

    def collect_stats(self, stats_collector: StatsCollector, puzzle: List[List[int]],
                     result: Optional[List[List[int]]], solution: List[List[int]], solve_time: float) -> None:
        """
        Collect rule-based solver statistics.

        Args:
            stats_collector (StatsCollector): Instance to collect stats.
            puzzle (List[List[int]]): Original puzzle.
            result (Optional[List[List[int]]]): Solver's result.
            solution (List[List[int]]): True solution.
            solve_time (float): Time taken to solve.
        """
        stats_collector.add_rule_based_stat("steps_taken", self.steps)
        stats_collector.add_rule_based_stat("rules_triggered", self.rules_triggered)
        stats_collector.add_rule_based_stat("time_per_step", solve_time / max(1, self.steps))
        stats_collector.add_rule_based_stat("invalid_attempts", self.invalid_attempts)
        fail_rate = 1.0 if result is None or not np.array_equal(result, solution) else 0.0
        stats_collector.add_rule_based_stat("fail_rate", fail_rate)
        stats_collector.add_rule_based_stat("memory_usage", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)