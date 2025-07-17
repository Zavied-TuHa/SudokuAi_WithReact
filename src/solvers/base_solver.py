from abc import ABC, abstractmethod
from typing import List, Optional
import logging
from src.utils.stats_collector import StatsCollector

# Configure logger
logger = logging.getLogger("sudoku_ai.base_solver")

class BaseSolver(ABC):
    """
    Abstract base class for Sudoku solvers.
    """
    @abstractmethod
    def solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Solve the given Sudoku puzzle.

        Args:
            puzzle (List[List[int]]): 9x9 grid representing the Sudoku puzzle (0 for empty cells).

        Returns:
            Optional[List[List[int]]]: Solved 9x9 grid or None if no solution exists.
        """
        pass

    @abstractmethod
    def collect_stats(self, stats_collector: StatsCollector, puzzle: List[List[int]],
                     result: Optional[List[List[int]]], solution: List[List[int]], solve_time: float) -> None:
        """
        Collect solver-specific statistics.

        Args:
            stats_collector (StatsCollector): Instance to collect stats.
            puzzle (List[List[int]]): Original puzzle.
            result (Optional[List[List[int]]]): Solver's result.
            solution (List[List[int]]): True solution.
            solve_time (float): Time taken to solve.
        """
        pass