import time
import logging
import numpy as np
from typing import List, Dict, Optional
from src.utils.sudoku_utils import is_valid_grid
from src.utils.stats_collector import StatsCollector

# Configure logger
logger = logging.getLogger("sudoku_ai.metrics")

def compute_metrics(
    solver,
    puzzles: List[List[List[int]]],
    solutions: List[List[List[int]]],
    stats_collector: StatsCollector
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a solver.

    Args:
        solver: Solver instance (RuleBasedSolver, ProbabilisticLogicSolver, or RandomForestSolver).
        puzzles (List[List[List[int]]]): List of 9x9 puzzle grids.
        solutions (List[List[List[int]]]): List of corresponding 9x9 solution grids.
        stats_collector (StatsCollector): Instance to collect and store metrics.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.

    Raises:
        ValueError: If inputs are invalid.
    """
    try:
        if len(puzzles) != len(solutions):
            logger.error("Number of puzzles and solutions must match")
            raise ValueError("INVALID_INPUT: Number of puzzles and solutions must match")

        logger.info(f"Computing metrics for {len(puzzles)} puzzles")
        accuracy = 0.0
        total_time = 0.0
        valid_puzzles = 0
        solved_count = 0

        for puzzle, solution in zip(puzzles, solutions):
            if not is_valid_grid(puzzle, allow_empty=True) or not is_valid_grid(solution, allow_empty=False):
                logger.warning("Skipping invalid puzzle or solution")
                continue

            start_time = time.time()
            result = solver.solve(puzzle)
            end_time = time.time()

            solve_time = end_time - start_time
            stats_collector.add_solve_time(solve_time)

            if result is not None and is_valid_grid(result, allow_empty=False) and np.array_equal(result, solution):
                accuracy += 1
                solved_count += 1
            else:
                logger.debug("Solver failed to produce valid solution")
            valid_puzzles += 1

            solver.collect_stats(stats_collector, puzzle, result, solution, solve_time)

        if valid_puzzles == 0:
            logger.warning("No valid puzzles to evaluate")
            return {
                "accuracy": 0.0,
                "solved_rate": 0.0,
                "avg_solve_time": 0.0,
                "memory_usage": stats_collector.get_memory_usage()
            }

        metrics = {
            "accuracy": accuracy / valid_puzzles,
            "solved_rate": solved_count / valid_puzzles,
            "avg_solve_time": total_time / valid_puzzles,
            "memory_usage": stats_collector.get_memory_usage()
        }

        metrics.update(stats_collector.get_stats())
        logger.info(f"Metrics computed: {metrics}")
        return metrics

    except ValueError as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error computing metrics: {str(e)}")
        raise ValueError(f"UNEXPECTED_ERROR: {str(e)}")