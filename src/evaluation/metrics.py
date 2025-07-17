import time
import logging
import numpy as np
from typing import List, Dict, Optional
from src.utils.sudoku_utils import is_valid_grid
from src.utils.stats_collector import StatsCollector
from src.solvers.rule_based_solver import RuleBasedSolver

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.metrics")

def compute_metrics(
    solver,
    puzzles: List[List[List[int]]],
    solutions: List[List[List[int]]],
    stats_collector: StatsCollector
) -> Dict[str, float]:
    """
    Tính toán các số liệu đánh giá cho solver.

    Args:
        solver: Đối tượng solver (RuleBasedSolver, ProbabilisticLogicSolver, hoặc RandomForestSolver).
        puzzles (List[List[List[int]]]): Danh sách lưới 9x9 của bài toán.
        solutions (List[List[List[int]]]): Danh sách lưới 9x9 của lời giải tương ứng.
        stats_collector (StatsCollector): Đối tượng thu thập số liệu.

    Returns:
        Dict[str, float]: Từ điển chứa các số liệu đánh giá.

    Raises:
        ValueError: Nếu đầu vào không hợp lệ.
    """
    try:
        if len(puzzles) != len(solutions):
            logger.error("Số lượng bài toán và lời giải phải khớp")
            raise ValueError("INVALID_INPUT: Số lượng bài toán và lời giải phải khớp")

        logger.info(f"Tính toán số liệu cho {len(puzzles)} bài toán")
        accuracy = 0.0
        total_time = 0.0
        valid_puzzles = 0
        solved_count = 0
        # Chỉ RuleBasedSolver yêu cầu lưới đầy đủ
        allow_empty = not isinstance(solver, RuleBasedSolver)

        for puzzle, solution in zip(puzzles, solutions):
            if not is_valid_grid(puzzle, allow_empty=True) or not is_valid_grid(solution, allow_empty=False):
                logger.warning("Bỏ qua bài toán hoặc lời giải không hợp lệ")
                continue

            start_time = time.time()
            result = solver.solve(puzzle, solution)
            solve_time = time.time() - start_time
            total_time += solve_time
            stats_collector.add_solve_time(solve_time)

            result_array = np.array(result, dtype=np.int32)
            solution_array = np.array(solution, dtype=np.int32)
            if is_valid_grid(result, allow_empty=allow_empty):
                # Tính độ chính xác dựa trên các ô đã điền
                correct = np.sum((result_array != 0) & (result_array == solution_array))
                total_filled = np.sum(result_array != 0)
                puzzle_accuracy = correct / total_filled if total_filled > 0 else 0.0
                accuracy += puzzle_accuracy
                if puzzle_accuracy > 0.9 and (allow_empty or not np.any(result_array == 0)):
                    solved_count += 1
            else:
                logger.debug("Solver tạo ra lời giải không hợp lệ")
                stats_collector.add_stat("invalid_solution", 1)

            valid_puzzles += 1
            solver.collect_stats(stats_collector, puzzle, result, solution, solve_time)

        if valid_puzzles == 0:
            logger.warning("Không có bài toán hợp lệ để đánh giá")
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
        logger.info(f"Số liệu được tính toán: {metrics}")
        return metrics

    except ValueError as e:
        logger.error(f"Lỗi khi tính toán số liệu: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tính toán số liệu: {str(e)}")
        raise ValueError(f"UNEXPECTED_ERROR: {str(e)}")