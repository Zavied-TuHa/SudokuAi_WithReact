from .base_solver import BaseSolver
from src.utils.sudoku_utils import is_valid_move, is_valid_grid
from src.utils.stats_collector import StatsCollector
import logging
from typing import List, Optional, Tuple
import psutil
import os
import numpy as np

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.rule_based_solver")

class RuleBasedSolver(BaseSolver):
    """
    Rule-based solver using backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    def __init__(self):
        """Khởi tạo solver với các bộ đếm cho bước và quy tắc được kích hoạt."""
        self.steps = 0
        self.rules_triggered = 0
        self.invalid_attempts = 0

    def solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Giải Sudoku bằng thuật toán backtracking với heuristic MRV.
        Yêu cầu điền đầy đủ bảng.

        Args:
            puzzle (List[List[int]]): Lưới 9x9 của bài toán Sudoku (0 cho ô trống).

        Returns:
            Optional[List[List[int]]]: Lưới 9x9 đã giải hoặc None nếu không có lời giải.
        """
        try:
            logger.info("Bắt đầu giải Sudoku bằng RuleBasedSolver")
            if not is_valid_grid(puzzle):
                logger.error("Lưới đầu vào không hợp lệ")
                raise ValueError("Lưới đầu vào không hợp lệ")

            grid = np.array(puzzle, dtype=np.int32)
            self.steps = 0
            self.rules_triggered = 0
            self.invalid_attempts = 0
            if self._backtrack(grid):
                if np.any(grid == 0):
                    logger.warning("Lời giải chứa ô trống")
                    return None
                if not is_valid_grid(grid.tolist()):
                    logger.warning("Lời giải không hợp lệ")
                    return None
                logger.info("Giải bài toán thành công")
                return grid.tolist()
            logger.warning("Không tìm thấy lời giải")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi giải bài toán: {str(e)}")
            raise ValueError(f"Lỗi khi giải bài toán: {str(e)}")

    def _get_mrv_cell(self, grid: np.ndarray) -> Optional[Tuple[int, int, List[int]]]:
        """
        Tìm ô có số lượng giá trị hợp lệ tối thiểu.

        Args:
            grid (np.ndarray): Lưới 9x9.

        Returns:
            Optional[Tuple[int, int, List[int]]]: (row, col, valid_values) hoặc None nếu không có ô trống hoặc trạng thái không hợp lệ.
        """
        empty_cells = np.where(grid == 0)
        if not empty_cells[0].size:
            return None

        min_values = float('inf')
        best_cell = None
        best_valid_nums = None

        for row, col in zip(empty_cells[0], empty_cells[1]):
            valid_nums = [num for num in range(1, 10) if is_valid_move(grid.tolist(), row, col, num)]
            if len(valid_nums) < min_values:
                min_values = len(valid_nums)
                best_cell = (row, col)
                best_valid_nums = valid_nums
            if min_values == 0:
                self.invalid_attempts += 1
                return None

        return best_cell[0], best_cell[1], best_valid_nums

    def _backtrack(self, grid: np.ndarray) -> bool:
        """
        Thuật toán backtracking đệ quy với heuristic MRV.

        Args:
            grid (np.ndarray): Lưới 9x9 cần giải.

        Returns:
            bool: True nếu tìm thấy lời giải, False nếu không.
        """
        self.steps += 1
        cell = self._get_mrv_cell(grid)
        if cell is None:
            return not np.any(grid == 0)  # Kiểm tra nếu lưới đã đầy

        row, col, valid_nums = cell
        if not valid_nums:
            self.invalid_attempts += 1
            return False

        for num in valid_nums:
            grid[row, col] = num
            self.rules_triggered += 1
            logger.debug(f"Thử {num} tại ({row}, {col}), bước {self.steps}")
            if self._backtrack(grid):
                return True
            grid[row, col] = 0

        return False

    def collect_stats(self, stats_collector: StatsCollector, puzzle: List[List[int]],
                     result: Optional[List[List[int]]], solution: List[List[int]], solve_time: float) -> None:
        """
        Thu thập thống kê của solver.

        Args:
            stats_collector (StatsCollector): Đối tượng thu thập thống kê.
            puzzle (List[List[int]]): Bài toán gốc.
            result (Optional[List[List[int]]]): Kết quả của solver.
            solution (List[List[int]]): Lời giải đúng.
            solve_time (float): Thời gian giải.
        """
        stats_collector.add_rule_based_stat("steps_taken", self.steps)
        stats_collector.add_rule_based_stat("rules_triggered", self.rules_triggered)
        stats_collector.add_rule_based_stat("time_per_step", solve_time / max(1, self.steps))
        stats_collector.add_rule_based_stat("invalid_attempts", self.invalid_attempts)
        fail_rate = 1.0 if result is None or not np.array_equal(result, solution) else 0.0
        stats_collector.add_rule_based_stat("fail_rate", fail_rate)
        stats_collector.add_rule_based_stat("memory_usage", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)