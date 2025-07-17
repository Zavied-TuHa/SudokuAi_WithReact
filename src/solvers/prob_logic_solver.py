from .base_solver import BaseSolver
from src.utils.sudoku_utils import is_valid_move, is_valid_grid
from src.utils.stats_collector import StatsCollector
import logging
import psutil
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.prob_logic_solver")


class ProbabilisticLogicSolver(BaseSolver):
    """
    Solver kết hợp thống kê xác suất từ dữ liệu huấn luyện với lan truyền ràng buộc để giải Sudoku.
    Cho phép lời giải không hoàn chỉnh và tính toán độ chính xác.
    """

    def __init__(self, probability_table: Optional[Dict[Tuple[int, int], List[float]]] = None):
        """
        Khởi tạo solver với bảng xác suất và các biến thống kê.

        Args:
            probability_table (Optional[Dict[Tuple[int, int], List[float]]]): Bảng xác suất cho mỗi ô và số.
        """
        self.iterations = 0
        self.conflicts = 0
        self.invalid_attempts = 0
        self.probability_table = probability_table or {}
        self.accuracy = 0.0

    @staticmethod
    def build_probability_table(solutions: pd.DataFrame) -> Dict[Tuple[int, int], List[float]]:
        """
        Xây dựng bảng xác suất từ các lời giải Sudoku.

        Args:
            solutions (pd.DataFrame): DataFrame chứa cột 'solution' với các lời giải.

        Returns:
            Dict[Tuple[int, int], List[float]]: Xác suất của từng số (1-9) cho mỗi ô.
        """
        logger.info("Xây dựng bảng xác suất từ dữ liệu huấn luyện")
        prob_table = {(row, col): [0.0] * 9 for row in range(9) for col in range(9)}
        total_solutions = len(solutions)

        for _, row in solutions.iterrows():
            grid = [list(map(int, row['solution'][i * 9:(i + 1) * 9])) for i in range(9)]
            for r in range(9):
                for c in range(9):
                    num = grid[r][c]
                    prob_table[(r, c)][num - 1] += 1.0

        # Chuẩn hóa xác suất
        for pos in prob_table:
            total = sum(prob_table[pos])
            if total > 0:
                prob_table[pos] = [count / total for count in prob_table[pos]]
            else:
                prob_table[pos] = [1.0 / 9] * 9
        logger.info("Bảng xác suất được xây dựng thành công")
        return prob_table

    def solve(self, puzzle: List[List[int]], solution: Optional[List[List[int]]] = None) -> List[List[int]]:
        """
        Giải Sudoku bằng logic xác suất kết hợp lan truyền ràng buộc.

        Args:
            puzzle (List[List[int]]): Lưới 9x9 của bài toán Sudoku (0 cho ô trống).
            solution (Optional[List[List[int]]]): Lời giải đúng (dùng để tính độ chính xác).

        Returns:
            List[List[int]]: Lưới 9x9 đã giải (có thể không hoàn chỉnh).
        """
        try:
            logger.info("Bắt đầu giải Sudoku bằng ProbabilisticLogicSolver")
            if not is_valid_grid(puzzle, allow_empty=True):
                logger.error("Lưới đầu vào không hợp lệ")
                raise ValueError("Lưới đầu vào không hợp lệ")

            self.iterations = 0
            self.conflicts = 0
            self.invalid_attempts = 0
            grid = [row[:] for row in puzzle]
            candidates = self._initialize_candidates(grid)
            self._solve_probabilistic(grid, candidates)

            # Tính độ chính xác nếu có solution
            if solution:
                correct = sum(1 for i in range(9) for j in range(9)
                              if grid[i][j] != 0 and grid[i][j] == solution[i][j])
                total_filled = sum(1 for i in range(9) for j in range(9) if grid[i][j] != 0)
                self.accuracy = correct / total_filled if total_filled > 0 else 0.0
                logger.info(f"Độ chính xác: {self.accuracy * 100:.2f}%")

            logger.info("Hoàn tất giải bài toán")
            return grid
        except Exception as e:
            logger.error(f"Lỗi khi giải bài toán: {str(e)}")
            return puzzle

    def _initialize_candidates(self, grid: List[List[int]]) -> List[List[List[int]]]:
        """
        Khởi tạo danh sách ứng viên cho mỗi ô.

        Args:
            grid (List[List[int]]): Lưới 9x9.

        Returns:
            List[List[List[int]]]: Danh sách ứng viên cho mỗi ô.
        """
        candidates = [[[0] for _ in range(9)] for _ in range(9)]
        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0:
                    candidates[row][col] = [num for num in range(1, 10) if is_valid_move(grid, row, col, num)]
                else:
                    candidates[row][col] = [grid[row][col]]
        return candidates

    def _propagate_constraints(self, grid: List[List[int]], candidates: List[List[List[int]]], row: int, col: int,
                               num: int) -> bool:
        """
        Lan truyền ràng buộc sau khi đặt một số.

        Args:
            grid (List[List[int]]): Lưới 9x9.
            candidates (List[List[List[int]]]): Danh sách ứng viên.
            row (int): Chỉ số hàng.
            col (int): Chỉ số cột.
            num (int): Số được đặt.

        Returns:
            bool: True nếu lan truyền thành công, False nếu có xung đột.
        """
        grid[row][col] = num
        candidates[row][col] = [num]
        for j in range(9):
            if j != col and num in candidates[row][j]:
                candidates[row][j].remove(num)
                if not candidates[row][j]:
                    self.conflicts += 1
                    return False
        for i in range(9):
            if i != row and num in candidates[i][col]:
                candidates[i][col].remove(num)
                if not candidates[i][col]:
                    self.conflicts += 1
                    return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if (i != row or j != col) and num in candidates[i][j]:
                    candidates[i][j].remove(num)
                    if not candidates[i][j]:
                        self.conflicts += 1
                        return False
        return True

    def _get_best_cell(self, candidates: List[List[List[int]]], grid: List[List[int]]) -> Optional[
        Tuple[int, int, List[Tuple[int, float]]]]:
        """
        Tìm ô có xác suất cao nhất cho số tiếp theo.

        Args:
            candidates (List[List[List[int]]]): Danh sách ứng viên.
            grid (List[List[int]]): Lưới hiện tại.

        Returns:
            Optional[Tuple[int, int, List[Tuple[int, float]]]]: (row, col, [(num, prob), ...]) hoặc None.
        """
        min_candidates = float('inf')
        best_cell = None
        best_probs = None

        for row in range(9):
            for col in range(9):
                if grid[row][col] == 0 and len(candidates[row][col]) > 0:
                    num_candidates = len(candidates[row][col])
                    if num_candidates < min_candidates:
                        min_candidates = num_candidates
                        best_cell = (row, col)
                        probs = []
                        for num in candidates[row][col]:
                            row_count = sum(1 for j in range(9) if num in candidates[row][j])
                            col_count = sum(1 for i in range(9) if num in candidates[i][col])
                            subgrid_count = sum(1 for i in range(3 * (row // 3), 3 * (row // 3) + 3)
                                                for j in range(3 * (col // 3), 3 * (col // 3) + 3)
                                                if num in candidates[i][j])
                            constraint_prob = 1.0 / max(1, row_count + col_count + subgrid_count)
                            precomputed_prob = self.probability_table.get((row, col), [1.0 / 9] * 9)[num - 1]
                            prob = 0.7 * constraint_prob + 0.3 * precomputed_prob
                            probs.append((num, prob))
                        total_prob = sum(p for _, p in probs)
                        if total_prob > 0:
                            probs = [(num, p / total_prob) for num, p in probs]
                        best_probs = sorted(probs, key=lambda x: x[1], reverse=True)
                    if min_candidates == 0:
                        self.invalid_attempts += 1
                        return None

        if best_cell is None:
            logger.debug("Không tìm thấy ô hợp lệ để gán")
            return None
        return best_cell[0], best_cell[1], best_probs

    def _solve_probabilistic(self, grid: List[List[int]], candidates: List[List[List[int]]]) -> bool:
        """
        Giải bài toán bằng logic xác suất với lan truyền ràng buộc.

        Args:
            grid (List[List[int]]): Lưới 9x9 cần giải.
            candidates (List[List[List[int]]]): Danh sách ứng viên.

        Returns:
            bool: True nếu có thay đổi, False nếu không.
        """
        self.iterations += 1
        cell = self._get_best_cell(candidates, grid)
        if not cell:
            return False

        row, col, probs = cell
        if not probs:
            self.conflicts += 1
            self.invalid_attempts += 1
            return False

        # Chọn giá trị có xác suất cao nhất nếu vượt ngưỡng
        num, prob = probs[0]
        threshold = 0.3
        if prob >= threshold:
            logger.debug(f"Gán {num} vào ({row}, {col}) với xác suất {prob:.3f}")
            if self._propagate_constraints(grid, candidates, row, col, num):
                return True
        return False

    def collect_stats(self, stats_collector: StatsCollector, puzzle: List[List[int]],
                      result: List[List[int]], solution: List[List[int]], solve_time: float) -> None:
        """
        Thu thập thống kê của solver.

        Args:
            stats_collector (StatsCollector): Đối tượng thu thập thống kê.
            puzzle (List[List[int]]): Bài toán gốc.
            result (List[List[int]]): Kết quả của solver.
            solution (List[List[int]]): Lời giải đúng.
            solve_time (float): Thời gian giải.
        """
        stats_collector.add_prob_logic_stat("iterations_to_converge", self.iterations)
        stats_collector.add_prob_logic_stat("number_of_conflicts", self.conflicts)
        stats_collector.add_prob_logic_stat("invalid_attempts", self.invalid_attempts)
        stats_collector.add_prob_logic_stat("accuracy", self.accuracy)
        stats_collector.add_prob_logic_stat("memory_usage", psutil.Process().memory_info().rss / 1024 / 1024)