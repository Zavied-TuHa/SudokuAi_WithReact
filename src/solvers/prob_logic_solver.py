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
        prob_table = {(row, col): np.zeros(9, dtype=np.float32) for row in range(9) for col in range(9)}
        total_solutions = len(solutions)

        # Tối ưu hóa bằng vector hóa
        for _, row in solutions.iterrows():
            grid = np.array([list(map(int, row['solution'][i * 9:(i + 1) * 9])) for i in range(9)])
            for r in range(9):
                for c in range(9):
                    num = grid[r][c]
                    prob_table[(r, c)][num - 1] += 1.0

        # Chuẩn hóa xác suất
        for pos in prob_table:
            total = prob_table[pos].sum()
            prob_table[pos] = prob_table[pos] / total if total > 0 else np.full(9, 1.0 / 9, dtype=np.float32)
        logger.info("Bảng xác suất được xây dựng thành công")
        return prob_table

    def solve(self, puzzle: List[List[int]], solution: Optional[List[List[int]]] = None) -> List[List[int]]:
        """
        Giải Sudoku bằng logic xác suất kết hợp lan truyền ràng buộc.
        Không yêu cầu điền hết bảng, cho phép điền thiếu và sai.

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
            grid = np.array(puzzle, dtype=np.int32)
            candidates = self._initialize_candidates(grid)

            # Tiếp tục giải cho đến khi không còn ô nào có xác suất cao
            while self._solve_probabilistic(grid, candidates):
                self.iterations += 1
                if self.iterations > 1000:  # Giới hạn số vòng lặp để tránh treo
                    logger.info("Đạt giới hạn vòng lặp, dừng giải")
                    break

            # Tính độ chính xác nếu có solution
            if solution is not None:
                solution = np.array(solution, dtype=np.int32)
                correct = np.sum((grid != 0) & (grid == solution))
                total_filled = np.sum(grid != 0)
                self.accuracy = correct / total_filled if total_filled > 0 else 0.0
                logger.info(f"Độ chính xác: {self.accuracy * 100:.2f}%")

            logger.info("Hoàn tất giải bài toán")
            return grid.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi giải bài toán: {str(e)}")
            return puzzle

    def _initialize_candidates(self, grid: np.ndarray) -> np.ndarray:
        """
        Khởi tạo danh sách ứng viên cho mỗi ô bằng numpy.

        Args:
            grid (np.ndarray): Lưới 9x9.

        Returns:
            np.ndarray: Mảng ứng viên (9x9x10, với 0 không được dùng).
        """
        candidates = np.ones((9, 9, 10), dtype=np.bool_)
        candidates[:, :, 0] = False  # Không sử dụng số 0
        for row in range(9):
            for col in range(9):
                if grid[row, col] != 0:
                    candidates[row, col, :] = False
                    candidates[row, col, grid[row, col]] = True
                else:
                    for num in range(1, 10):
                        if not is_valid_move(grid.tolist(), row, col, num):
                            candidates[row, col, num] = False
        return candidates

    def _propagate_constraints(self, grid: np.ndarray, candidates: np.ndarray, row: int, col: int,
                               num: int) -> bool:
        """
        Lan truyền ràng buộc sau khi đặt một số.

        Args:
            grid (np.ndarray): Lưới 9x9.
            candidates (np.ndarray): Mảng ứng viên.
            row (int): Chỉ số hàng.
            col (int): Chỉ số cột.
            num (int): Số được đặt.

        Returns:
            bool: True nếu lan truyền thành công, False nếu có xung đột.
        """
        if not is_valid_move(grid.tolist(), row, col, num):
            self.invalid_attempts += 1
            return False

        grid[row, col] = num
        candidates[row, col, :] = False
        candidates[row, col, num] = True

        # Cập nhật hàng
        candidates[row, :, num] = False
        if np.any(candidates[row, :, 1:].sum(axis=1) == 0):
            self.conflicts += 1
            return False

        # Cập nhật cột
        candidates[:, col, num] = False
        if np.any(candidates[:, col, 1:].sum(axis=1) == 0):
            self.conflicts += 1
            return False

        # Cập nhật ô 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        candidates[start_row:start_row + 3, start_col:start_col + 3, num] = False
        if np.any(candidates[start_row:start_row + 3, start_col:start_col + 3, 1:].sum(axis=2).ravel() == 0):
            self.conflicts += 1
            return False

        return True

    def _get_best_cell(self, candidates: np.ndarray, grid: np.ndarray) -> Optional[Tuple[int, int, List[Tuple[int, float]]]]:
        """
        Tìm ô có xác suất cao nhất cho số tiếp theo.

        Args:
            candidates (np.ndarray): Mảng ứng viên.
            grid (np.ndarray): Lưới hiện tại.

        Returns:
            Optional[Tuple[int, int, List[Tuple[int, float]]]]: (row, col, [(num, prob), ...]) hoặc None.
        """
        empty_cells = np.where((grid == 0) & (candidates[:, :, 1:].sum(axis=2) > 0))
        if not empty_cells[0].size:
            return None

        # Tìm ô có ít ứng viên nhất
        candidate_counts = candidates[empty_cells[0], empty_cells[1], 1:].sum(axis=1)
        min_candidates = candidate_counts.min()
        if min_candidates == 0:
            self.invalid_attempts += 1
            return None

        min_indices = np.where(candidate_counts == min_candidates)[0]
        row, col = empty_cells[0][min_indices[0]], empty_cells[1][min_indices[0]]

        # Tính xác suất
        probs = []
        for num in range(1, 10):
            if candidates[row, col, num]:
                row_count = candidates[row, :, num].sum()
                col_count = candidates[:, col, num].sum()
                subgrid_count = candidates[3 * (row // 3):3 * (row // 3) + 3,
                                          3 * (col // 3):3 * (col // 3) + 3, num].sum()
                constraint_prob = 1.0 / max(1, row_count + col_count + subgrid_count)
                precomputed_prob = self.probability_table.get((row, col), np.full(9, 1.0 / 9))[num - 1]
                prob = 0.7 * constraint_prob + 0.3 * precomputed_prob
                probs.append((num, prob))

        total_prob = sum(p for _, p in probs)
        if total_prob > 0:
            probs = [(num, p / total_prob) for num, p in probs]
        probs.sort(key=lambda x: x[1], reverse=True)
        return row, col, probs

    def _solve_probabilistic(self, grid: np.ndarray, candidates: np.ndarray) -> bool:
        """
        Giải bài toán bằng logic xác suất với lan truyền ràng buộc.

        Args:
            grid (np.ndarray): Lưới 9x9 cần giải.
            candidates (np.ndarray): Mảng ứng viên.

        Returns:
            bool: True nếu có thay đổi, False nếu không.
        """
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
            return self._propagate_constraints(grid, candidates, row, col, num)
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