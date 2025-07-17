from .base_solver import BaseSolver
from src.utils.sudoku_utils import is_valid_move, is_valid_grid
from src.utils.stats_collector import StatsCollector
from src.data.preprocessor import preprocess_puzzle
import numpy as np
import logging
import time
import joblib
import os
import psutil
from typing import List, Optional

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.random_forest_solver")

class RandomForestSolver(BaseSolver):
    """
    Solver sử dụng Random Forest Classifier để dự đoán lưới Sudoku.
    Cho phép lời giải không hoàn chỉnh và tính toán độ chính xác.
    """
    def __init__(self, model_path: str):
        """
        Khởi tạo solver Random Forest.

        Args:
            model_path (str): Đường dẫn đến mô hình Random Forest đã huấn luyện.
        """
        self.model_path = model_path
        self.model = None
        self.inference_time = 0.0
        self.empty_cell_count = 0
        self.accuracy = 0.0
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Đã tải mô hình Random Forest từ {model_path}")
        except FileNotFoundError:
            logger.warning(f"Không tìm thấy file mô hình: {model_path}. RandomForestSolver bị vô hiệu hóa.")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}. RandomForestSolver bị vô hiệu hóa.")
            self.model = None

    def solve(self, puzzle: List[List[int]], solution: Optional[List[List[int]]]=None) -> List[List[int]]:
        """
        Giải Sudoku bằng mô hình Random Forest Classifier.
        Không yêu cầu điền hết bảng, chỉ điền các ô có độ tin cậy cao.

        Args:
            puzzle (List[List[int]]): Lưới 9x9 của bài toán Sudoku (0 cho ô trống).
            solution (Optional[List[List[int]]]): Lời giải đúng (dùng để tính độ chính xác).

        Returns:
            List[List[int]]: Lưới 9x9 đã giải (có thể không hoàn chỉnh).
        """
        if self.model is None:
            logger.warning("Mô hình Random Forest chưa được tải. Trả về lưới đầu vào.")
            return puzzle

        try:
            logger.info("Bắt đầu giải Sudoku bằng RandomForestSolver")
            if not is_valid_grid(puzzle, allow_empty=True):
                logger.error("Lưới đầu vào không hợp lệ")
                raise ValueError("Lưới đầu vào không hợp lệ")

            grid = np.array(puzzle, dtype=np.int32)
            empty_cells = np.where(grid == 0)
            if not empty_cells[0].size:
                logger.info("Không có ô trống để dự đoán")
                return grid.tolist()

            # Tiền xử lý hàng loạt
            input_data = []
            cell_indices = []
            for row, col in zip(empty_cells[0], empty_cells[1]):
                input_data.append(preprocess_puzzle(grid.tolist(), for_ml=True))
                cell_indices.append((row, col))
            input_data = np.array(input_data, dtype=np.int32)

            # Dự đoán hàng loạt
            start_time = time.time()
            if input_data.size:
                pred_probs = self.model.predict_proba(input_data)
                predictions = np.argmax(pred_probs, axis=1) + 1  # Giá trị từ 1-9
                confidences = np.max(pred_probs, axis=1)
            else:
                predictions = []
                confidences = []
            self.inference_time += time.time() - start_time

            # Điền các ô có độ tin cậy cao
            confidence_threshold = 0.6  # Ngưỡng độ tin cậy
            for (row, col), pred, conf in zip(cell_indices, predictions, confidences):
                if conf >= confidence_threshold and is_valid_move(grid.tolist(), row, col, pred):
                    grid[row, col] = pred

            # Tính độ chính xác nếu có solution
            if solution is not None:
                solution = np.array(solution, dtype=np.int32)
                correct = np.sum((grid != 0) & (grid == solution))
                total_filled = np.sum(grid != 0)
                self.accuracy = correct / total_filled if total_filled > 0 else 0.0
                logger.info(f"Độ chính xác: {self.accuracy * 100:.2f}%")

            self.empty_cell_count = np.sum(grid == 0)
            logger.info("Hoàn tất giải bài toán")
            return grid.tolist()
        except Exception as e:
            logger.error(f"Lỗi khi giải bài toán: {str(e)}")
            return puzzle

    def collect_stats(self, stats_collector: StatsCollector, puzzle: List[List[int]],
                     result: List[List[int]], solution: List[List[int]], solve_time: float) -> None:
        """
        Thu thập thống kê của solver Random Forest.

        Args:
            stats_collector (StatsCollector): Đối tượng thu thập thống kê.
            puzzle (List[List[int]]): Bài toán gốc.
            result (List[List[int]]): Kết quả của solver.
            solution (List[List[int]]): Lời giải đúng.
            solve_time (float): Thời gian giải.
        """
        stats_collector.add_rf_stat("inference_time", self.inference_time)
        stats_collector.add_rf_stat("accuracy", self.accuracy)
        stats_collector.add_rf_stat("model_size", os.path.getsize(self.model_path) / 1024 / 1024 if self.model and os.path.exists(self.model_path) else 0.0)
        stats_collector.add_rf_stat("memory_usage", psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        stats_collector.add_rf_stat("empty_cells_detected", self.empty_cell_count)