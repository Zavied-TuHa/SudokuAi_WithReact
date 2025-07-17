import logging
import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import setup_logging
from src.utils.config_parser import load_config
from src.utils.stats_collector import StatsCollector
from src.data.preprocessor import preprocess_puzzle, string_to_grid
from src.data.data_processor import clean_data
from src.evaluation.metrics import compute_metrics
from src.solvers.prob_logic_solver import ProbabilisticLogicSolver
from src.solvers.random_forest_solver import RandomForestSolver
from src.utils.sudoku_utils import is_valid_grid
import psutil
import joblib
import pandas as pd

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.train")

def check_solution_validity(solver, puzzle: list, solution: list, stats_collector: StatsCollector) -> bool:
    """
    Kiểm tra tính hợp lệ của lời giải từ solver.

    Args:
        solver: Đối tượng solver.
        puzzle (List[List[int]]): Lưới bài toán.
        solution (List[List[int]]): Lời giải mong đợi.
        stats_collector (StatsCollector): Đối tượng thu thập thống kê.

    Returns:
        bool: True nếu lời giải hợp lệ, False nếu không.
    """
    try:
        result = solver.solve(puzzle, solution)
        if not is_valid_grid(result, allow_empty=True):
            logger.warning("Lời giải không hợp lệ")
            stats_collector.add_stat("invalid_solution", 1)
            return False
        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra lời giải: {str(e)}")
        stats_collector.add_stat("solver_error", 1)
        return False

def train_random_forest(train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict, model_path: str, stats_collector: StatsCollector, stats_path: str) -> None:
    """
    Huấn luyện và đánh giá RandomForestSolver.

    Args:
        train_data (pd.DataFrame): Dữ liệu huấn luyện.
        test_data (pd.DataFrame): Dữ liệu kiểm tra.
        config (dict): Cấu hình.
        model_path (str): Đường dẫn lưu mô hình.
        stats_collector (StatsCollector): Đối tượng thu thập thống kê.
        stats_path (str): Đường dẫn lưu thống kê.
    """
    logger.info("Huấn luyện RandomForestSolver")
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    # Kiểm tra tính hợp lệ của dữ liệu huấn luyện
    valid_rows = []
    for _, row in train_data.iterrows():
        try:
            puzzle = string_to_grid(row["puzzle"])
            solution = string_to_grid(row["solution"])
            if is_valid_grid(puzzle, allow_empty=True) and is_valid_grid(solution, allow_empty=False):
                valid_rows.append(row)
        except:
            continue
    train_data = pd.DataFrame(valid_rows)
    logger.info(f"Dữ liệu huấn luyện sau khi làm sạch: {len(train_data)} mẫu")

    # Sử dụng toàn bộ dữ liệu huấn luyện
    batch_size = 10000
    X_train = []
    y_train = []
    for i in tqdm(range(0, len(train_data), batch_size), desc="Chuẩn bị dữ liệu huấn luyện"):
        batch = train_data[i:i + batch_size]
        X_train.extend([preprocess_puzzle(row["puzzle"], for_ml=True) for _, row in batch.iterrows()])
        y_train.extend([preprocess_puzzle(row["solution"], for_ml=True) for _, row in batch.iterrows()])
    X_train = np.array(X_train, dtype=np.int32)
    y_train = np.array(y_train, dtype=np.int32)

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        min_samples_split=config["model"]["min_samples_split"],
        class_weight="balanced",
        random_state=config["model"]["random_state"],
        n_jobs=-1
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    stats_collector.add_rf_stat("train_time", train_time)
    mem_after = process.memory_info().rss / 1024 / 1024
    stats_collector.add_rf_stat("training_memory_increase", mem_after - mem_before)

    joblib.dump(model, model_path, compress=3)
    logger.info(f"Mô hình Random Forest được lưu tại {model_path}")

    # Đánh giá trên tập test
    solver = RandomForestSolver(model_path)
    valid_solutions = 0
    total_puzzles = len(test_data)
    for _, row in tqdm(test_data.iterrows(), total=total_puzzles, desc="Đánh giá Random Forest"):
        puzzle = string_to_grid(row["puzzle"])
        solution = string_to_grid(row["solution"])
        if check_solution_validity(solver, puzzle, solution, stats_collector):
            valid_solutions += 1

    metrics = compute_metrics(solver, [string_to_grid(row["puzzle"]) for _, row in test_data.iterrows()],
                             [string_to_grid(row["solution"]) for _, row in test_data.iterrows()], stats_collector)
    stats_collector.add_rf_stat("valid_solutions", valid_solutions / max(1, total_puzzles))
    stats_collector.save_stats(stats_path)
    logger.info(f"Độ chính xác RandomForestSolver: {metrics}")

def test_prob_logic(test_data: pd.DataFrame, train_data: pd.DataFrame, stats_collector: StatsCollector, stats_path: str) -> None:
    """
    Đánh giá ProbabilisticLogicSolver trên tập dữ liệu kiểm tra.

    Args:
        test_data (pd.DataFrame): Dữ liệu kiểm tra.
        train_data (pd.DataFrame): Dữ liệu huấn luyện để xây bảng xác suất.
        stats_collector (StatsCollector): Đối tượng thu thập thống kê.
        stats_path (str): Đường dẫn lưu thống kê.
    """
    logger.info("Xây dựng bảng xác suất cho ProbabilisticLogicSolver")
    prob_table = ProbabilisticLogicSolver.build_probability_table(train_data)
    logger.info("Đánh giá ProbabilisticLogicSolver")
    solver = ProbabilisticLogicSolver(probability_table=prob_table)
    valid_solutions = 0
    total_puzzles = len(test_data)

    for _, row in tqdm(test_data.iterrows(), total=total_puzzles, desc="Đánh giá Prob-Logic"):
        puzzle = string_to_grid(row["puzzle"])
        solution = string_to_grid(row["solution"])
        if check_solution_validity(solver, puzzle, solution, stats_collector):
            valid_solutions += 1

    metrics = compute_metrics(solver, [string_to_grid(row["puzzle"]) for _, row in test_data.iterrows()],
                             [string_to_grid(row["solution"]) for _, row in test_data.iterrows()], stats_collector)
    stats_collector.add_stat("prob_logic_valid_solutions", valid_solutions / max(1, total_puzzles))
    stats_collector.save_stats(stats_path)
    logger.info(f"Độ chính xác ProbabilisticLogicSolver: {metrics}")

def main():
    """
    Hàm chính để huấn luyện RandomForestSolver và kiểm tra ProbabilisticLogicSolver.
    """
    setup_logging()
    config = load_config()
    stats_collector = StatsCollector()

    logger.info("Tải và làm sạch dữ liệu")
    train_df, test_df = clean_data(config["data"]["sudoku_3m_path"], config)
    logger.info(f"Phân chia dữ liệu: {len(train_df)} hàng huấn luyện, {len(test_df)} hàng kiểm tra")

    if abs(len(train_df) / (len(train_df) + len(test_df)) - 0.9) > 0.01:
        logger.error("Tỷ lệ phân chia train-test không đạt khoảng 9:1")
        raise ValueError("Tỷ lệ phân chia train-test không đạt khoảng 9:1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path_rf = os.path.join(config["data"]["model_save_path"], f"random_forest_model_{timestamp}.pkl")
    stats_path_prob = os.path.join(config["data"]["stats_save_path"], f"prob_logic_stats_{timestamp}.json")
    stats_path_rf = os.path.join(config["data"]["stats_save_path"], f"random_forest_stats_{timestamp}.json")

    train_random_forest(train_df, test_df, config, model_path_rf, stats_collector, stats_path_rf)
    test_prob_logic(test_df, train_df, stats_collector, stats_path_prob)

if __name__ == "__main__":
    main()