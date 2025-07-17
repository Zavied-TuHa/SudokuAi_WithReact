import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import time
import os
import numpy as np
from src.utils.logger import setup_logging
from src.utils.config_parser import load_config
from src.solvers import RuleBasedSolver, ProbabilisticLogicSolver, RandomForestSolver
from src.data.preprocessor import string_to_grid, grid_to_string
from src.utils.stats_collector import StatsCollector
from src.utils.sudoku_utils import is_valid_grid

logger = logging.getLogger("sudoku_ai.main")
app = FastAPI()

if os.getenv("ENV") == "production":
    app.mount("/", StaticFiles(directory="web/dist", html=True), name="static")

class SolvePuzzleRequest(BaseModel):
    """
    Mô hình Pydantic để xác thực yêu cầu giải bài toán.
    """
    puzzle: str
    algorithm: str

@app.post("/api/solve")
async def solve_puzzle_api(request: SolvePuzzleRequest):
    """
    Điểm cuối API để giải một bài toán Sudoku.

    Args:
        request (SolvePuzzleRequest): Yêu cầu chứa chuỗi puzzle và thuật toán.

    Returns:
        dict: Thống kê, kết quả đã giải, và thông điệp.

    Raises:
        HTTPException: Nếu đầu vào không hợp lệ hoặc solver thất bại.
    """
    try:
        logger.info(f"Nhận yêu cầu giải với thuật toán: {request.algorithm}")
        config = load_config()
        stats_collector = StatsCollector()

        if len(request.puzzle) != 81 or not all(c in '0123456789.' for c in request.puzzle):
            logger.error("Định dạng puzzle không hợp lệ")
            raise HTTPException(status_code=400, detail="Puzzle phải là chuỗi 81 ký tự (0-9 hoặc '.')")

        puzzle_grid = string_to_grid(request.puzzle)
        if not is_valid_grid(puzzle_grid, allow_empty=True):
            logger.error("Bài toán Sudoku không hợp lệ")
            raise HTTPException(status_code=400, detail="Bài toán Sudoku không hợp lệ")

        solver_map = {
            "rule-based": RuleBasedSolver(),
            "prob-logic": ProbabilisticLogicSolver(),
            "random-forest": RandomForestSolver(os.path.join(config["data"]["model_save_path"], "random_forest_model.pkl"))
        }
        solver = solver_map.get(request.algorithm)
        if not solver:
            logger.error(f"Thuật toán không hợp lệ: {request.algorithm}")
            raise HTTPException(status_code=400, detail=f"Thuật toán không hợp lệ: {request.algorithm}")

        start_time = time.time()
        result = solver.solve(puzzle_grid)
        solve_time = time.time() - start_time

        message = "Giải bài toán thành công!"
        is_complete = not any(0 in row for row in result)
        is_valid = is_valid_grid(result, allow_empty=(request.algorithm != "rule-based"))
        if request.algorithm == "rule-based" and (not is_complete or not is_valid):
            message = "Không thể giải bài toán hoàn chỉnh!"

        solver.collect_stats(stats_collector, puzzle_grid, result, result, solve_time)

        stats_path = os.path.join(config["data"]["stats_save_path"], f"{request.algorithm}_stats.json")
        await stats_collector.save_stats(stats_path)

        return {
            "metrics": {
                "accuracy": stats_collector.get_stats().get(request.algorithm.replace('-', '_'), {}).get("accuracy", 1.0),
                "solve_time": solve_time,
                "memory_usage": stats_collector.get_memory_usage()
            },
            "stats": stats_collector.get_stats(),
            "sample_puzzle": request.puzzle,
            "solved_puzzle": grid_to_string(result).replace('0', '.') if result else request.puzzle,
            "message": message
        }

    except ValueError as e:
        logger.error(f"Lỗi giá trị: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi không xác định khi giải bài toán: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi không xác định: {str(e)}")

def main():
    """
    Hàm chính để khởi động server FastAPI.
    """
    setup_logging()
    config = load_config()
    logger.info("Khởi động ứng dụng Sudoku AI")
    port = config["ui"]["port"] if os.getenv("ENV") != "production" else 80
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()