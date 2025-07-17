import numpy as np
import logging
from typing import List, Union
from src.utils.sudoku_utils import is_valid_move

# Configure logger
logger = logging.getLogger("sudoku_ai.preprocessor")

def string_to_grid(puzzle: str) -> List[List[int]]:
    """
    Chuyển đổi chuỗi puzzle thành lưới 9x9.

    Args:
        puzzle (str): Chuỗi 81 ký tự (0-9 hoặc '.' cho ô trống).

    Returns:
        List[List[int]]: Lưới 9x9 biểu diễn bài toán Sudoku.

    Raises:
        ValueError: Nếu chuỗi puzzle không hợp lệ.
    """
    if not isinstance(puzzle, str):
        logger.error(f"Puzzle phải là chuỗi, nhận được {type(puzzle)}")
        raise ValueError("Puzzle phải là chuỗi")
    if len(puzzle) != 81:
        logger.error(f"Độ dài puzzle không hợp lệ: {len(puzzle)}, cần 81")
        raise ValueError("Puzzle phải dài 81 ký tự")
    if not all(c in '0123456789.' for c in puzzle):
        logger.error("Puzzle chứa ký tự không hợp lệ")
        raise ValueError("Puzzle chứa ký tự không hợp lệ")

    try:
        puzzle = puzzle.replace(".", "0")
        grid = [[int(puzzle[i * 9 + j]) for j in range(9)] for i in range(9)]
        logger.debug("Chuyển đổi chuỗi puzzle thành lưới thành công")
        return grid
    except ValueError as e:
        logger.error(f"Định dạng puzzle không hợp lệ: {str(e)}")
        raise ValueError(f"Định dạng puzzle không hợp lệ: {str(e)}")

def grid_to_string(grid: List[List[int]]) -> str:
    """
    Chuyển đổi lưới 9x9 thành chuỗi puzzle.

    Args:
        grid (List[List[int]]): Lưới 9x9 biểu diễn bài toán Sudoku.

    Returns:
        str: Chuỗi puzzle 81 ký tự.

    Raises:
        ValueError: Nếu lưới không hợp lệ.
    """
    if not isinstance(grid, list) or len(grid) != 9 or any(len(row) != 9 for row in grid):
        logger.error("Kích thước lưới không hợp lệ, cần 9x9")
        raise ValueError("Lưới phải là 9x9")
    if not all(all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row) for row in grid):
        logger.error("Lưới chứa giá trị không hợp lệ")
        raise ValueError("Giá trị lưới phải là số nguyên từ 0 đến 9")

    puzzle = "".join(str(cell) for row in grid for cell in row)
    logger.debug("Chuyển đổi lưới thành chuỗi puzzle thành công")
    return puzzle

def preprocess_puzzle(puzzle: Union[str, List[List[int]]], for_ml: bool = False) -> np.ndarray:
    """
    Tiền xử lý bài toán Sudoku cho solver hoặc mô hình ML.

    Args:
        puzzle (Union[str, List[List[int]]]): Bài toán dưới dạng chuỗi hoặc lưới 9x9.
        for_ml (bool): Nếu True, tiền xử lý cho mô hình ML (làm phẳng và thêm đặc trưng).

    Returns:
        np.ndarray: Mảng numpy biểu diễn bài toán đã tiền xử lý.

    Raises:
        ValueError: Nếu puzzle không hợp lệ.
    """
    try:
        if isinstance(puzzle, str):
            grid = string_to_grid(puzzle)
        elif isinstance(puzzle, list):
            grid = puzzle
            grid_to_string(grid)  # Kiểm tra tính hợp lệ
        else:
            logger.error(f"Puzzle phải là chuỗi hoặc lưới 9x9, nhận được {type(puzzle)}")
            raise ValueError("Puzzle phải là chuỗi hoặc lưới 9x9")

        grid_array = np.array(grid, dtype=np.int32)
        if for_ml:
            features = []
            for row in range(9):
                for col in range(9):
                    cell_value = grid_array[row, col]
                    # Mã hóa ô trống thành -1
                    cell_feature = -1 if cell_value == 0 else cell_value
                    # Đếm số ô trống trong hàng, cột, ô 3x3
                    row_counts = sum(1 for c in range(9) if grid_array[row, c] == 0)
                    col_counts = sum(1 for r in range(9) if grid_array[r, col] == 0)
                    subgrid_counts = sum(1 for r in range(3 * (row // 3), 3 * (row // 3) + 3)
                                        for c in range(3 * (col // 3), 3 * (col // 3) + 3)
                                        if grid_array[r, c] == 0)
                    # Số giá trị khả thi cho ô
                    possible_values = len([num for num in range(1, 10) if is_valid_move(grid_array, row, col, num)])
                    # Vị trí ô
                    row_pos = row
                    col_pos = col
                    features.extend([cell_feature, row_counts, col_counts, subgrid_counts, possible_values, row_pos, col_pos])
            grid_array = np.array(features, dtype=np.int32)
            logger.debug("Tiền xử lý puzzle cho mô hình ML với đặc trưng bổ sung")
        else:
            grid_array = grid_array.flatten()
            logger.debug("Tiền xử lý puzzle cho solver")
        return grid_array
    except Exception as e:
        logger.error(f"Lỗi khi tiền xử lý puzzle: {str(e)}")
        raise ValueError(f"Lỗi khi tiền xử lý puzzle: {str(e)}")