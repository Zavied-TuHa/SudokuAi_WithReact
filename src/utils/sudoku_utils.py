import logging
from typing import List, Optional

# Configure logger
logger = logging.getLogger("sudoku_ai.sudoku_utils")

def is_valid_move(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    """
    Check if placing a number in the given cell is valid.

    Args:
        grid (List[List[int]]): 9x9 Sudoku grid.
        row (int): Row index.
        col (int): Column index.
        num (int): Number to place.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    try:
        # Check row
        if num in grid[row]:
            logger.debug(f"Invalid move: {num} already in row {row}")
            return False
        # Check column
        if num in [grid[i][col] for i in range(9)]:
            logger.debug(f"Invalid move: {num} already in column {col}")
            return False
        # Check 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == num:
                    logger.debug(f"Invalid move: {num} already in subgrid ({start_row}, {start_col})")
                    return False
        return True
    except Exception as e:
        logger.error(f"Error checking move validity: {str(e)}")
        return False

def is_valid_grid(grid: List[List[int]], allow_empty: bool = True) -> bool:
    """
    Check if the Sudoku grid is valid.

    Args:
        grid (List[List[int]]): 9x9 Sudoku grid.
        allow_empty (bool): Allow empty cells (0) if True.

    Returns:
        bool: True if the grid is valid, False otherwise.

    Raises:
        ValueError: If grid dimensions or values are invalid.
    """
    try:
        if not isinstance(grid, list) or len(grid) != 9 or any(len(row) != 9 for row in grid):
            logger.error("Invalid grid dimensions: Expected 9x9")
            raise ValueError("INVALID_GRID_DIMENSIONS: Grid must be 9x9")
        for row in range(9):
            for col in range(9):
                num = grid[row][col]
                if num == 0 and not allow_empty:
                    logger.error(f"Empty cell found at ({row}, {col}) when not allowed")
                    raise ValueError(f"INVALID_GRID_STATE: Empty cell at ({row}, {col})")
                if num == 0:
                    continue
                if not isinstance(num, int) or num < 1 or num > 9:
                    logger.error(f"Invalid value {num} at ({row}, {col})")
                    raise ValueError(f"INVALID_GRID_VALUE: Value {num} at ({row}, {col})")
                # Temporarily remove the number to check its validity
                temp = grid[row][col]
                grid[row][col] = 0
                if not is_valid_move(grid, row, col, temp):
                    grid[row][col] = temp
                    logger.debug(f"Invalid number {temp} at ({row}, {col})")
                    return False
                grid[row][col] = temp
        logger.debug("Grid is valid")
        return True
    except ValueError as e:
        logger.error(f"Grid validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error checking grid validity: {str(e)}")
        raise ValueError(f"UNEXPECTED_ERROR: {str(e)}")