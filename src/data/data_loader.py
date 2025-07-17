import pandas as pd
import logging

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.data_loader")

def load_sudoku_data(file_path: str) -> pd.DataFrame:
    """
    Tải bộ dữ liệu Sudoku từ file CSV, tối ưu cho Google Colab.

    Args:
        file_path (str): Đường dẫn đến file CSV.

    Returns:
        pd.DataFrame: Bộ dữ liệu đã tải.

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại.
        ValueError: Nếu định dạng dữ liệu không hợp lệ.
    """
    try:
        logger.info(f"Tải bộ dữ liệu Sudoku từ {file_path}")
        df = pd.read_csv(file_path)

        # Kiểm tra cấu trúc DataFrame
        expected_columns = ["puzzle", "solution"]
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"Thiếu các cột bắt buộc trong bộ dữ liệu: {expected_columns}")
            raise ValueError(f"Bộ dữ liệu phải chứa các cột: {expected_columns}")

        logger.info(f"Tải bộ dữ liệu thành công từ {file_path}")
        return df

    except FileNotFoundError:
        logger.error(f"Không tìm thấy file: {file_path}")
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    except pd.errors.ParserError as e:
        logger.error(f"Định dạng CSV không hợp lệ: {str(e)}")
        raise ValueError(f"Định dạng CSV không hợp lệ: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi tải dữ liệu: {str(e)}")
        raise ValueError(f"Lỗi không xác định khi tải dữ liệu: {str(e)}")