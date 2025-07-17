import pandas as pd
import logging
from typing import Tuple
from src.utils.config_parser import load_config
from src.utils.file_utils import ensure_directory

# Cấu hình logger
logger = logging.getLogger("sudoku_ai.data_processor")

def clean_data(file_path: str, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Làm sạch và chia bộ dữ liệu Sudoku thành tập huấn luyện và kiểm tra (9:1).

    Args:
        file_path (str): Đường dẫn đến file CSV Sudoku.
        config (dict): Từ điển cấu hình với ngưỡng độ khó.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tập huấn luyện và kiểm tra.

    Raises:
        FileNotFoundError: Nếu file đầu vào không tồn tại.
        ValueError: Nếu không còn dữ liệu hợp lệ sau khi làm sạch.
    """
    try:
        logger.info(f"Làm sạch dữ liệu từ {file_path}")
        df = pd.read_csv(file_path)

        # Kiểm tra và làm sạch dữ liệu
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Loại bỏ {initial_rows - len(df)} hàng có giá trị thiếu")

        # Xác thực định dạng puzzle và solution
        def validate_row(row):
            puzzle = row['puzzle']
            solution = row['solution']
            if not isinstance(puzzle, str) or not isinstance(solution, str):
                return False
            if len(puzzle) != 81 or len(solution) != 81:
                return False
            if not all(c in '0123456789.' for c in puzzle) or not all(c in '123456789' for c in solution):
                return False
            return True

        df = df[df.apply(validate_row, axis=1)]
        logger.info(f"Còn {len(df)} hàng hợp lệ sau khi xác thực")

        if len(df) == 0:
            logger.error("Không còn dữ liệu hợp lệ sau khi làm sạch")
            raise ValueError("Không còn dữ liệu hợp lệ sau khi làm sạch")

        # Chuẩn hóa puzzle (thay '0' bằng '.')
        df['puzzle'] = df['puzzle'].str.replace('0', '.', regex=False)

        # Thêm số lượng gợi ý nếu chưa có
        if 'clues' not in df.columns:
            df['clues'] = df['puzzle'].apply(lambda x: sum(1 for c in x if c != '.'))

        # Chuẩn hóa độ khó
        def map_difficulty(clues: int) -> str:
            difficulty_levels = config["solver"]["difficulty_levels"]
            if clues > difficulty_levels["easy"]["min_clues"]:
                return "dễ"
            elif clues >= difficulty_levels["medium"]["min_clues"]:
                return "trung bình"
            else:
                return "khó"

        df['difficulty'] = df['clues'].apply(map_difficulty)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra (9:1)
        train_df = df.sample(frac=0.9, random_state=42)
        test_df = df.drop(train_df.index)

        logger.info(f"Chia dữ liệu: {len(train_df)} hàng huấn luyện, {len(test_df)} hàng kiểm tra")
        return train_df, test_df

    except FileNotFoundError:
        logger.error(f"File đầu vào không tồn tại: {file_path}")
        raise FileNotFoundError(f"File đầu vào không tồn tại: {file_path}")
    except Exception as e:
        logger.error(f"Lỗi không xác định khi làm sạch dữ liệu: {str(e)}")
        raise ValueError(f"Lỗi không xác định khi làm sạch dữ liệu: {str(e)}")