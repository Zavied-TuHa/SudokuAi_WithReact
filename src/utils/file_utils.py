import os
import logging
import asyncio
from typing import List

# Configure logger
logger = logging.getLogger("sudoku_ai.file_utils")

async def write_file_async(file_path: str, content: str) -> None:
    """
    Asynchronously write content to a file to avoid I/O blocking.

    Args:
        file_path (str): Path to the file.
        content (str): Content to write.

    Raises:
        IOError: If writing to file fails.
    """
    try:
        logger.debug(f"Writing to file: {file_path}")
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            f.write(content)
        logger.debug(f"Successfully wrote to {file_path}")
    except Exception as e:
        logger.error(f"Error writing to file {file_path}: {str(e)}")
        raise IOError(f"Error writing to file {file_path}: {str(e)}")

def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory (str): Path to the directory.

    Raises:
        OSError: If directory creation fails.
    """
    try:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        raise OSError(f"Error creating directory {directory}: {str(e)}")