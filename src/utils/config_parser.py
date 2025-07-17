import yaml
import logging
from typing import Dict, Any

# Configure logger
logger = logging.getLogger("sudoku_ai.config_parser")

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or invalid.
    """
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            logger.error("Configuration file is empty")
            raise ValueError("Configuration file is empty")
        logger.info("Configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {str(e)}")
        raise ValueError(f"Error parsing config file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error loading config: {str(e)}")
        raise ValueError(f"Unexpected error loading config: {str(e)}")