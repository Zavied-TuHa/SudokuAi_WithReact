import logging
import logging.config
import yaml

# Configure logger
logger = logging.getLogger("sudoku_ai")

def setup_logging(config_path: str = "config/logging_config.yaml") -> None:
    """
    Set up logging configuration from a YAML file.

    Args:
        config_path (str): Path to the logging configuration YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is empty or invalid.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not config:
            logger.error("Logging configuration file is empty")
            raise ValueError("Logging configuration file is empty")
        logging.config.dictConfig(config)
        logger.info("Logging configured successfully")
    except FileNotFoundError:
        logger.error(f"Logging config file not found: {config_path}")
        raise FileNotFoundError(f"Logging config file not found: {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing logging config: {str(e)}")
        raise ValueError(f"Error parsing logging config: {str(e)}")
    except Exception as e:
        logger.error(f"Error setting up logging: {str(e)}")
        raise ValueError(f"Error setting up logging: {str(e)}")