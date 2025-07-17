import json
import psutil
import os
from typing import Dict, Any
import logging
from src.utils.file_utils import write_file_async

# Configure logger
logger = logging.getLogger("sudoku_ai.stats_collector")

class StatsCollector:
    """
    Class to collect and serialize solver statistics.
    """
    def __init__(self):
        """Initialize the StatsCollector with dictionaries for different solver stats."""
        self.stats = {
            "general": {
                "solve_times": [],
                "memory_usage": 0.0,
                "invalid_solution": 0,
                "mismatch_solution": 0,
                "empty_cells": 0,
                "solver_error": 0
            },
            "rule_based": {},
            "prob_logic": {},
            "rf": {}
        }
        self.process = psutil.Process(os.getpid())

    def add_stat(self, key: str, value: Any) -> None:
        """
        Add a general statistic.

        Args:
            key (str): Statistic name.
            value (Any): Statistic value.
        """
        if key in self.stats["general"]:
            self.stats["general"][key] += value
        else:
            self.stats["general"][key] = value
        logger.debug(f"Added general stat: {key} = {value}")

    def add_solve_time(self, solve_time: float) -> None:
        """
        Add solve time to general stats.

        Args:
            solve_time (float): Time taken to solve a puzzle.
        """
        self.stats["general"]["solve_times"].append(solve_time)
        logger.debug(f"Added solve time: {solve_time:.4f}s")

    def add_rule_based_stat(self, key: str, value: Any) -> None:
        """
        Add a rule-based solver statistic.

        Args:
            key (str): Statistic name.
            value (Any): Statistic value.
        """
        self.stats["rule_based"][key] = value
        logger.debug(f"Added rule-based stat: {key} = {value}")

    def add_prob_logic_stat(self, key: str, value: Any) -> None:
        """
        Add a probabilistic logic solver statistic.

        Args:
            key (str): Statistic name.
            value (Any): Statistic value.
        """
        self.stats["prob_logic"][key] = value
        logger.debug(f"Added prob-logic stat: {key} = {value}")

    def add_rf_stat(self, key: str, value: Any) -> None:
        """
        Add a Random Forest solver statistic.

        Args:
            key (str): Statistic name.
            value (Any): Statistic value.
        """
        if key in self.stats["rf"]:
            self.stats["rf"][key] += value
        else:
            self.stats["rf"][key] = value
        logger.debug(f"Added Random Forest stat: {key} = {value}")

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            float: Memory usage in MB.
        """
        try:
            mem = self.process.memory_info().rss / 1024 / 1024
            self.stats["general"]["memory_usage"] = mem
            return mem
        except Exception as e:
            logger.error(f"Error measuring memory usage: {str(e)}")
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get all collected statistics.

        Returns:
            Dict[str, Any]: Dictionary of all statistics.
        """
        return self.stats

    async def save_stats(self, file_path: str) -> None:
        """
        Save statistics to a JSON file asynchronously.

        Args:
            file_path (str): Path to save the stats.

        Raises:
            IOError: If saving stats fails.
        """
        try:
            logger.info(f"Saving stats to {file_path}")
            await write_file_async(file_path, json.dumps(self.stats, indent=2))
            logger.info("Stats saved successfully")
        except Exception as e:
            logger.error(f"Error saving stats: {str(e)}")
            raise IOError(f"Error saving stats: {str(e)}")