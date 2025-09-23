"""Configuration management for KnowHunt."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file and environment variables."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Default configuration
    config = {
        "collectors": {
            "arxiv": {
                "rate_limit": float(os.getenv("ARXIV_RATE_LIMIT", "1.0")),
                "max_results": int(os.getenv("ARXIV_MAX_RESULTS", "50"))
            }
        },
        "storage": {
            "type": os.getenv("STORAGE_TYPE", "postgresql"),
            "connection_string": os.getenv("DATABASE_URL", "postgresql://localhost/knowhunt")
        },
        "analysis": {
            "enable_nlp": os.getenv("ENABLE_NLP", "false").lower() == "true",
            "model_name": os.getenv("NLP_MODEL", "en_core_web_sm")
        },
        "api": {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
            "debug": os.getenv("API_DEBUG", "false").lower() == "true"
        }
    }
    
    # Load from YAML file if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config is None:
                        raise ValueError(f"Config file {config_path} is empty or invalid")
                    config = merge_configs(config, file_config)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_data_dir() -> Path:
    """Get data directory path."""
    data_dir = Path(os.getenv("KNOWHUNT_DATA_DIR", "data"))
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_log_dir() -> Path:
    """Get log directory path."""
    log_dir = Path(os.getenv("KNOWHUNT_LOG_DIR", "logs"))
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_cache_dir() -> Path:
    """Get cache directory path."""
    cache_dir = Path(os.getenv("KNOWHUNT_CACHE_DIR", "cache"))
    cache_dir.mkdir(exist_ok=True)
    return cache_dir