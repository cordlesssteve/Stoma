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
            "model_name": os.getenv("NLP_MODEL", "en_core_web_sm"),

            # Deep Research Integration
            "enable_deep_research": os.getenv("ENABLE_DEEP_RESEARCH", "true").lower() == "true",
            "deep_research": {
                "summarization_model": os.getenv("DR_SUMMARIZATION_MODEL", "openai:gpt-4.1-mini"),
                "research_model": os.getenv("DR_RESEARCH_MODEL", "openai:gpt-4.1"),
                "compression_model": os.getenv("DR_COMPRESSION_MODEL", "openai:gpt-4.1-mini"),
                "final_report_model": os.getenv("DR_FINAL_REPORT_MODEL", "openai:gpt-4.1"),
                "max_researcher_iterations": int(os.getenv("DR_MAX_ITERATIONS", "4")),
                "max_concurrent_research_units": int(os.getenv("DR_MAX_CONCURRENT", "3")),
                "max_react_tool_calls": int(os.getenv("DR_MAX_TOOL_CALLS", "10")),
                "allow_clarification": os.getenv("DR_ALLOW_CLARIFICATION", "false").lower() == "true",
                "search_api": os.getenv("DR_SEARCH_API", "tavily"),
                "max_tokens": int(os.getenv("DR_MAX_TOKENS", "8192")),
                "temperature": float(os.getenv("DR_TEMPERATURE", "0.1"))
            }
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