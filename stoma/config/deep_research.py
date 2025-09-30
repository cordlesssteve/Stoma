"""Deep Research configuration utilities for Stoma."""

from typing import Dict, Any, Optional
from ..integrations.deep_research_bridge import DeepResearchConfig


def build_deep_research_config_from_settings(settings: Dict[str, Any]) -> DeepResearchConfig:
    """
    Build DeepResearchConfig from Stoma settings.

    Args:
        settings: Stoma configuration dictionary

    Returns:
        Configured DeepResearchConfig instance
    """
    dr_settings = settings.get("analysis", {}).get("deep_research", {})

    return DeepResearchConfig(
        summarization_model=dr_settings.get("summarization_model", "openai:gpt-4.1-mini"),
        research_model=dr_settings.get("research_model", "openai:gpt-4.1"),
        compression_model=dr_settings.get("compression_model", "openai:gpt-4.1-mini"),
        final_report_model=dr_settings.get("final_report_model", "openai:gpt-4.1"),
        max_researcher_iterations=dr_settings.get("max_researcher_iterations", 4),
        max_concurrent_research_units=dr_settings.get("max_concurrent_research_units", 3),
        max_react_tool_calls=dr_settings.get("max_react_tool_calls", 10),
        allow_clarification=dr_settings.get("allow_clarification", False),
        search_api=dr_settings.get("search_api", "tavily"),
        max_tokens=dr_settings.get("max_tokens", 8192),
        temperature=dr_settings.get("temperature", 0.1)
    )


def is_deep_research_enabled(settings: Dict[str, Any]) -> bool:
    """
    Check if deep research is enabled in settings.

    Args:
        settings: Stoma configuration dictionary

    Returns:
        True if deep research is enabled
    """
    return settings.get("analysis", {}).get("enable_deep_research", True)


def validate_deep_research_config(config: DeepResearchConfig) -> Dict[str, str]:
    """
    Validate deep research configuration.

    Args:
        config: DeepResearchConfig to validate

    Returns:
        Dictionary of validation errors (empty if valid)
    """
    errors = {}

    # Check model format
    models = [
        ("summarization_model", config.summarization_model),
        ("research_model", config.research_model),
        ("compression_model", config.compression_model),
        ("final_report_model", config.final_report_model)
    ]

    for model_name, model_value in models:
        if not model_value or ":" not in model_value:
            errors[model_name] = f"Invalid model format: {model_value}. Expected 'provider:model' format."

    # Check numeric values
    if config.max_researcher_iterations <= 0:
        errors["max_researcher_iterations"] = "Must be greater than 0"

    if config.max_concurrent_research_units <= 0:
        errors["max_concurrent_research_units"] = "Must be greater than 0"

    if config.max_react_tool_calls <= 0:
        errors["max_react_tool_calls"] = "Must be greater than 0"

    if config.max_tokens <= 0:
        errors["max_tokens"] = "Must be greater than 0"

    if not (0.0 <= config.temperature <= 2.0):
        errors["temperature"] = "Must be between 0.0 and 2.0"

    # Check search API
    valid_search_apis = ["tavily", "openai", "anthropic", "none"]
    if config.search_api not in valid_search_apis:
        errors["search_api"] = f"Must be one of: {', '.join(valid_search_apis)}"

    return errors