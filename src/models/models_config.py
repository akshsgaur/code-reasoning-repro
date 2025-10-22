#!/usr/bin/env python3
"""
Model configurations for LeetCode dataset evaluation
Supports: DeepSeek-R1, OpenAI (GPT-4o, GPT-4o-mini, o3-mini), Google Gemini
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a single model"""
    name: str
    identifier: str
    provider: str  # 'openai', 'deepseek', 'google'
    api_key_env: str  # Environment variable name for API key
    temperature: float = 0.0  # Default to 0 for deterministic output
    max_tokens: int = 2048
    supports_reasoning: bool = False  # For o3-mini, DeepSeek-R1, Gemini Deep Think


# Model configurations
MODELS = {
    # DeepSeek Models
    "deepseek-r1": ModelConfig(
        name="DeepSeek-R1",
        identifier="deepseek-ai/DeepSeek-R1",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        temperature=0.0,
        max_tokens=4096,
        supports_reasoning=True
    ),

    # OpenAI Models
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        identifier="gpt-4o-mini-2024-07-18",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=0.0,
        max_tokens=2048,
        supports_reasoning=False
    ),

    "gpt-4o": ModelConfig(
        name="GPT-4o",
        identifier="gpt-4o-2024-08-06",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=0.0,
        max_tokens=2048,
        supports_reasoning=False
    ),

    "o3-mini": ModelConfig(
        name="o3-mini",
        identifier="o3-mini-2025-01-31",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        temperature=1.0,  # o3-mini uses temperature for reasoning
        max_tokens=4096,
        supports_reasoning=True
    ),

    # Google Gemini Models
    "gemini-2.5-pro": ModelConfig(
        name="Gemini-2.5-Pro",
        identifier="gemini-2.5-pro",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        temperature=0.0,
        max_tokens=2048,
        supports_reasoning=True  # Deep Think capability
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model"""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]


def list_available_models() -> List[str]:
    """List all available model names"""
    return list(MODELS.keys())


def get_models_by_provider(provider: str) -> Dict[str, ModelConfig]:
    """Get all models from a specific provider"""
    return {k: v for k, v in MODELS.items() if v.provider == provider}
