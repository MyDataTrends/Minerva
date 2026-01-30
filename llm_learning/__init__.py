"""
LLM Learning System - Enable LLMs to improve over time.

Components:
- example_store: Store and retrieve successful interactions
- interaction_logger: Log all chat/action interactions  
- embeddings: Vector similarity for RAG
- training_formatter: Export data for fine-tuning
"""

from llm_learning.example_store import ExampleStore, get_example_store
from llm_learning.interaction_logger import InteractionLogger, get_interaction_logger
