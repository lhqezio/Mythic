"""
Custom exceptions for the LLM system.
"""

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMInitializationError(LLMError):
    """Raised when LLM initialization fails"""
    pass


class LLMInferenceError(LLMError):
    """Raised when LLM inference fails"""
    pass
