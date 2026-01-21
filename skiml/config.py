"""Configuration for Gemini ReAct Agent."""
import os
from google import genai
from google.api_core import retry


def setup_retry_policy():
    """Setup retry policy for API calls."""
    is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
    genai.models.Models.generate_content = retry.Retry(
        predicate=is_retriable)(genai.models.Models.generate_content)


def get_api_key():
    """Get Google API key from environment or secrets."""
    # Try environment variable first
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        try:
            # Try Kaggle secrets if available
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        except ImportError:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it as an environment variable "
                "or configure Kaggle secrets."
            )
    
    return api_key
