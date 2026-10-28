import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


def load_model():
    """
    Create and return the Groq client.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in the environment."
        )

    return Groq(api_key=api_key)
