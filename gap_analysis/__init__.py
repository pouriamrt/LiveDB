"""Gap analysis package — shared resources."""

from openai import AsyncOpenAI

from Config import config

# Single shared AsyncOpenAI client for all gap_analysis modules
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
