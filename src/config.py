import os
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class Configuration:
    # Get the path from an environment variable
    env_path = os.path.expanduser("~/Documents/pw/.env")
    load_dotenv(env_path)  # Load the .env file

    openai_api_key = os.getenv("API_KEY")  # your api key for openai
    gpt_version = "gpt-3.5-turbo"  # "gpt-4o" # "gpt-3.5-turbo-0125"  # "gpt-4-turbo-preview"  # "gpt-4-1106-preview"

    token_limit = 6000  # token limits
    temperature = 0.6  # temperature for the LLM
    path_to_story = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "stories", "example"
    )
    # the following are debug parameters
    mockup_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mockups", "describer_example", "json"
    )
    debug = True
    save_traffic = True
    reasoning = True
