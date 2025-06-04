import os
import sys
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from utils import add_argument_parser

if __name__ == "__main__":
    # NOTE litellm.py automagically loads .env file with the OPENROUTER_API_KEY
    try: 
        open('.env', 'r')
    except FileNotFoundError:
        print("Error: .env file not found. Please create a .env file with your OpenRouter API key.")
        sys.exit(1)
    parser = add_argument_parser(model_name=True)
    args = parser.parse_args()
    print(f"{args=}")
    try:
        check_litellm_key(args)
    except AssertionError as e:
        print(f"Error: {e}")
        print(f"{os.environ.get('OPENROUTER_API_KEY')=}")
        print("Please set the OPENROUTER_API_KEY in your .env file.")
        sys.exit(1)
    print("Successfully checked OpenRouter API key.")
    