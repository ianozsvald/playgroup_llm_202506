# GOAL
# Can you make it describe any image pair for a single problem?

import logging
import os
import sys
import time
from collections import Counter

# from litellm import completion
import litellm
from dotenv import load_dotenv

import utils
from config import BREAK_IF_NOT_CHECKED_IN, providers
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt
from run_code import execute_transform
from utils import encode_image_to_base64, extract_from_code_block

# from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template


logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
disable_litellm_logging()

load_dotenv()


if __name__ == "__main__":
    if BREAK_IF_NOT_CHECKED_IN:
        utils.break_if_not_git_committed()
    parser = utils.add_argument_parser(
        problem_name=True, template_name=True, iterations=True, model_name=True
    )
    args = parser.parse_args()
    print(args)
    check_litellm_key(args)
    utils.initial_log(logger, args)

    problems = utils.get_examples(args.problem_name)

    # HERE WE HARDCODE A SINGLE IMAGE-PAIR
    # YOU NEED TO REPLACE THIS WITH A FUNCTION TO MAKE IMAGES
    image_path = "test_images/small_problem_pair.png"
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    func_dict = get_func_dict()
    prompt = make_prompt(
        args.template_name, problems, target="train", func_dict=func_dict
    )

    print("PROMPT (printed once, as we iterate on the same prompt):")
    print(prompt)

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                # "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                "url": data_url
            },
        },
    ]
    messages = [{"content": content, "role": "user"}]
    # print(f"{messages=}")
    model = args.model_name

    llm_responses = []
    rr_trains = []

    provider = providers[args.model_name]

    response = call_llm(model, messages, provider)
    assert response is not None, "No response from LLM after retries"
    llm_responses.append(response)
    print(response)
    print(response.choices[0].message.content)

    # extract the code from the response
    # code_as_string = extract_from_code_block(response.choices[0].message.content)

    cnt_provider = Counter([response.provider for response in llm_responses])

    # rr_trains[0][0].transform_ran_and_matched_for_all_inputs
    print(
        f"Got {sum([rr_train[0].transform_ran_and_matched_for_all_inputs for rr in rr_trains])} of {len(rr_trains)} runs correct"
    )

    # show responses
    print(
        "\n--\n".join(
            [response.choices[0].message.content for response in llm_responses]
        )
    )

    print(f"Provider counts: {cnt_provider}")
