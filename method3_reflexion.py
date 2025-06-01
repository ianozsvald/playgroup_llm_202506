import argparse
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime

# from litellm import completion
# import litellm
from dotenv import load_dotenv

import utils
from config import BREAK_IF_NOT_CHECKED_IN, providers
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt

# from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from prompt_feedback import feedback_on_executions
from run_code import execute_transform
from utils import add_argument_parser, extract_from_code_block, write_grid

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
disable_litellm_logging()


load_dotenv()

# DEBUG ENABLED FOR LITELLM
# litellm._turn_on_debug()


def run_reflexion(model, provider, llm_responses, template_name):
    # take the initial prompt and run it, then make a decision
    # make a prompt
    NBR_REFLEXIONS = 3  # how many times to run the reflexion loop
    # make the first descriptive prompt, we'll modify it in the loop
    func_dict = get_func_dict()
    prompt = make_prompt(template_name, problems, target="train", func_dict=func_dict)
    print("\n\n--------------------------------------")
    for reflexion_n in range(NBR_REFLEXIONS):
        print(f"\n\n----------REFLEXION START----------")
        print(f"----------REFLEXION {reflexion_n=}----------")
        print(f"\n\n\n\n\n\n----------------------------\nPROMPT:\n{prompt}")
        # input('Press Enter to continue...') # CAN HELP TO SEE WHAT'S GOING ON WITH THIS

        # make the initial prompt
        content = [{"type": "text", "text": prompt}]
        messages = [{"content": content, "role": "user"}]
        # print(f"{messages=}")
        response = call_llm(model, messages, provider)
        assert response is not None, "No response from LLM after retries"
        print("--------------\n")
        print("LLM response:")
        print(response.choices[0].message.content)
        print("--------------\n")
        llm_responses.append(response)

        # we get the _response_ which doesn't contain all the prior verbiage
        code_as_string = extract_from_code_block(response.choices[0].message.content)

        train_problems = problems["train"]
        rr_train = execute_transform(code_as_string, train_problems)
        # print(rr_train)

        if rr_train[0].transform_ran_and_matched_for_all_inputs:
            # we've succeeded
            return rr_train

        feedback = feedback_on_executions(rr_train)
        new_prompt = prompt + response.choices[0].message.content + feedback
        prompt = new_prompt

    return rr_train


def run_experiment(model, provider, iterations, problems, template_name):
    """Given a model"""
    llm_responses = []
    rr_trains = []
    for n in range(iterations):
        print(f"--------------------------\nPrompt {n}")
        rr_train = run_reflexion(model, provider, llm_responses, template_name)
        rr_trains.append(rr_train)

    return llm_responses, rr_trains


if __name__ == "__main__":
    if BREAK_IF_NOT_CHECKED_IN:
        utils.break_if_not_git_committed()

    parser = utils.add_argument_parser(
        problem_name=True, template_name=True, iterations=True, model_name=True
    )
    args = parser.parse_args()
    print(args)
    start_dt = datetime.now()

    utils.initial_log(logger, args)

    check_litellm_key(args)

    # load a single problem
    problems = utils.get_examples(args.problem_name)

    model = args.model_name

    llm_responses, rr_trains = run_experiment(
        model=model,
        provider=providers[args.model_name],
        iterations=args.iterations,
        problems=problems,
        template_name=args.template_name,
    )

    # show responses
    # print("Responses from LLM:")
    # print(
    #    "\n--\n".join(
    #        [response.choices[0].message.content for response in llm_responses]
    #    )
    # )

    # rr_trains[0][0].transform_ran_and_matched_for_all_inputs
    # rr_trains is the list of RunResult pairs for the training problems
    # each pair contains the RunResult and the grids initial/final/generated
    # rr is each instance of the pair so rr[0] is the RunResult
    # and we want to know how often transform_ran_and_matched_for_all_inputs
    # was True, i.e. how many runs were correct
    run_summary = f"Got {sum([rr[0].transform_ran_and_matched_for_all_inputs for rr in rr_trains])} of {len(rr_trains)} runs correct"
    logger.info(run_summary)
    print(run_summary)

    cnt_provider = Counter([response.provider for response in llm_responses])
    # print(f"Provider counts: {cnt_provider}")
    logger.info(f"{cnt_provider=}")

    all_token_usages = [
        llm_response.usage.total_tokens for llm_response in llm_responses
    ]
    print(f"Max token usage on a call was {max(all_token_usages)}")
    print(
        f"Median token usage on a call was {sorted(all_token_usages)[int(len(all_token_usages)/2)]}"
    )
    end_dt = datetime.now()
    dt_delta = end_dt - start_dt
    print(f"Experiment took {dt_delta}")
