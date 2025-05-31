import argparse
import logging
import os
import sys
import time
from collections import Counter

# from litellm import completion
# import litellm
from dotenv import load_dotenv

import utils
from config import BREAK_IF_NOT_CHECKED_IN, providers
from litellm_helper import call_llm, check_litellm_key, disable_litellm_logging
from prompt import get_func_dict, make_prompt
from run_code import execute_transform
from utils import add_argument_parser, extract_from_code_block, write_grid

# from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template


logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
disable_litellm_logging()


load_dotenv()

# DEBUG ENABLED FOR LITELLM
# litellm._turn_on_debug()


def feedback_on_prompt(rr_train):

    # rr_eo is the RunResult ExecutionOutcome
    # and since we have 2-4 train examples, we'll have 2-4 rr_eo results
    feedback_pieces = []
    for rr_eo in rr_train[1]:
        # rr_input_output is a triple of
        # initial (input)
        # final (expected)
        # generated_final - the result of running the code on initial
        initial_as_text = write_grid(rr_eo.initial)
        final_as_text = write_grid(rr_eo.final)
        generated_as_text = write_grid(rr_eo.generated_final)
        was_correct = rr_eo.was_correct
        if was_correct:
            feedback_direction = "You got this example CORRECT, please preserve your logic so that this continues to work"
        else:
            feedback_direction = "You got this example WRONG, please improve your logic and code to fix it"
        feedback = f"""
Given this input
{initial_as_text}
and the expected output of
{final_as_text}
your code generated
{generated_as_text}
{feedback_direction}
."""
        feedback_pieces.append(feedback)

    reflexion_feedback = "\n\n".join(feedback_pieces)

    # new_prompt = prompt
    # now add the response from the llm
    feedback += f"\n\nHere is some feedback on the execution of your code:\n{reflexion_feedback}\n"
    feedback += "Please explain first what you got wrong, and then how you could improve, and then write better code.\n"
    return feedback


def run_reflexion(model, provider, llm_responses, template_name):
    # take the initial prompt and run it, then make a decision
    # make a prompt
    NBR_REFLEXIONS = 3  # how many times to run the reflexion loop
    # make the first descriptive prompt, we'll modify it in the loop
    func_dict = get_func_dict()
    prompt = make_prompt(
        template_name, problems, target="train", func_dict=func_dict
    )
    print("\n\n--------------------------------------")
    for reflexion_n in range(NBR_REFLEXIONS):
        print(f"----------REFLEXION START----------")
        print(f"----------REFLEXION {reflexion_n=}----------")
        print(f"Prompt:\n{prompt}\n")

        # make the initial prompt
        content = [{"type": "text", "text": prompt}]
        messages = [{"content": content, "role": "user"}]
        # print(f"{messages=}")
        response = call_llm(model, messages, provider)
        assert response is not None, "No response from LLM after retries"
        print("LLM response:")
        print(response.choices[0].message.content)
        llm_responses.append(response)

        # we get the _response_ which doesn't contain all the prior verbiage
        code_as_string = extract_from_code_block(response.choices[0].message.content)

        train_problems = problems["train"]
        rr_train = execute_transform(code_as_string, train_problems)
        print(rr_train)

        if rr_train[0].transform_ran_and_matched_for_all_inputs:
            # we've succeeded
            return rr_train

        feedback = feedback_on_prompt(rr_train)
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
    print(
        "\n--\n".join(
            [response.choices[0].message.content for response in llm_responses]
        )
    )

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
