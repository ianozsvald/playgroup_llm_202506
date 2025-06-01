import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from collections import namedtuple
from pathlib import PurePath

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filename="experiment.log",  # File to write logs to
    filemode="a",  # ISSUE if this is 'w' then the log only persists during run and then is deleted?!  # Append mode with a (use 'w' to overwrite)
)


def initial_log(logger, args):
    logger.info(f"{args=}")
    githash = os.popen("git rev-parse HEAD").read().strip()
    logger.info(f"{githash=}")


def get_examples(pattern_name):
    """Load arc examples, return the requested example"""
    # path = "/media/ian/data/llms/kaggle/202406_arc/ARC-AGI/data/training"
    path = "/media/ian/data/llms/kaggle/202503_arc_2025/arc-prize-2025"
    training_name = "arc-agi_training_challenges.json"
    filename = PurePath(path, training_name)
    # fully_filename = PurePath(path, pattern_name)
    # load the json
    with open(filename) as f:
        example = json.load(f)
    single_ex = example[pattern_name]
    return single_ex


def break_if_not_git_committed():
    """Exit if git not checked in"""
    if os.popen("git ls-files -m -o --exclude-from=.gitignore").read().strip() != "":
        print("!! UNCOMMITTED CHANGES !!")
        sys.exit(1)


RunResult = namedtuple(
    "run_result",
    [
        "code_did_execute",
        "code_ran_on_all_inputs",
        "transform_ran_and_matched_for_all_inputs",
        "transform_ran_and_matched_at_least_once",
        "transform_ran_and_matched_score",
    ],
)


class ExecutionOutcome:
    def __init__(self, initial, final, generated_final, was_correct):
        self.initial = np.array(initial)
        self.final = np.array(final)
        try:
            arr = np.array(generated_final)
            assert arr.ndim == 2
            self.generated_final = arr
        except (ValueError, AssertionError):
            # if the generated final is not a 2d grid, we can't make it an array
            self.generated_final = None
        self.was_correct = was_correct
        # generated_final_message = "..." or None?

    def __repr__(self):
        return f"""initial:\n{self.initial}\nfinal:\n{self.final}\ngenerated:
{self.generated_final}\nwas_correct: {self.was_correct}\n"""


def get_grid_size(grid):
    """Return e.g. (3,3) for grid size"""
    assert isinstance(grid, list)
    grid = np.array(grid)
    return grid.shape


def extract_from_code_block(text):
    "Extract the first code block in a text string"
    try:
        # result = re.search(r"```\s(.*?)\s```", text, re.DOTALL).group(1)
        # this also gets ///python
        re_groups = re.search(r"```[a-zA-Z]*\s(.*?)\s```", text, re.DOTALL)
        # group(0) is the whole match, group(1) is the first capture group
        result = re_groups.group(1)
    except AttributeError:
        result = None
    return result


def extract_json_from_response(response):
    try:
        container = json.loads(response)
    except json.decoder.JSONDecodeError:
        container = None
    return container


def make_list_of_lists(var, input):
    return f"{var} = {repr(input)}"


def parse_response_for_function(response):
    capture = False
    function_lines = []
    for line in response.split("\n"):
        if capture:
            if not line.startswith("    "):
                capture = False
        if line.startswith("def "):
            capture = True
        if capture:
            function_lines.append(line)
    return function_lines


def make_grid_plain(input):
    """Make 2D grid from list of lists"""
    return "\n".join(["".join(map(str, row)) for row in input])


def make_grid_csv(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    # e.g.
    # [[2 2 2]
    #  [2 1 8]
    #  [2 8 8]]
    return "\n".join([", ".join(map(str, row)) for row in input])


def make_grid_csv_english_words(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    # e.g.
    # one, one, one
    # eight, one, three
    # eight, two, two
    map_ints_to_words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }
    return "\n".join(
        [", ".join(map(lambda x: map_ints_to_words[x], row)) for row in input]
    )


def make_grid_csv_quoted(input):
    """Make 2D grid from list of lists, comma separated numbers"""
    return "\n".join([", ".join(map(lambda v: f'"{v}"', row)) for row in input])
    # return "\n".join([",".join(map(lambda v: f'"{v}"', row)) for row in input])
    # return "\n".join([", ".join(map(lambda v: f"'{v}'", row)) for row in input])


def write_grid(input):
    """Write out a 2D grid according to some rules..."""
    # print(type(input))
    # print(f"input: {input}")
    # return make_grid_csv_english_words(input)
    return make_grid_plain(input)


def add_argument_parser(
    problem_name=False,
    template_name=False,
    iterations=False,
    model_name=False,
    code_filename=False,
):
    parser = argparse.ArgumentParser(description=__doc__)  # read __doc__ attribute
    if problem_name:
        parser.add_argument(
            "-p",
            "--problem_name",
            type=str,
            nargs="?",
            help="name of an ARC AGI problem e.g. 9565186b"
            " (default: %(default)s))",  # help msg 2 over lines with default
            default="9565186b",
        )  # some default
    if template_name:
        parser.add_argument(
            "-t",
            "--template_name",
            type=str,
            nargs="?",
            help="template to use in ./templates/"
            " (default: %(default)s))",  # help msg 2 over lines with default
            default="prompt_baseline.txt",
        )
    if iterations:
        parser.add_argument(
            "-i",
            "--iterations",
            type=int,
            nargs="?",
            help="number of iterations to run",
            default=1,
        )
    if model_name:
        parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            nargs="?",
            help="openrouter model name (default: %(default)s)",
            # DeepSeek r3 March 2024 release (their latest)
            # https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free
            # default="openrouter/deepseek/deepseek-chat-v3-0324:free",
            # https://openrouter.ai/meta-llama/llama-4-scout/api
            default="openrouter/meta-llama/llama-4-scout",
        )
    if code_filename:
        parser.add_argument(
            "-c",
            "--code_filename",
            type=str,
            help="Name of the file e.g. /tmp/solution.py to run",
            default="/tmp/solution.py",
        )
    return parser


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
